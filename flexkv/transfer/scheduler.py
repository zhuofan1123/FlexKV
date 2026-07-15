from typing import Dict, List, Tuple
from collections import OrderedDict

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType


class TransferScheduler:
    """Incremental, foreground-priority transfer graph scheduler.

    (A) Incremental scheduling (PR #235). A graph can only expose new ready ops
        when it is newly added or when one of its OWN ops just finished --
        otherwise its ready set is byte-identical to last wake. So each wake we
        process only the "dirty" set (newly-added graphs UNION graphs owning a
        finished op) instead of sweeping every in-flight graph: O(changed), not
        O(backlog). This kills the positive-feedback stall (the P999 spike) that
        the old full sweep hit under a submission burst. Graphs are mutated
        solely by the single TE scheduler thread, so no locking is needed.

    (B) Foreground priority. engine GET/PUT transfers (H2D/D2H -- a request is
        synchronously waiting) go in a foreground bucket; external prefetch
        transfers (DISK2H -- fire-and-forget warmup, nothing waits) go in a
        background bucket. Each wake we drain the foreground dirty set BEFORE the
        background one, so a prefetch burst can't push engine H2D ops to the back
        of the FIFO. next_ops keeps all foreground ops (in graph insertion order)
        ahead of all background ops.

    Standing requirement (A): any path that completes an op or otherwise changes
    a graph's readiness MUST dirty that graph, else it is never revisited.
    """

    def __init__(self) -> None:
        # Foreground = engine (H2D/D2H); background = external prefetch (DISK2H).
        self._fg: "OrderedDict[int, TransferOpGraph]" = OrderedDict()
        self._bg: "OrderedDict[int, TransferOpGraph]" = OrderedDict()
        # graph_id -> is_prefetch: doubles as the membership oracle for a
        # finished op (which bucket) and is popped in lockstep with the graph.
        self._graph_bucket: Dict[int, bool] = {}
        # Graph ids dirtied since the last schedule(), insertion-ordered (dict
        # with None values == ordered set: O(1) add + dedup, preserves the
        # original per-bucket iteration order).
        self._dirty_fg: "OrderedDict[int, None]" = OrderedDict()
        self._dirty_bg: "OrderedDict[int, None]" = OrderedDict()

    def add_transfer_graph(self, graph: TransferOpGraph, is_prefetch: bool = False) -> None:
        """Add a new transfer graph. is_prefetch routes it to the background
        bucket (deprioritized); default False keeps engine transfers foreground."""
        gid = graph.graph_id
        if is_prefetch:
            self._bg[gid] = graph
            self._dirty_bg[gid] = None
        else:
            self._fg[gid] = graph
            self._dirty_fg[gid] = None
        self._graph_bucket[gid] = is_prefetch

    def _drain_bucket(self,
                      dirty: "OrderedDict[int, None]",
                      graphs: "OrderedDict[int, TransferOpGraph]",
                      next_ops: List[TransferOp],
                      completed_graph_ids: List[int]) -> None:
        """Drain one bucket's dirty set with revisit-on-raise discipline.

        Peek at the head and drop the dirty bit only once the graph is fully
        processed, so a raise below leaves the id dirty and the graph gets
        revisited -- the caller (TransferEngine._scheduler_loop) just logs and
        keeps looping, so a dropped bit would strand that graph for good. This
        recovers the dirty BIT, not the work: ops already collected into next_ops
        are discarded with the exception, exactly as under the full sweep.

        VIRTUAL ops self-complete (they carry no data) and that can unblock
        same-graph successors, so they re-queue the graph for another pass.
        """
        while dirty:
            graph_id = next(iter(dirty))
            # Defensive: every dirtied id should resolve, since the only writers
            # pair the bucket and its dirty set. Skipping beats a KeyError the
            # caller would swallow then retry forever on the same id.
            graph = graphs.get(graph_id)
            revisit = False
            if graph is not None:
                for op_id in graph.take_ready_ops():
                    op = graph._op_map[op_id]
                    if op.transfer_type == TransferType.VIRTUAL:
                        graph.mark_completed(op_id)
                        revisit = True
                    next_ops.append(op)
                if graph.all_transfer_ops_completed():
                    completed_graph_ids.append(graph_id)
                    del graphs[graph_id]
                    self._graph_bucket.pop(graph_id, None)
                    # A finished graph needs no further pass -- the common
                    # terminal-virtual-sink case both set revisit and completes,
                    # so clearing it here keeps the re-queue off the hot path.
                    revisit = False
            del dirty[graph_id]
            if revisit:
                dirty[graph_id] = None  # re-queue at the tail

    def schedule(self,
                finished_ops: List[TransferOp]
               ) -> Tuple[List[int], List[TransferOp]]:
        """Schedule transfer operations (incremental, foreground-first).

        Returns:
            Tuple[List[int], List[TransferOp]]:
                - completed transfer graph ids (foreground first)
                - next executable transfer ops (all foreground before background,
                  each in graph insertion order)
        """
        # Mark completed ops. Dirty the graph BEFORE completing the op:
        # mark_completed() clears the op from its successors' predecessor sets in
        # a loop, so a raise partway leaves the graph half-advanced and needing a
        # revisit -- free insurance for that partial-mutation case.
        for op in finished_ops:
            is_pref = self._graph_bucket.get(op.graph_id)
            if is_pref is None:
                continue  # graph already completed/removed
            (self._dirty_bg if is_pref else self._dirty_fg)[op.graph_id] = None
            (self._bg if is_pref else self._fg)[op.graph_id].mark_completed(op.op_id)

        # Drain foreground before background; each bucket keeps HEAD's per-graph
        # peek-head / revisit-on-raise discipline and removes completed graphs
        # inline.
        next_ops: List[TransferOp] = []
        completed_graph_ids: List[int] = []
        self._drain_bucket(self._dirty_fg, self._fg, next_ops, completed_graph_ids)
        self._drain_bucket(self._dirty_bg, self._bg, next_ops, completed_graph_ids)

        return completed_graph_ids, next_ops
