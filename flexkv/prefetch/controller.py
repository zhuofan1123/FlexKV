# SPDX-License-Identifier: Apache-2.0
"""
External prefetch controller for the FlexKV radix-shmem path.

An external process (e.g. the process that drives requests to a vLLM server)
can run a `PrefetchController` to warm the shared KV cache ahead of the real
request: it attaches to the *same* radix-shmem index that internal FlexKV DP
schedulers use, generates a CPU-only upload TransferOpGraph via the shared cache
engine, and submits it to the *single shared* TransferEngine (TE) process over a
reserved shm channel.

Why this is safe to run alongside internal FlexKV:

  * **Shared index**: attaches to the CPU/SSD radix regions as
    `shmradix.RadixClient`s (via
    `flexkv.cache.radix_shmem_engine.CacheEngineRadixShmem`). This process is
    just another client.
  * **Slot safety**: CPU/SSD slot allocation goes through shmradix's
    process-shared mempool (`allocate_slots`/`recycle_slots`), so this process
    cannot collide slots with internal FlexKV — the allocator is the single
    shared authority.
  * **Completion isolation**: the TE dispatcher routes each `CompletedOp` back
    to the submitting channel purely by `graph_id`. We claim a disjoint
    `graph_id`/`op_id` range (beyond all internal DP client ids), so every
    completion delivered to our channel is ours and internal FlexKV never sees
    our graph_ids (and vice versa).

Prerequisites (must match the running FlexKV instance):
  * `FLEXKV_RADIX_SHMEM=1` and the same `shm_radix_server_id`.
  * The same `tokens_per_block` / `num_cpu_blocks` / `num_ssd_blocks` / model
    KV geometry, so the attached radix regions and the TE agree on layout.
  * The FlexKV bootstrap must have reserved at least one extra TE channel
    (`num_extra_te_channels >= 1`, default 1) so `channel_id = total_clients +
    external_index` exists.

This controller only does prefetch (SSD/Remote -> CPU, `ignore_gpu=True`), so it
never registers GPU blocks and never touches internal DP GPU memory.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Dict, List, Optional

import numpy as np

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType, TransferOp, TransferOpGraph, TransferType
from flexkv.transfer.shm_channel_handle import TransferManagerShmChannelHandle


class _PrefetchState:
    __slots__ = ("task_id", "callback", "task_end_op_id", "num_ops", "num_blocks")

    def __init__(self, task_id: int, callback, task_end_op_id: int,
                 num_ops: int, num_blocks: int):
        self.task_id = task_id
        self.callback = callback
        self.task_end_op_id = task_end_op_id
        self.num_ops = num_ops
        self.num_blocks = num_blocks


class PrefetchController:
    """External SSD/Remote -> CPU prefetch driver over FlexKV's radix-shmem.

    Attaches directly to the shared CPU (+ SSD) radix regions as RadixClients and
    hand-builds the DISK2H graph, submitting it to the shared TransferEngine over
    a reserved shm channel. No FlexKVConfig / model-geometry / block-count math:
    the only values that matter are

      - server_id          : names the shm radix regions + TE channels
      - tokens_per_block    : block-hash boundary; MUST equal vLLM block_size
      - num_layers          : transfer op layer_granularity
      - enable_ssd/remote   : which source tiers exist
      - instance_num/dp_size: to place channel_id beyond internal DP clients

    Block counts / byte layout are irrelevant (capacity comes from the shm
    region; byte offsets are computed by the TE from its own registered layout).

    Usage:
        pc = PrefetchController(server_id="node0", tokens_per_block=16,
                                num_layers=64)
        pc.start()                         # wait until TE ready
        tid = pc.prefetch(token_ids)       # returns task id
        while not pc.is_done(tid):         # drive completions yourself
            pc.poll(timeout=0.005)
        pc.shutdown()

    poll() is the only drain primitive and MUST be called (in a loop, or
    periodically) — a task's finalize (set_ready + dec_ref) fires from it, so
    skipping it leaks refs and starves the shared mempool.
    """

    def __init__(self,
                 *,
                 server_id: str,
                 tokens_per_block: int,
                 num_layers: int,
                 external_index: int = 0,
                 enable_ssd: bool = True,
                 instance_num: int = 1,
                 dp_size: int = 1):
        from flexkv.cache.radix_shmem_engine import CacheEngineRadixShmem
        from flexkv.server.shm_radix_bootstrap import shm_name_for

        self.num_layers = num_layers
        self.tokens_per_block = tokens_per_block

        self.external_index = external_index
        self.instance_num = instance_num
        self.server_id = server_id

        # Internal DP clients occupy client ids / channel ids [0, total_clients).
        # We live strictly beyond that band.
        total_clients = self.instance_num * dp_size
        self.total_clients = total_clients
        self.external_client_id = total_clients + external_index
        self.channel_id = total_clients + external_index

        num_extra = getattr(GLOBAL_CONFIG_FROM_ENV, "num_extra_te_channels", 1)
        if external_index >= num_extra:
            raise ValueError(
                f"external_index={external_index} exceeds reserved extra TE "
                f"channels ({num_extra}); increase FLEXKV_NUM_EXTRA_TE_CHANNELS "
                f"on the FlexKV bootstrap process."
            )

        # Claim a disjoint graph_id / op_id range so our completions are cleanly
        # isolated from internal FlexKV's (routing is graph_id based). 2^32 ids
        # per client, high bits = external_client_id.
        TransferOpGraph.set_graph_id_range(
            self.external_client_id << 32,
            (self.external_client_id + 1) << 32,
        )
        TransferOp.set_op_id_range(
            self.external_client_id << 32,
            (self.external_client_id + 1) << 32,
        )

        self._task_id_counter = 0
        self._task_id_lock = threading.Lock()
        self._pending: Dict[int, _PrefetchState] = {}  # graph_id -> state
        self._task_to_graph: Dict[int, int] = {}       # task_id -> graph_id
        self._started = False

        # Cumulative instrumentation: how many prefetch tasks submitted a
        # non-empty graph to the TE (vs. the empty-graph fast path where the
        # prefix was already ready in CPU).
        self.submitted_count = 0
        self.noop_count = 0

        # In-flight gauges: prefetch tasks currently submitted to the TE but not
        # yet drained (finalize not fired), and the total SSD->CPU blocks they
        # are moving. Incremented at submit, decremented when poll() finalizes.
        self.inflight_requests = 0
        self.inflight_blocks = 0

        # Attach one RadixClient per source tier directly. num_total_blocks is
        # cosmetic (capacity is defined by the already-created shm region).
        self.cpu_engine = CacheEngineRadixShmem(
            device_type=DeviceType.CPU, num_total_blocks=0,
            tokens_per_block=tokens_per_block,
            shm_name=shm_name_for(DeviceType.CPU, self.server_id))
        self.ssd_engine = None
        if enable_ssd:
            self.ssd_engine = CacheEngineRadixShmem(
                device_type=DeviceType.SSD, num_total_blocks=0,
                tokens_per_block=tokens_per_block,
                shm_name=shm_name_for(DeviceType.SSD, self.server_id))

        # CE-side shm channel to the shared TE (created by the FlexKV bootstrap).
        # The handle attaches purely by (server_id, channel_id).
        self.handle = TransferManagerShmChannelHandle(
            None, None, self.server_id, self.channel_id)

    # ---- lifecycle ----

    def start(self, ready_timeout_s: float = 60.0) -> None:
        self.handle.start()  # no-op for shm; TE owns the channel
        import time
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if self.handle.is_ready():
                self._started = True
                flexkv_logger.info(
                    f"PrefetchController ready: server_id={self.server_id} "
                    f"channel_id={self.channel_id} "
                    f"id_range=[{self.external_client_id << 32}, "
                    f"{(self.external_client_id + 1) << 32})"
                )
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"PrefetchController: shared TE not ready in {ready_timeout_s}s "
            f"(server_id={self.server_id}, channel_id={self.channel_id})"
        )

    def is_ready(self) -> bool:
        return self.handle.is_ready()

    def shutdown(self) -> None:
        # Close only our channel. The TE keeps running for internal FlexKV, and
        # the radix regions are owned by the FlexKV bootstrap process — we must
        # NOT unlink shm or tear them down here.
        with contextlib.suppress(Exception):
            self.handle.shutdown()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()

    # ---- id helper ----

    def _gen_task_id(self) -> int:
        with self._task_id_lock:
            tid = self._task_id_counter
            self._task_id_counter += 1
            return tid

    # ---- prefetch ----

    def prefetch(self,
                 token_ids: np.ndarray,
                 namespace: Optional[List[str]] = None) -> int:
        """Warm the KV for `token_ids` from SSD/Remote into CPU (ready state).

        Returns the task id. If nothing needs to be transferred (full CPU hit or
        no matched blocks to move), the task completes immediately and its id is
        still returned (wait() returns instantly).
        """
        if not self._started:
            raise RuntimeError("PrefetchController.start() must be called first")

        if isinstance(token_ids, np.ndarray):
            token_ids = np.ascontiguousarray(token_ids)
        else:
            token_ids = np.asarray(token_ids)

        task_id = self._gen_task_id()

        graph, callback, task_end_op_id, num_blocks = self._build_prefetch_graph(
            task_id, token_ids, namespace)

        num_ops = graph.num_ops
        state = _PrefetchState(task_id, callback, task_end_op_id, num_ops,
                               num_blocks)
        self._task_to_graph[task_id] = graph.graph_id
        self._pending[graph.graph_id] = state

        if num_ops == 0:
            # Nothing to move. This covers every "don't bother the TM" case:
            #   - SSD had no hit (or SSD disabled)         -> n <= 0
            #   - SSD hit is fully contained in CPU already -> n <= 0
            #   - CPU pool couldn't allocate room           -> aborted
            # We do NOT submit an empty graph to the TE; just run the (possibly
            # None) callback and reap so is_done() is immediately True.
            self.noop_count += 1
            self._finish(graph.graph_id)
        else:
            self.submitted_count += 1
            self.inflight_requests += 1
            self.inflight_blocks += num_blocks
            self.handle.submit(graph)

        return task_id

    def _build_prefetch_graph(self, task_id, token_ids, namespace):
        """Hand-build the SSD->CPU (DISK2H) graph.

        Mirrors the collapsed `_get_impl_local` slice for
        ignore_gpu+ignore_gds+no-p2p: a single DISK2H op moving the blocks that
        are ready in SSD but not yet in CPU. Returns (graph, callback,
        task_end_op_id). callback fires the insert finalize (set_ready+dec_ref),
        releases the CPU node lock and the match pre-locks, and recycles unused
        slots — it MUST run on completion (via poll/wait) or refs leak and the
        shared mempool starves.
        """
        from flexkv.common.block import SequenceMeta

        seq = SequenceMeta(token_ids=token_ids,
                           tokens_per_block=self.tokens_per_block,
                           namespace=namespace)

        empty_graph = TransferOpGraph.create_empty_graph()

        cpu_m = self.cpu_engine.match(seq)
        ssd_m = self.ssd_engine.match(seq) if self.ssd_engine is not None else None

        cpu_ready = cpu_m.num_ready_matched_blocks
        ssd_ready = ssd_m.num_ready_matched_blocks if ssd_m is not None else 0

        # Blocks ready in SSD but not yet in CPU: the set we can warm.
        n = ssd_ready - cpu_ready

        def release_prelocks():
            # match(lock=True) atomically inc_ref'd the ready prefix; release
            # symmetrically on every path so we never leak a match ref.
            pre = getattr(cpu_m, "pre_locked_node", None)
            if pre is not None:
                self.cpu_engine.unlock(pre)
                cpu_m.pre_locked_node = None
            if ssd_m is not None:
                pre_s = getattr(ssd_m, "pre_locked_node", None)
                if pre_s is not None:
                    self.ssd_engine.unlock(pre_s)
                    ssd_m.pre_locked_node = None

        if n <= 0:
            # Nothing to warm (already in CPU, or SSD doesn't have more). Still
            # must drop the match pre-locks.
            release_prelocks()
            return empty_graph, None, -1, 0

        ssd_slots = np.asarray(ssd_m.physical_blocks[:ssd_ready], dtype=np.int64)
        # The tail `n` blocks are the ones missing from CPU (CPU has the first
        # cpu_ready of the shared prefix).
        src_ssd = ssd_slots[cpu_ready:cpu_ready + n]

        cpu_slots = self.cpu_engine.take(
            num_required_blocks=n,
            protected_node=cpu_m.last_node,
            strict=False,
        )
        if len(cpu_slots) < n:
            # Not enough CPU space even after LRU eviction — skip (do not crash).
            self.cpu_engine.recycle(cpu_slots)
            release_prelocks()
            return empty_graph, None, -1, 0

        graph = TransferOpGraph()
        op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.DISK2H,
            src_block_ids=np.asarray(src_ssd, dtype=np.int64),
            dst_block_ids=np.asarray(cpu_slots, dtype=np.int64),
            layer_id=0,
            layer_granularity=self.num_layers,
        )
        graph.add_transfer_op(op)

        # Insert the freshly-allocated CPU slots as an UNREADY suffix so a
        # concurrent reader won't treat them as valid before the transfer lands.
        # insert() auto-inc_refs (locks) the node and arms a finalize
        # (set_ready + dec_ref). We additionally lock_node for the hand-off,
        # mirroring _get_impl_local.
        cpu_node, cpu_unused = self.cpu_engine.insert(
            seq, np.asarray(cpu_slots, dtype=np.int64),
            num_insert_blocks=ssd_ready, is_ready=False, match_result=cpu_m)
        if cpu_node is not None:
            self.cpu_engine.lock_node(cpu_node)

        # Take over protection, then drop the match pre-locks (as _get_impl_local
        # does): lock_node above already took an independent ref for matched
        # nodes; inserted nodes carry their own armed finalize.
        release_prelocks()

        def callback():
            # cpu_node here is always an *inserted* node (insert(is_ready=False)),
            # so it carries shmradix's armed finalize (= set_ready + dec_ref).
            # unlock() fires that finalize in one atomic shot — no separate
            # set_ready needed (set_ready() is a no-op on inserted nodes anyway).
            if cpu_node is not None:
                self.cpu_engine.unlock(cpu_node)
            if cpu_unused is not None and getattr(cpu_unused, "size", 0) > 0:
                self.cpu_engine.recycle(cpu_unused)

        return graph, callback, op.op_id, n

    # ---- completion draining ----

    def _finish(self, graph_id: int) -> None:
        """Run the graph's finalize callback (once) and reap its tracking state.

        Reaping here (rather than in a separate wait()) keeps `_pending` /
        `_task_to_graph` bounded for a long-running driver.
        """
        state = self._pending.pop(graph_id, None)
        if state is None:
            return
        self._task_to_graph.pop(state.task_id, None)
        # Only submitted (non-empty) graphs were counted as in-flight.
        if state.num_ops > 0:
            self.inflight_requests -= 1
            self.inflight_blocks -= state.num_blocks
        if state.callback is not None:
            try:
                state.callback()
            except Exception as e:  # pragma: no cover
                flexkv_logger.error(
                    f"PrefetchController callback failed for graph {graph_id}: {e}",
                    exc_info=True,
                )

    def poll(self, timeout: float = 0.0) -> None:
        """Drain completed ops from our channel and finalize done graphs.

        This is the ONLY drain primitive — a prefetch task's finalize
        (set_ready + dec_ref) fires from here, so the caller MUST call poll()
        (in a loop, or after issuing prefetches) or refs leak and the shared
        mempool starves. Combine with `is_done(task_id)` to build a wait loop.
        """
        for cop in self.handle.wait(timeout):
            if cop.is_graph_completed():  # op_id == -1: whole graph done
                self._finish(cop.graph_id)

    def is_done(self, task_id: int) -> bool:
        """True once the task's graph completed (and was reaped) — or if it was
        a no-op / unknown id. Call poll() first to drain fresh completions."""
        return task_id not in self._task_to_graph
