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

Prerequisites (the running FlexKV instance provides these via shm; the caller
only needs the server_id):
  * FlexKV running in radix_shmem mode with a known `shm_radix_server_id`.
  * The FlexKV bootstrap must have reserved at least one extra TE channel
    (`num_extra_te_channels >= 1`, default 1) so a reserved channel exists at
    `channel_id = total_clients` (published in the TE ctrl block).
  * tokens_per_block is recovered from the radix region itself
    (RadixClient.block_size()); no need to pass model KV geometry.

This controller only does prefetch (SSD/Remote -> CPU, `ignore_gpu=True`), so it
never registers GPU blocks and never touches internal DP GPU memory.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType, TransferOp, TransferOpGraph, TransferType
from flexkv.transfer.shm_channel import ShmControlBlock, _safe_id
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
    a reserved shm channel. Everything but `server_id` is auto-discovered from
    shm:
      - tokens_per_block : read from the radix region (RadixClient.block_size()).
      - channel_id       : -1 (default) auto-resolves to the first reserved
                           external channel (= total_clients, read from the TE
                           ctrl block); pass an explicit id to override.
      - layer count      : not carried on the op at all; the TE derives it from
                           the layout registered by the serving process.

    A flock on the chosen channel is held as insurance: if a second external
    controller targets the same channel it fails loudly instead of silently
    sharing the ring. The kernel releases the lock on exit (incl. crash).

    Usage:
        pc = PrefetchController(server_id="node0")
        pc.start()                         # ready + starts background drain
        tid = pc.prefetch(token_ids)       # fire-and-forget from any thread
        ...                                # optionally check pc.is_done(tid)
        pc.shutdown()                      # drains, stops drain thread, unlocks

    start() launches a daemon thread that drains completions every ~1 ms and
    publishes each landed transfer into the CPU tree. The caller's thread only
    calls prefetch(); it never polls. Thread-safe: prefetch() may be called
    concurrently with the drain thread.

    Draining is load-bearing, not bookkeeping: radixshmem only accepts blocks
    that already hold data, so the insert runs from the completion callback. A
    task that is never drained leaves its CPU slots outside the tree, where
    nothing can find them and nothing can evict them.
    """

    def __init__(self,
                 *,
                 server_id: str,
                 channel_id: int = -1,
                 enable_ssd: bool = True):
        from flexkv.cache.radix_shmem_engine import CacheEngineRadixShmem
        from flexkv.server.shm_radix_bootstrap import shm_name_for

        self.server_id = server_id
        self._ext_lock_fd: Optional[int] = None

        # Read the TE ctrl block to discover total_clients (= first reserved
        # external channel). Poll for the ctrl file since the TE may still be
        # coming up (same wait discipline as TransferManagerShmChannelHandle).
        self._ctrl = self._attach_ctrl(server_id)
        total_clients = self._ctrl.get_total_clients()
        self.total_clients = total_clients

        # Resolve channel_id: -1 -> first reserved slot.
        if channel_id < 0:
            channel_id = total_clients
        elif channel_id < total_clients:
            raise ValueError(
                f"channel_id={channel_id} is inside the internal DP band "
                f"[0, {total_clients}); external channels start at "
                f"{total_clients}.")
        self.channel_id = channel_id
        self.external_client_id = channel_id

        # flock insurance: exclusive-claim this channel across processes. Kernel
        # auto-releases on exit/crash, so no stale-slot cleanup is needed.
        self._acquire_ext_lock(server_id, channel_id)

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

        # Concurrency: a background thread drains completions (poll) while the
        # caller's thread issues prefetch(). _state_lock guards the bookkeeping
        # dicts + inflight gauges; _poll_lock keeps the result ring single-
        # consumer (poll must not run from two threads at once).
        self._state_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._poll_interval_s = 0.001  # 1 ms idle futex wait between drains

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

        # Attach one RadixClient per source tier. tokens_per_block=-1 => recover
        # it from the region (RadixClient.block_size()); num_total_blocks is
        # cosmetic (capacity comes from the already-created shm region).
        self.cpu_engine = CacheEngineRadixShmem(
            device_type=DeviceType.CPU, num_total_blocks=0,
            tokens_per_block=-1,
            shm_name=shm_name_for(DeviceType.CPU, self.server_id))
        self.tokens_per_block = self.cpu_engine.tokens_per_block
        self.ssd_engine = None
        if enable_ssd:
            self.ssd_engine = CacheEngineRadixShmem(
                device_type=DeviceType.SSD, num_total_blocks=0,
                tokens_per_block=-1,
                shm_name=shm_name_for(DeviceType.SSD, self.server_id))

        # CE-side shm channel to the shared TE (created by the FlexKV bootstrap).
        # The handle attaches purely by (server_id, channel_id).
        self.handle = TransferManagerShmChannelHandle(
            None, None, self.server_id, self.channel_id)

    @staticmethod
    def _attach_ctrl(server_id: str,
                     wait_timeout_s: float = 60.0) -> ShmControlBlock:
        safe = _safe_id(server_id)
        ctrl_path = f"/dev/shm/flexkv_te_ctrl_{safe}"
        deadline = time.monotonic() + wait_timeout_s
        while time.monotonic() < deadline:
            if os.path.exists(ctrl_path):
                return ShmControlBlock(server_id, create=False)
            time.sleep(0.05)
        raise RuntimeError(
            f"PrefetchController: TE ctrl block {ctrl_path} not found in "
            f"{wait_timeout_s}s — is FlexKV (radix_shmem) running for "
            f"server_id={server_id}?")

    def _acquire_ext_lock(self, server_id: str, channel_id: int) -> None:
        safe = _safe_id(server_id)
        path = f"/dev/shm/flexkv_te_extlock_{safe}_{channel_id}"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError(
                f"another external PrefetchController already holds channel "
                f"{channel_id} for server {server_id} (flock on {path})")
        self._ext_lock_fd = fd

    # ---- lifecycle ----

    def start(self, ready_timeout_s: float = 60.0) -> None:
        self.handle.start()  # no-op for shm; TE owns the channel
        import time
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if self.handle.is_ready():
                self._started = True
                # Launch the background drain thread. From here the caller only
                # calls prefetch(); completions (and their finalize) are drained
                # automatically every ~1 ms — the caller's thread never polls.
                self._stop.clear()
                self._poll_thread = threading.Thread(
                    target=self._poll_loop, name="prefetch-poll", daemon=True)
                self._poll_thread.start()
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

    def _poll_loop(self) -> None:
        # Blocks up to _poll_interval_s in a futex inside handle.wait() when the
        # ring is empty, so this thread costs ~nothing while idle.
        while not self._stop.is_set():
            try:
                self.poll(timeout=self._poll_interval_s)
            except Exception as e:  # pragma: no cover
                flexkv_logger.error(f"prefetch poll loop error: {e}",
                                    exc_info=True)

    def is_ready(self) -> bool:
        return self.handle.is_ready()

    def shutdown(self, drain_timeout_s: float = 2.0) -> None:
        # Stop the drain thread first so nothing touches the channel/engines
        # after we close them.
        self._stop.set()
        th = self._poll_thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._poll_thread = None
        self._drain_inflight(drain_timeout_s)
        # Close only our channel. The TE keeps running for internal FlexKV, and
        # the radix regions are owned by the FlexKV bootstrap process — we must
        # NOT unlink shm or tear them down here.
        with contextlib.suppress(Exception):
            self.handle.shutdown()
        # Release the channel flock (kernel also releases on process exit).
        if self._ext_lock_fd is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(self._ext_lock_fd, fcntl.LOCK_UN)
                os.close(self._ext_lock_fd)
            self._ext_lock_fd = None
        if getattr(self, "_ctrl", None) is not None:
            with contextlib.suppress(Exception):
                self._ctrl.close()
            self._ctrl = None

    def _drain_inflight(self, timeout_s: float) -> None:
        """Give already-submitted graphs a last chance to publish before we go.

        A task whose completion we never drain leaves its staged CPU slots
        attached to nothing: unreachable by any query and invisible to eviction,
        so they are lost until the region is recreated. Aborting them instead is
        not an option — the TE may still be writing into them — so the choice is
        wait, then say what was left behind.
        """
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            with self._state_lock:
                if not self._pending:
                    return
            if time.monotonic() >= deadline:
                break
            with contextlib.suppress(Exception):
                self.poll(timeout=0.01)
        with self._state_lock:
            stranded = len(self._pending)
            blocks = sum(s.num_blocks for s in self._pending.values())
        flexkv_logger.warning(
            f"PrefetchController shutting down with {stranded} prefetch "
            f"task(s) still in flight; {blocks} staged CPU block(s) stay "
            f"claimed in region {self.server_id} (a transfer may still be "
            f"writing into them, so they cannot be recycled here)")

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
        """Warm the KV for `token_ids` from the local SSD tier into CPU.

        The warmed blocks become visible to every attached process (and, on a
        clustered region, to peers) when the drain thread publishes them.

        Returns the task id. If nothing needs to be transferred (full CPU hit or
        no matched blocks to move), the task completes immediately and its id is
        still returned (is_done() is instantly True).
        """
        if not self._started:
            raise RuntimeError("PrefetchController.start() must be called first")

        if isinstance(token_ids, np.ndarray):
            token_ids = np.ascontiguousarray(token_ids)
        else:
            token_ids = np.asarray(token_ids)

        task_id = self._gen_task_id()

        # Hold _state_lock across the whole build: it serializes shmradix engine
        # access (match/take/insert here vs. the poll thread's finalize
        # unlock/recycle — pybind releases the GIL, so without this two threads
        # would enter the same RadixClient in parallel). It also makes
        # "register pending" happen-before "submit", so a completion can never
        # arrive at the poll thread before _pending knows the graph.
        with self._state_lock:
            graph, callback, task_end_op_id, num_blocks = \
                self._build_prefetch_graph(task_id, token_ids, namespace)

            num_ops = graph.num_ops
            state = _PrefetchState(task_id, callback, task_end_op_id, num_ops,
                                   num_blocks)
            self._task_to_graph[task_id] = graph.graph_id
            self._pending[graph.graph_id] = state

            if num_ops == 0:
                # Nothing to move. Covers every "don't bother the TM" case:
                #   - SSD had no hit (or SSD disabled)          -> n <= 0
                #   - SSD hit is fully contained in CPU already -> n <= 0
                #   - CPU pool couldn't allocate room           -> aborted
                # We do NOT submit an empty graph; run the (possibly None)
                # callback and reap so is_done() is immediately True.
                self.noop_count += 1
                self._finish_locked(graph.graph_id)
            else:
                self.submitted_count += 1
                self.inflight_requests += 1
                self.inflight_blocks += num_blocks
                # A raise here means the submit ring is misconfigured or wedged
                # (the only two paths in `submit_send`), so the controller is
                # done for anyway; let it propagate and leave the staged slots
                # be rather than carry a cleanup path for it.
                self.handle.submit(graph)

        return task_id

    def _build_prefetch_graph(self, task_id, token_ids, namespace):
        """Hand-build the SSD->CPU (DISK2H) graph.

        Mirrors the collapsed `_get_impl_radixshmem` slice for ignore_gpu: a
        single DISK2H op moving the blocks the local SSD tree holds but the
        local CPU tree does not. Returns (graph, callback,
        task_end_op_id, num_blocks). `callback` publishes the staged slots into
        the CPU tree and drops the match refs — it MUST run on completion (via
        poll/wait) or the staged slots stay outside the tree and the shared
        mempool starves.
        """
        from flexkv.cache.radix_shmem_engine import ShmRadixMatch, StagedRadixInsert
        from flexkv.common.block import SequenceMeta

        seq = SequenceMeta(token_ids=token_ids,
                           tokens_per_block=self.tokens_per_block,
                           namespace=namespace)

        empty_graph = TransferOpGraph.create_empty_graph()

        # Local-only on both tiers. A peer tail is useless here: this controller
        # emits DISK2H, which reads the local SSD region, so a spliced match
        # would hand it slot ids out of somebody else's mempool. No SSD tier
        # stands in as an empty match, like `_match_radixshmem` does, so
        # everything below reads the pair without re-testing the tier.
        cpu_m = self.cpu_engine.match(seq, with_peer=False)
        ssd_m = (self.ssd_engine.match(seq, with_peer=False)
                 if self.ssd_engine is not None else ShmRadixMatch())

        # match(lock=True) inc_ref'd each matched prefix; release() is the only
        # thing that drops those refs, and it must run on every path. It is
        # idempotent, so a path that runs both cleanups is fine.
        def drop_match_refs():
            for m in (cpu_m, ssd_m):
                try:
                    m.release()
                except Exception as e:  # keep releasing the rest
                    flexkv_logger.error(
                        f"prefetch task {task_id}: match release failed: {e}")

        # Local-only matches, so num_local_blocks is the whole hit.
        cpu_hit = cpu_m.num_local_blocks
        ssd_hit = ssd_m.num_local_blocks

        # Blocks the SSD tier holds beyond what CPU already has: what we can warm.
        n = ssd_hit - cpu_hit

        if n <= 0:
            # Nothing to warm (already in CPU, or SSD doesn't have more).
            drop_match_refs()
            return empty_graph, None, -1, 0

        # The tail `n` blocks are the ones missing from CPU (CPU has the first
        # cpu_hit of the shared prefix).
        src_ssd = np.asarray(ssd_m.local_range(cpu_hit, ssd_hit), dtype=np.int64)

        cpu_slots = self.cpu_engine.take(num_required_blocks=n, strict=False)
        if len(cpu_slots) < n:
            # Not enough CPU space even after LRU eviction — skip (do not crash).
            self.cpu_engine.recycle(cpu_slots)
            drop_match_refs()
            return empty_graph, None, -1, 0

        cpu_slots = np.asarray(cpu_slots, dtype=np.int64)
        graph = TransferOpGraph()
        op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.DISK2H,
            src_block_ids=np.asarray(src_ssd, dtype=np.int64),
            dst_block_ids=cpu_slots,
        )
        graph.add_transfer_op(op)

        # radixshmem publishes only complete blocks, so the CPU slots stay out of
        # the tree until the DISK2H lands — nothing else can see them, and
        # nothing else can free them either, hence the staged handle.
        # publish() inserts at start = ssd_hit - n = cpu_hit, which the local tree
        # must still reach; cpu_m's ref is what keeps it there, so the refs go to
        # the staged insert -- both its exits drop them, after the insert.
        staged = StagedRadixInsert(
            engine=self.cpu_engine,
            sequence_meta=seq,
            slots=cpu_slots,
            path_end=ssd_hit,
            label=f"prefetch task {task_id} CPU",
            holds=[drop_match_refs],
        )

        return graph, staged.publish, op.op_id, n

    # ---- completion draining ----

    def _finish_locked(self, graph_id: int) -> None:
        """Run the graph's finalize callback (once) and reap its tracking state.

        Caller MUST hold `_state_lock` (serializes shmradix engine access and
        the bookkeeping dicts). Reaping here keeps `_pending` / `_task_to_graph`
        bounded for a long-running driver.
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

        Normally driven by the internal background thread (~1 ms); callers do
        not need to call this. Publishing a landed transfer into the CPU tree
        happens here. `_poll_lock` keeps the result ring single-consumer; the
        insert itself runs under `_state_lock`.
        """
        with self._poll_lock:
            completed = [cop for cop in self.handle.wait(timeout)
                         if cop.is_graph_completed()]  # op_id == -1
        if not completed:
            return
        with self._state_lock:
            for cop in completed:
                self._finish_locked(cop.graph_id)

    def is_done(self, task_id: int) -> bool:
        """True once the task's graph completed (and was reaped) — or if it was
        a no-op / unknown id."""
        with self._state_lock:
            return task_id not in self._task_to_graph
