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
      - layer count      : op uses layer_granularity=-1; the TE fills its own.

    A flock on the chosen channel is held as insurance: if a second external
    controller targets the same channel it fails loudly instead of silently
    sharing the ring. The kernel releases the lock on exit (incl. crash).

    Usage:
        pc = PrefetchController(server_id="node0")
        pc.start()                         # ready + starts background drain
        tid = pc.prefetch(token_ids)       # fire-and-forget from any thread
        ...                                # optionally check pc.is_done(tid)
        pc.shutdown()                      # stops drain thread, releases lock

    start() launches a daemon thread that drains completions every ~1 ms and
    fires each task's finalize (set_ready + dec_ref). The caller's thread only
    calls prefetch(); it never polls. Thread-safe: prefetch() may be called
    concurrently with the drain thread.
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

    def shutdown(self) -> None:
        # Stop the drain thread first so nothing touches the channel/engines
        # after we close them.
        self._stop.set()
        th = self._poll_thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._poll_thread = None
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
            # -1 => TE fills its own registered layer count (from vLLM's GPU
            # layout); the controller needn't know num_layers.
            layer_granularity=-1,
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
        not need to call this. Firing a task's finalize (set_ready + dec_ref)
        happens here. `_poll_lock` keeps the result ring single-consumer; the
        finalize itself runs under `_state_lock`.
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
