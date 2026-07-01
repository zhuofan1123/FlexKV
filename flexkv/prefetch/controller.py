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

  * **Shared index**: `GlobalCacheEngine` in radix_shmem mode attaches to the
    radix regions as a `shmradix.RadixClient` (see
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
import copy
import threading
from typing import Dict, List, Optional

import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import TransferOp, TransferOpGraph
from flexkv.cache.cache_engine import GlobalCacheEngine, DEFAULT_CACHE_STRATEGY
from flexkv.transfer.shm_channel_handle import TransferManagerShmChannelHandle


class _PrefetchState:
    __slots__ = ("task_id", "callback", "task_end_op_id", "done", "num_ops")

    def __init__(self, task_id: int, callback, task_end_op_id: int, num_ops: int):
        self.task_id = task_id
        self.callback = callback
        self.task_end_op_id = task_end_op_id
        self.num_ops = num_ops
        self.done = False


class PrefetchController:
    """Prefetch driver that shares FlexKV's radix-shmem index and TE.

    Usage:
        pc = PrefetchController(model_config, cache_config)
        pc.start()                         # attach + wait until TE ready
        tid = pc.prefetch(token_ids)       # returns task id (or -1 if no-op)
        ...                                # do other work
        pc.wait([tid], timeout=5.0)        # block until warmed into CPU
        pc.shutdown()
    """

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 external_index: int = 0,
                 server_id: Optional[str] = None):
        if not bool(getattr(GLOBAL_CONFIG_FROM_ENV, "radix_shmem", False)):
            raise RuntimeError(
                "PrefetchController requires radix_shmem mode; set "
                "FLEXKV_RADIX_SHMEM=1 (and GLOBAL_CONFIG_FROM_ENV.radix_shmem) "
                "and a matching shm_radix_server_id before constructing it."
            )
        if not cache_config.enable_cpu:
            raise ValueError("PrefetchController requires enable_cpu=True")

        self.model_config = model_config
        self.cache_config = cache_config
        self.external_index = external_index

        self.instance_num = GLOBAL_CONFIG_FROM_ENV.instance_num
        self.server_id = server_id or getattr(
            GLOBAL_CONFIG_FROM_ENV, "shm_radix_server_id", "default")

        # Internal DP clients occupy client ids / channel ids [0, total_clients).
        # We live strictly beyond that band.
        total_clients = self.instance_num * self.model_config.dp_size
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

        # GlobalCacheEngine in radix_shmem mode attaches to the shared radix
        # regions as RadixClient(s) — one per enabled device type.
        self.cache_engine = GlobalCacheEngine(cache_config, model_config)

        # CE-side shm channel to the shared TE (created by the FlexKV bootstrap).
        self.handle = TransferManagerShmChannelHandle(
            model_config, cache_config, self.server_id, self.channel_id)

        self._task_id_counter = 0
        self._task_id_lock = threading.Lock()
        self._pending: Dict[int, _PrefetchState] = {}  # graph_id -> state
        self._task_to_graph: Dict[int, int] = {}       # task_id -> graph_id
        self._started = False

        # Instrumentation: how many prefetch tasks actually submitted a
        # non-empty graph to the TE (vs. the empty-graph fast path where the
        # prefix was already ready in CPU). Useful for benchmarks and tests.
        self.submitted_count = 0
        self.noop_count = 0

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

        # CPU-only upload, mirroring KVTaskManager.create_prefetch_task.
        temp = copy.deepcopy(DEFAULT_CACHE_STRATEGY)
        temp.ignore_gpu = True   # do not touch GPU
        temp.ignore_gds = True   # no GDS path

        fake_slot_mapping = np.zeros_like(token_ids)
        fake_token_mask = np.ones_like(token_ids)

        graph, _return_mask, callback, _op_callback_dict, task_end_op_id = \
            self.cache_engine.get(
                request_id=task_id,
                token_ids=token_ids,
                token_mask=fake_token_mask,
                slot_mapping=fake_slot_mapping,
                layer_num=self.model_config.num_layers,
                temp_cache_strategy=temp,
                namespace=namespace,
            )

        num_ops = graph.num_ops
        state = _PrefetchState(task_id, callback, task_end_op_id, num_ops)
        self._task_to_graph[task_id] = graph.graph_id
        self._pending[graph.graph_id] = state

        if num_ops == 0:
            # Nothing to move (already ready in CPU, or nothing matched). Run the
            # write-path callback (set_ready + dec_ref) and mark done.
            self.noop_count += 1
            self._finish(graph.graph_id)
        else:
            self.submitted_count += 1
            self.handle.submit(graph)

        return task_id

    # ---- completion draining ----

    def _finish(self, graph_id: int) -> None:
        state = self._pending.get(graph_id)
        if state is None or state.done:
            return
        if state.callback is not None:
            try:
                state.callback()
            except Exception as e:  # pragma: no cover
                flexkv_logger.error(
                    f"PrefetchController callback failed for graph {graph_id}: {e}",
                    exc_info=True,
                )
        state.done = True

    def poll(self, timeout: float = 0.0) -> None:
        """Drain completed ops from our channel and finalize done graphs."""
        for cop in self.handle.wait(timeout):
            state = self._pending.get(cop.graph_id)
            if state is None:
                # Defensive: our graph_id range is disjoint, so this shouldn't
                # happen unless a stale completion arrives after wait() dropped
                # the task. Ignore.
                continue
            if cop.is_graph_completed():  # op_id == -1: whole graph done
                self._finish(cop.graph_id)

    def is_done(self, task_id: int) -> bool:
        graph_id = self._task_to_graph.get(task_id)
        if graph_id is None:
            return True  # unknown / already reaped
        state = self._pending.get(graph_id)
        return state is None or state.done

    def wait(self,
             task_ids,
             timeout: float = 20.0) -> Dict[int, bool]:
        """Block until the given prefetch tasks complete (or timeout).

        Returns {task_id: success_bool}. A task with no graph ops returns True
        immediately. After a task completes it is reaped from internal maps.
        """
        import time
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        results: Dict[int, bool] = {}
        start = time.time()

        self.poll(timeout=0.0)
        remaining = list(task_ids)
        while remaining:
            still: List[int] = []
            for tid in remaining:
                if self.is_done(tid):
                    results[tid] = True
                    self._reap(tid)
                else:
                    still.append(tid)
            remaining = still
            if not remaining:
                break
            if time.time() - start > timeout:
                for tid in remaining:
                    results[tid] = False
                break
            self.poll(timeout=0.001)
        return results

    def _reap(self, task_id: int) -> None:
        graph_id = self._task_to_graph.pop(task_id, None)
        if graph_id is not None:
            self._pending.pop(graph_id, None)
