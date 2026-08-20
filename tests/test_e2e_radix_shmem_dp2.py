"""End-to-end test for FLEXKV_RADIX_SHMEM=1 with dp_size=2.

Two independent DP scheduler processes share one radix-shmem index and one
shared TransferEngine:

  * dp0 is the bootstrap proc: it creates the per-device shm radix regions and
    spawns the single shared TE subprocess.
  * dp1 attaches to the existing shm radix regions as a RadixClient and feeds
    the same shared TE over its own shm channel.

What this validates:
  1. Two KVManager(radix_shmem) procs come up concurrently against one server_id.
  2. Both can put_async through the single shared TE (disjoint op_id ranges).
  3. Cross-DP prefix sharing: a sequence written by dp0 is visible in the shared
     index to dp1 via get_match (the whole point of the radix-shmem path).

Requires >=2 CUDA devices. Run inside the container:

    PYTHONPATH=/raid/zfl/FlexKV \\
    LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH \\
    python3 tests/test_e2e_radix_shmem_dp2.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch


TOKENS_PER_BLOCK = 16
NUM_GPU_BLOCKS = 256
BLOCK_PER_REQUEST = 32
# A fixed prefix that dp0 writes and dp1 later looks up across the shared index.
SHARED_SEED = 0x5EED


def _build_request(rng_seed: int, start_block: int, n_blocks: int):
    rng = np.random.default_rng(rng_seed)
    block_ids = np.arange(start_block, start_block + n_blocks, dtype=np.int64)
    slot_mapping = (np.repeat(block_ids, TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK
                    + np.tile(np.arange(TOKENS_PER_BLOCK), n_blocks))
    token_ids = rng.integers(0, 32000, size=slot_mapping.shape, dtype=np.int64)
    return token_ids, slot_mapping, block_ids


def _tp_client_proc(dp_client_id, tp_rank, server_recv_port,
                    model_config, cache_config, num_gpu_blocks, child_conn):
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.common.memory_handle import TensorSharedHandle
    from flexkv.server.client import KVTPClient

    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads // model_config.tp_size,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    gpu_blocks = [
        torch.empty(size=tuple(gpu_layout.kv_shape[1:]),
                    dtype=model_config.dtype).cuda(device_id)
        for _ in range(model_config.num_layers)
    ]
    tp_client.register_to_server(gpu_blocks, gpu_layout)
    child_conn.send([TensorSharedHandle(t) for t in gpu_blocks])
    child_conn.close()
    while True:
        time.sleep(1)


def _dp_proc(dp_client_id, dp_size, server_id, write_barrier, result_q):
    """Full lifecycle of one DP scheduler process."""
    os.environ["FLEXKV_RADIX_SHMEM"] = "1"
    os.environ["FLEXKV_SHM_RADIX_ID"] = server_id
    os.environ["FLEXKV_ENABLE_MPS"] = "0"
    # All DP procs share a single TE, so they must agree on server_recv_port
    # (and therefore the gpu_register_port the TE listens on).
    os.environ["FLEXKV_SERVER_RECV_PORT"] = f"ipc:///tmp/flexkv_{server_id}"

    from flexkv.common.config import (ModelConfig, CacheConfig,
                                      GLOBAL_CONFIG_FROM_ENV)
    from flexkv.common.request import KVResponseStatus
    from flexkv.kvmanager import KVManager

    GLOBAL_CONFIG_FROM_ENV.radix_shmem = True
    GLOBAL_CONFIG_FROM_ENV.shm_radix_server_id = server_id
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    GLOBAL_CONFIG_FROM_ENV.server_recv_port = f"ipc:///tmp/flexkv_{server_id}"

    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=128,
        dtype=torch.float16, use_mla=False, tp_size=1, dp_size=dp_size,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK, enable_cpu=True, enable_ssd=False,
        num_cpu_blocks=4096,
    )

    tag = f"[dp{dp_client_id}]"
    try:
        kvm = KVManager(model_config, cache_config, dp_client_id=dp_client_id)
        kvm.start()

        # Bring up this DP's TP client (registers its GPU blocks with shared TE).
        mp_ctx = mp.get_context("spawn")
        parent, child = mp_ctx.Pipe()
        p = mp_ctx.Process(
            target=_tp_client_proc,
            args=(dp_client_id, 0, kvm.gpu_register_port,
                  model_config, cache_config, NUM_GPU_BLOCKS, child),
            daemon=True,
        )
        p.start()
        _ = parent.recv()
        print(f"{tag} TP client registered", flush=True)

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not kvm.is_ready():
            time.sleep(0.1)
        if not kvm.is_ready():
            raise RuntimeError(f"{tag} not ready in 60s")
        print(f"{tag} READY", flush=True)

        # --- Phase 1: each DP writes its own private requests concurrently. ---
        own = []
        for i in range(3):
            tok, slot, blk = _build_request(
                rng_seed=1000 * dp_client_id + i,
                start_block=i * BLOCK_PER_REQUEST, n_blocks=BLOCK_PER_REQUEST)
            tid = kvm.put_async(token_ids=tok, slot_mapping=slot)
            own.append(tid)
        st = kvm.wait(own, timeout=60, completely=True)
        own_ok = sum(1 for s in st.values()
                     if s.status == KVResponseStatus.SUCCESS)
        print(f"{tag} private puts: {own_ok}/{len(own)} ok", flush=True)

        # --- Phase 2: dp0 writes a SHARED prefix; dp1 waits, then matches it. --
        shared_hit_blocks = -1
        if dp_client_id == 0:
            tok, slot, blk = _build_request(
                rng_seed=SHARED_SEED, start_block=128,
                n_blocks=BLOCK_PER_REQUEST)
            tid = kvm.put_async(token_ids=tok, slot_mapping=slot)
            s = kvm.wait([tid], timeout=60, completely=True)
            shared_ok = all(v.status == KVResponseStatus.SUCCESS
                            for v in s.values())
            print(f"{tag} shared-prefix put ok={shared_ok}", flush=True)
            write_barrier.wait()  # release dp1 to look it up
        else:
            write_barrier.wait()  # wait until dp0 finished the shared put
            # Same token_ids as dp0's shared write -> should hit shared index.
            tok, _, _ = _build_request(
                rng_seed=SHARED_SEED, start_block=128,
                n_blocks=BLOCK_PER_REQUEST)
            # poll get_match a few times: index visibility is async post-store.
            for _ in range(50):
                _tid, mask = kvm.get_match(token_ids=tok)
                shared_hit_blocks = int(np.count_nonzero(mask)) // TOKENS_PER_BLOCK \
                    if mask is not None else 0
                if shared_hit_blocks > 0:
                    break
                time.sleep(0.2)
            print(f"{tag} cross-DP match hit_blocks={shared_hit_blocks}",
                  flush=True)

        p.terminate()
        p.join()
        kvm.shutdown()
        result_q.put((dp_client_id, own_ok, len(own), shared_hit_blocks))
    except Exception:
        import traceback
        traceback.print_exc()
        result_q.put((dp_client_id, -1, -1, -1))


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("SKIP: need >=2 CUDA devices")
        return 0

    dp_size = 2
    server_id = f"dp2_{os.getpid()}"
    ctx = mp.get_context("spawn")
    write_barrier = ctx.Barrier(dp_size)
    result_q = ctx.Queue()

    procs = [
        ctx.Process(target=_dp_proc,
                    args=(dp, dp_size, server_id, write_barrier, result_q),
                    daemon=False)
        for dp in range(dp_size)
    ]
    for p in procs:
        p.start()

    results = {}
    deadline = time.monotonic() + 180
    while len(results) < dp_size and time.monotonic() < deadline:
        try:
            dp, ok, total, hit = result_q.get(timeout=5)
            results[dp] = (ok, total, hit)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    print("\n=== RESULTS ===")
    print(results)

    # Validate.
    if len(results) != dp_size:
        print(f"FAIL: only {len(results)}/{dp_size} DP procs reported")
        return 1
    for dp, (ok, total, hit) in results.items():
        if ok != total or ok < 0:
            print(f"FAIL: dp{dp} private puts {ok}/{total}")
            return 1
    # dp1 must have seen dp0's shared prefix through the shared index.
    dp1_hit = results[1][2]
    if dp1_hit <= 0:
        print(f"FAIL: cross-DP prefix sharing not observed (dp1 hit={dp1_hit})")
        return 1

    print(f"PASS: dp2 radix-shmem OK — both DPs put through shared TE; "
          f"dp1 matched {dp1_hit} blocks of dp0's prefix via shared index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
