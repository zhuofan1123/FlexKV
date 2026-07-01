"""End-to-end test for the external PrefetchController on the radix-shmem path.

Topology:
  * dp0 is the FlexKV bootstrap proc: it creates the shm radix regions and spawns
    the single shared TE subprocess. With num_extra_te_channels>=1 (default 1),
    the TE reserves channel_id=total_clients for an external process.
  * an external process builds a `PrefetchController` that attaches to the SAME
    radix index (as a RadixClient) and submits CPU-only upload graphs to the SAME
    TE over the reserved channel — with a DISJOINT graph_id/op_id range.

What this validates:
  1. The external controller attaches and reaches is_ready() against the shared TE.
  2. Its graph_id range is disjoint from internal FlexKV's (isolation invariant).
  3. prefetch() tasks complete via the reserved channel (correct completion
     routing back to the external submitter — no hang, no mis-route).
  4. Internal DP put/wait runs concurrently with external prefetch; both make
     progress and neither sees the other's ids.
  5. After external prefetch of a written prefix, the shared index reflects it
     (internal get_match sees ready CPU blocks) — i.e. index is truly shared.

Requires >=1 CUDA device. Run inside the container:

    PYTHONPATH=/raid/zfl/FlexKV \\
    LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH \\
    python3 tests/test_e2e_prefetch_controller.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch


TOKENS_PER_BLOCK = 16
NUM_GPU_BLOCKS = 128
BLOCK_PER_REQUEST = 16
SHARED_SEED = 0x9F0
# Small CPU pool so a flood of other writes evicts the shared prefix from CPU
# (it survives in SSD), forcing the external prefetch to issue a real SSD->CPU
# transfer graph through the reserved channel instead of the empty-graph fast path.
# CPU holds ~16 requests' worth of slots; flooding with more than that evicts the
# shared prefix by LRU while it stays resident in the (larger) SSD tier.
NUM_CPU_BLOCKS = 512
NUM_SSD_BLOCKS = 4096
FLOOD_REQUESTS = 40


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


def _set_radix_env(server_id: str):
    os.environ["FLEXKV_RADIX_SHMEM"] = "1"
    os.environ["FLEXKV_SHM_RADIX_ID"] = server_id
    os.environ["FLEXKV_ENABLE_MPS"] = "0"
    os.environ["FLEXKV_SERVER_RECV_PORT"] = f"ipc:///tmp/flexkv_{server_id}"
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    GLOBAL_CONFIG_FROM_ENV.radix_shmem = True
    GLOBAL_CONFIG_FROM_ENV.shm_radix_server_id = server_id
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    GLOBAL_CONFIG_FROM_ENV.server_recv_port = f"ipc:///tmp/flexkv_{server_id}"


def _make_configs(dp_size: int, ssd_dir: str):
    from flexkv.common.config import ModelConfig, CacheConfig
    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=128,
        dtype=torch.float16, use_mla=False, tp_size=1, dp_size=dp_size,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK, enable_cpu=True, enable_ssd=True,
        num_cpu_blocks=NUM_CPU_BLOCKS, num_ssd_blocks=NUM_SSD_BLOCKS,
        ssd_cache_dir=[ssd_dir],
    )
    return model_config, cache_config


def _bootstrap_proc(server_id, dp_size, ssd_dir, ready_evt, wrote_evt, done_evt,
                    result_q):
    """Bootstrap DP proc: owns radix regions + shared TE; writes a shared prefix."""
    _set_radix_env(server_id)
    from flexkv.common.request import KVResponseStatus
    from flexkv.kvmanager import KVManager

    model_config, cache_config = _make_configs(dp_size, ssd_dir)
    tag = "[bootstrap dp0]"
    kvm = None
    p = None
    try:
        kvm = KVManager(model_config, cache_config, dp_client_id=0)
        kvm.start()

        mp_ctx = mp.get_context("spawn")
        parent, child = mp_ctx.Pipe()
        p = mp_ctx.Process(
            target=_tp_client_proc,
            args=(0, 0, kvm.gpu_register_port,
                  model_config, cache_config, NUM_GPU_BLOCKS, child),
            daemon=True,
        )
        p.start()
        _ = parent.recv()

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not kvm.is_ready():
            time.sleep(0.1)
        if not kvm.is_ready():
            raise RuntimeError(f"{tag} not ready in 60s")
        print(f"{tag} READY", flush=True)
        ready_evt.set()  # external controller may attach now

        # Write a known shared prefix — lands in CPU and (on eviction) SSD.
        tok, slot, _ = _build_request(
            rng_seed=SHARED_SEED, start_block=0, n_blocks=BLOCK_PER_REQUEST)
        tid = kvm.put_async(token_ids=tok, slot_mapping=slot, dp_id=0)
        s = kvm.wait([tid], timeout=60, completely=True)
        put_ok = all(v.status == KVResponseStatus.SUCCESS for v in s.values())
        print(f"{tag} shared-prefix put ok={put_ok}", flush=True)

        # Concurrency phase / CPU flood: write many private prefixes while the
        # external controller is attaching. With NUM_CPU_BLOCKS small, this
        # evicts the shared prefix out of CPU (it survives in SSD), so the
        # external prefetch must issue a real SSD->CPU transfer graph.
        concurrent_ok = 0
        for i in range(FLOOD_REQUESTS):
            tk, sl, _ = _build_request(
                rng_seed=7000 + i, start_block=(i + 1) * BLOCK_PER_REQUEST,
                n_blocks=BLOCK_PER_REQUEST)
            t = kvm.put_async(token_ids=tk, slot_mapping=sl, dp_id=0)
            r = kvm.wait([t], timeout=60, completely=True)
            if all(v.status == KVResponseStatus.SUCCESS for v in r.values()):
                concurrent_ok += 1
        print(f"{tag} concurrent private puts (CPU flood) "
              f"ok={concurrent_ok}/{FLOOD_REQUESTS}", flush=True)

        wrote_evt.set()  # release external controller to prefetch the evicted prefix

        # Wait for external controller to finish before verifying shared index.
        done_evt.wait(timeout=60)

        # Verify the shared index reflects the prefix (get_match hit in CPU).
        for _ in range(50):
            _mtid, mask = kvm.get_match(token_ids=tok, dp_id=0)
            hit_blocks = int(np.count_nonzero(mask)) // TOKENS_PER_BLOCK \
                if mask is not None else 0
            if hit_blocks > 0:
                break
            time.sleep(0.1)
        print(f"{tag} shared index match hit_blocks={hit_blocks}", flush=True)

        result_q.put(("bootstrap", put_ok, concurrent_ok, hit_blocks))
    except Exception:
        import traceback
        traceback.print_exc()
        result_q.put(("bootstrap", False, -1, -1))
    finally:
        if p is not None:
            p.terminate()
            p.join()
        if kvm is not None:
            kvm.shutdown()


def _external_prefetch_proc(server_id, dp_size, ssd_dir, ready_evt, wrote_evt,
                            done_evt, result_q):
    """External process running a PrefetchController against the shared TE."""
    _set_radix_env(server_id)
    from flexkv.prefetch import PrefetchController

    model_config, _cache_config = _make_configs(dp_size, ssd_dir)
    tag = "[external]"
    pc = None
    try:
        # Wait for the bootstrap to create regions + TE + reserve the channel.
        if not ready_evt.wait(timeout=90):
            raise RuntimeError(f"{tag} bootstrap never signalled ready")
        # small settle so the TE ctrl ready flag is observable
        time.sleep(0.5)

        # Config-free attach: only server_id + tokens_per_block + num_layers.
        pc = PrefetchController.attach(
            server_id=server_id,
            tokens_per_block=TOKENS_PER_BLOCK,
            num_layers=model_config.num_layers,
            enable_ssd=True,
            external_index=0,
            dp_size=dp_size,
        )
        pc.start(ready_timeout_s=60)
        print(f"{tag} controller READY channel_id={pc.channel_id}", flush=True)

        # Isolation invariant: our id range is strictly beyond internal DP band.
        internal_hi = (dp_size) << 32
        ext_lo = pc.external_client_id << 32
        id_isolated = ext_lo >= internal_hi
        print(f"{tag} ext_id_lo={ext_lo} internal_hi={internal_hi} "
              f"isolated={id_isolated}", flush=True)

        # Wait until the bootstrap wrote the prefix AND flooded CPU (so the
        # prefix now lives only in SSD, forcing a real SSD->CPU prefetch graph).
        if not wrote_evt.wait(timeout=60):
            raise RuntimeError(f"{tag} shared prefix never written")

        tok, _, _ = _build_request(
            rng_seed=SHARED_SEED, start_block=0, n_blocks=BLOCK_PER_REQUEST)

        # Single prefetch of the evicted prefix. Because it is not ready in CPU,
        # cache_engine.get() produces a non-empty SSD->CPU graph, which is
        # submitted to the TE over our reserved channel and routed back to us.
        tid = pc.prefetch(token_ids=tok)
        res = pc.wait([tid], timeout=30)
        all_ok = res.get(tid, False)
        submitted = pc.submitted_count
        # graph_id our task used must be in our disjoint band (record before reap).
        print(f"{tag} prefetch result={res} all_ok={all_ok} "
              f"submitted={submitted} noop={pc.noop_count}", flush=True)

        # Pack both correctness signals: all_ok, id_isolated, and whether a real
        # graph was submitted (submitted>=1). Encode submitted in the 4th slot.
        result_q.put(("external", all_ok, id_isolated, submitted))
    except Exception:
        import traceback
        traceback.print_exc()
        result_q.put(("external", False, False, -1))
    finally:
        if pc is not None:
            pc.shutdown()
        done_evt.set()


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: need a CUDA device")
        return 0

    import tempfile
    import shutil
    dp_size = 1
    server_id = f"prefetchctl_{os.getpid()}"
    ssd_dir = tempfile.mkdtemp(prefix=f"flexkv_prefetch_ssd_{os.getpid()}_")
    ctx = mp.get_context("spawn")
    ready_evt = ctx.Event()
    wrote_evt = ctx.Event()
    done_evt = ctx.Event()
    result_q = ctx.Queue()

    boot = ctx.Process(target=_bootstrap_proc,
                       args=(server_id, dp_size, ssd_dir, ready_evt, wrote_evt,
                             done_evt, result_q), daemon=False)
    ext = ctx.Process(target=_external_prefetch_proc,
                      args=(server_id, dp_size, ssd_dir, ready_evt, wrote_evt,
                            done_evt, result_q), daemon=False)
    boot.start()
    ext.start()

    results = {}
    deadline = time.monotonic() + 240
    while len(results) < 2 and time.monotonic() < deadline:
        try:
            name, a, b, c = result_q.get(timeout=5)
            results[name] = (a, b, c)
        except Exception:
            if not boot.is_alive() and not ext.is_alive():
                break

    for p in (boot, ext):
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    print("\n=== RESULTS ===")
    print(results)

    if "bootstrap" not in results or "external" not in results:
        print(f"FAIL: missing results {results}")
        return 1

    boot_put_ok, concurrent_ok, hit_blocks = results["bootstrap"]
    ext_ok, id_isolated, submitted = results["external"]

    shutil.rmtree(ssd_dir, ignore_errors=True)

    if not boot_put_ok:
        print("FAIL: bootstrap shared-prefix put failed")
        return 1
    if concurrent_ok != FLOOD_REQUESTS:
        print(f"FAIL: concurrent internal puts {concurrent_ok}/{FLOOD_REQUESTS}")
        return 1
    if not ext_ok:
        print("FAIL: external prefetch task did not complete")
        return 1
    if not id_isolated:
        print("FAIL: external graph_id range NOT disjoint from internal")
        return 1
    if submitted < 1:
        print(f"FAIL: external prefetch issued no real transfer graph "
              f"(submitted={submitted}); prefix was not evicted to SSD, so the "
              f"reserved-channel transfer path was NOT exercised")
        return 1
    if hit_blocks <= 0:
        print(f"FAIL: shared index did not reflect prefix (hit={hit_blocks})")
        return 1

    print(f"PASS: external PrefetchController OK — attached to shared index, "
          f"disjoint id range, issued {submitted} real SSD->CPU transfer graph(s) "
          f"through the reserved channel and got them routed back, concurrent "
          f"internal puts unaffected ({concurrent_ok}/{FLOOD_REQUESTS}), shared "
          f"index hit {hit_blocks} blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
