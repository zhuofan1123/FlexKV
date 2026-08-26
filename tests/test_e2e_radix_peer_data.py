"""End-to-end DATA check for the radixshmem peer path.

The 2-rank tests in `tests/test_radix_shmem_engine.py` cover the CONTROL plane
only: the writer publishes hashes without ever filling a block, and the reader
just inspects the spliced match (slot ids, owner ids) and releases it. Nothing
crosses the data plane there. This file closes that gap — two full FlexKV nodes,
one radixshmem cluster, one mooncake/Redis data plane, and a byte comparison of
what the reader's GPU ends up holding.

Topology (single host, two processes, two GPUs):

  * both nodes bootstrap their OWN shm radix regions (distinct
    FLEXKV_SHM_RADIX_ID, hence distinct region names) under one shared etcd
    namespace with world_size=2, so shmradix hands them dense cluster ranks --
    which become their FlexKV node ids -- and an RHT to route by; they share the
    loopback bootstrap address and take a distinct SHMRADIX_NODE_NAME each, since
    that name is the identity shmradix registers under in etcd;
  * each node registers its CPU block buffer with its own mooncake engine and
    publishes `meta:<node_id>` (mooncake addr + buffer base) to a shared Redis,
    which is how a peer read turns a peer's slot id into an RDMA/TCP transfer;
  * both tiers and both peer routes are enabled on both nodes, so nothing about
    the route is pinned by config. Which source served a block is read off the
    BYTES instead: the two nodes write different content for the same block, so
    a span boundary is where the content's writer changes.

What this validates, over two independent windows:
  1. a window node 1 holds nothing of comes back whole off node 0 -- a
     cross-node match over RDMA, then a PEERH2H that reads the RIGHT bytes:
     `ShmRadixMatch.peer_range` / `peer_node_ids` addressing a peer's slots is
     checked against the actual content, not just against expected ids;
  2. a window whose four prefixes are laid out
         node 1 CPU < node 0 CPU < node 1 SSD < node 0 SSD (= the whole window)
     comes back as the four-way splice `_shm_get_spans` exists for, each quarter
     carrying the bytes of the node whose tier it must have come from. Serving
     any of them off the wrong tier or the wrong node is a mismatch, not a pass.

Leaving a tier holding LESS than the tier behind it takes a CPU-tree reset: a
PUT is write-through, so it gives CPU and SSD the same prefix, and only a
shorter re-PUT after dropping the CPU tree separates them. That reset also wipes
the RHT shard the node hosts -- other nodes' entries included -- so each node
resets before publishing anything that must survive: node 1 lays its layers down
before node 0 publishes at all, and node 0 publishes the first window after its
own reset.

The spliced case also needs an RHT bucket with room for more than one owner, so
this sets SHMRADIX_RHT_SLOTS=4. Routing to a peer goes through the bucket of a
tree node's FIRST hash, a query republishes its own local hit into that bucket
before probing it, and at shmradix's default of one slot per bucket that write
is a blind overwrite -- so a reader holding the prefix always finds only itself
there and no peer tail can be routed at all. Nothing shorter than the 128-block
register chunk has a second probe position to fall back on.

Requires: >=2 CUDA devices, an ACTIVE RDMA port, `redis-server` on PATH, and an
etcd endpoint in FLEXKV_TEST_RADIX_REGISTRY (the radix cluster's only
rendezvous). Run inside the container:

    FLEXKV_TEST_RADIX_REGISTRY=http://127.0.0.1:2379 \\
    PYTHONPATH=/root/FlexKV \\
    LD_LIBRARY_PATH=$TORCH_LIB:$CUDART12_LIB:$LD_LIBRARY_PATH \\
    python3 tests/test_e2e_radix_peer_data.py

The mooncake engine must be built with `-DUSE_REDIS=ON`: both the redis
metadata backend below and FlexKV's `meta:<node_id>` address book live on the
one redis-server this test starts. A wheel build (no redis plugin) aborts at
`Unable to find metadata storage plugin redis`, and it also links
libcudart.so.12 -- hence `CUDART12_LIB`, the nvidia/cuda_runtime/lib wheel
directory, which a cu13 torch install does not otherwise provide.

Each node bootstraps two distributed trees, one per tier; they get separate etcd
namespaces and separate RHTs, so they do not collide.
"""
from __future__ import annotations

import contextlib
import glob
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch


TOKENS_PER_BLOCK = 16
NUM_GPU_BLOCKS = 64
NUM_CPU_BLOCKS = 512
NUM_SSD_BLOCKS = 512
# Two requests, on distinct GPU blocks and distinct tokens so that what a GET of
# the first promotes into node 1's tree cannot serve any part of the second.
NUM_REQUEST_BLOCKS = 16
FIRST_BLOCK = 8
SPLICED_FIRST_BLOCK = FIRST_BLOCK + NUM_REQUEST_BLOCKS
PATTERN_SEED = 0xBEEF
SPLICED_SEED = 0xCAFE
WORLD_SIZE = 2

# Prefix each tier of each node is left holding of the second window, and the
# resulting spans of its GET: read in place, PEERH2H, DISK2H, PEERSSD2H.
LOCAL_CPU_BLOCKS = 3
PEER_CPU_BLOCKS = 6
LOCAL_SSD_BLOCKS = 10
SPLICED_SPANS = (
    ("local_cpu", 0, LOCAL_CPU_BLOCKS, 1),
    ("peer_cpu", LOCAL_CPU_BLOCKS, PEER_CPU_BLOCKS, 0),
    ("local_ssd", PEER_CPU_BLOCKS, LOCAL_SSD_BLOCKS, 1),
    ("peer_ssd", LOCAL_SSD_BLOCKS, NUM_REQUEST_BLOCKS, 0),
)


def node_tag_for(run_id: str, rank: int) -> str:
    """Per-node tag of one of the co-located nodes.

    It names this node's shm regions (FLEXKV_SHM_RADIX_ID) and the identity
    it registers in etcd (SHMRADIX_NODE_NAME); both must differ between the two
    nodes or the second overwrites the first and the cluster never forms.
    """
    return f"{run_id}_r{rank}"


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _free_port_block(count: int) -> int:
    """First of `count` consecutive bindable ports.

    PEERSSD2H uses `local_zmq_port` for the metadata channel and the next port
    for completion status, so a rank needs a pair -- and two ranks must not have
    one rank's pair straddle the other's base.
    """
    for _ in range(64):
        base = _free_port()
        sockets = []
        try:
            for offset in range(count):
                sock = socket.socket()
                sock.bind(("127.0.0.1", base + offset))
                sockets.append(sock)
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError(f"no {count} consecutive free ports")


def _active_rdma_devices() -> list:
    """RDMA devices with at least one ACTIVE port, honoring the env override."""
    override = os.getenv("FLEXKV_RADIX_RDMA_DEV", "").strip()
    names = ([override] if override
             else sorted(os.path.basename(p)
                         for p in glob.glob("/sys/class/infiniband/*")))
    active = []
    for name in names:
        for state in glob.glob(f"/sys/class/infiniband/{name}/ports/*/state"):
            with contextlib.suppress(OSError):
                with open(state) as handle:
                    if "ACTIVE" in handle.read():
                        active.append(name)
                        break
    return active


def _block_pattern(layer: int, block_id: int, shape, dtype,
                   writer: int = 0) -> torch.Tensor:
    """Deterministic content for one (layer, block) as written by node `writer`.

    Random rather than structured so that a transfer landing on the wrong block
    or the wrong layer cannot accidentally compare equal, and generated on the
    CPU so both nodes derive it identically without talking to each other.
    `writer` makes the same block differ between the two nodes, which is how a
    byte comparison alone can tell which node's copy a GET served.
    """
    generator = torch.Generator().manual_seed(
        PATTERN_SEED + writer * 1_000_003 + block_id * 128 + layer)
    return torch.randn(tuple(shape), generator=generator).to(dtype)


def _write_pattern(gpu_tensors, block_ids, writer: int = 0) -> None:
    for layer, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            block = tensor[:, block_id]
            block.copy_(_block_pattern(layer, int(block_id), block.shape,
                                       tensor.dtype, writer))
    torch.cuda.synchronize()


def _clear_blocks(gpu_tensors, block_ids) -> None:
    """Zero the blocks the GET is supposed to fill, so a no-op fails the check."""
    for tensor in gpu_tensors:
        for block_id in block_ids:
            tensor[:, block_id].zero_()
    torch.cuda.synchronize()


def _mismatched_blocks(gpu_tensors, block_ids, writer: int = 0) -> list:
    """(layer, block) pairs whose content is not what node `writer` wrote."""
    bad = []
    for layer, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            got = tensor[:, block_id].cpu()
            want = _block_pattern(layer, int(block_id), got.shape, got.dtype,
                                  writer)
            if not torch.equal(got, want):
                bad.append((layer, int(block_id)))
    return bad


def _build_request(num_blocks: int, first_block: int, seed: int):
    """token_ids / slot_mapping for `num_blocks` GPU blocks starting at first."""
    rng = np.random.default_rng(seed)
    block_ids = np.arange(first_block, first_block + num_blocks, dtype=np.int64)
    slot_mapping = (np.repeat(block_ids, TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK
                    + np.tile(np.arange(TOKENS_PER_BLOCK), num_blocks))
    token_ids = rng.integers(0, 32000, size=slot_mapping.shape, dtype=np.int64)
    return token_ids, slot_mapping, block_ids


def _windows():
    """The peer-only window and the spliced one, in the order the reader uses
    them. Both nodes derive them the same way instead of exchanging them."""
    return (_build_request(NUM_REQUEST_BLOCKS, FIRST_BLOCK, PATTERN_SEED),
            _build_request(NUM_REQUEST_BLOCKS, SPLICED_FIRST_BLOCK,
                           SPLICED_SEED))


def _put_prefix(kvm, token_ids, slot_mapping, num_blocks: int) -> bool:
    """PUT the window's first `num_blocks` blocks; True if the task completed."""
    from flexkv.common.request import KVResponseStatus

    num_tokens = num_blocks * TOKENS_PER_BLOCK
    task_id = kvm.put_async(token_ids=token_ids[:num_tokens],
                            slot_mapping=slot_mapping[:num_tokens])
    status = kvm.wait([task_id], timeout=120, completely=True)
    return all(r.status == KVResponseStatus.SUCCESS for r in status.values())


def _get_once(kvm, token_ids, slot_mapping) -> int:
    """One GET; how many blocks it matched, 0 if it did not succeed."""
    from flexkv.common.request import KVResponseStatus

    task_id = kvm.get_async(token_ids=token_ids, slot_mapping=slot_mapping)
    response = kvm.wait([task_id], timeout=120, completely=True)[task_id]
    if response.status != KVResponseStatus.SUCCESS or response.return_mask is None:
        return 0
    return int(np.count_nonzero(response.return_mask)) // TOKENS_PER_BLOCK


def _get_until_full_hit(kvm, token_ids, slot_mapping, want_blocks: int,
                        timeout: float = 60.0) -> int:
    """GET until the whole window is matched, returning the last hit count.

    The peer's RHT publication and its Redis meta are both async, so early
    attempts can legitimately miss; a partial count is reported rather than
    raised so the caller can say how far it got.
    """
    hit = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hit = _get_once(kvm, token_ids, slot_mapping)
        if hit >= want_blocks:
            break
        time.sleep(0.5)
    return hit


def _write_mooncake_config(path: str, port: int, metadata_url: str,
                           protocol: str, device_name: str) -> None:
    """One mooncake engine config.

    The metadata backend is `redis`, pointed at the same redis-server this
    test already starts for FlexKV's own address book (`meta:<node_id>`), so
    a single helper process covers both. This requires a mooncake build
    compiled with `-DUSE_REDIS=ON`; a build without it aborts at
    `Unable to find metadata storage plugin redis`.
    """
    with open(path, "w") as handle:
        json.dump({
            "engine_ip": "127.0.0.1",
            "engine_port": port,
            "metadata_backend": "redis",
            "metadata_server": metadata_url,
            "metadata_server_auth": "",
            "protocol": protocol,
            "device_name": device_name if protocol == "rdma" else "",
        }, handle)


def _tp_client_proc(server_recv_port, model_config, cache_config,
                    num_gpu_blocks, child_conn):
    """Holds the node's GPU tensors and hands their IPC handles back."""
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.common.memory_handle import TensorSharedHandle
    from flexkv.server.client import KVTPClient

    tp_client = KVTPClient(server_recv_port, 0, 0)
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
        torch.zeros(size=tuple(gpu_layout.kv_shape[1:]),
                    dtype=model_config.dtype).cuda(0)
        for _ in range(model_config.num_layers)
    ]
    tp_client.register_to_server(gpu_blocks, gpu_layout)
    child_conn.send([TensorSharedHandle(t) for t in gpu_blocks])
    child_conn.close()
    while True:
        time.sleep(1)


def _node_proc(rank, gpu_id, run_id, registry, cluster_id, rdma_dev,
               redis_port, mooncake_config, zmq_port, ssd_dir,
               reader_ready, written, read_done, result_q):
    """One FlexKV node: rank 0 writes the prefixes, rank 1 reads them back."""
    # Before any CUDA context exists, so each node drives a different device
    # while still addressing it as device 0 (KVTPClient derives the device id
    # from dp_client_id/tp_rank, which are 0 on both nodes).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    node_tag = node_tag_for(run_id, rank)
    recv_port = f"ipc:///tmp/flexkv_{node_tag}"
    os.environ.update({
        "FLEXKV_RADIX_SHMEM": "1",
        "FLEXKV_SHM_RADIX_ID": node_tag,
        "FLEXKV_RADIX_WORLD_SIZE": str(WORLD_SIZE),
        "FLEXKV_RADIX_REGISTRY": registry,
        # shmradix's own namespace, shared by the two nodes: it keeps concurrent
        # runs on one etcd from joining each other's cluster.
        "SHMRADIX_CLUSTER_ID": cluster_id,
        # etcd keys membership by node identity, which defaults to the bind IP the
        # co-located nodes share -- so override it per node.
        "SHMRADIX_NODE_NAME": node_tag,
        # Single host: peers dial back over loopback (the port is OS-assigned).
        "FLEXKV_RADIX_RPC_ADDRESS": "127.0.0.1",
        "FLEXKV_RADIX_RDMA_DEV": rdma_dev,
        "FLEXKV_ENABLE_MPS": "0",
        "FLEXKV_SERVER_RECV_PORT": recv_port,
        "MOONCAKE_CONFIG_PATH": mooncake_config,
        # Log the peer query so a failure shows whether the match or the
        # transfer is the part that went wrong.
        "FLEXKV_TRACE_RADIX_PEER": "1",
        # Needed for the spliced case; see the module docstring.
        "SHMRADIX_RHT_SLOTS": os.getenv("SHMRADIX_RHT_SLOTS", "4"),
    })

    from flexkv.common.config import (CacheConfig, GLOBAL_CONFIG_FROM_ENV,
                                      ModelConfig)
    from flexkv.kvmanager import KVManager

    # The module-level config object is built from env at import time, which the
    # assignments above precede -- but this process may have imported it earlier
    # through a parent module, so set the fields that matter explicitly.
    GLOBAL_CONFIG_FROM_ENV.radix_shmem = True
    GLOBAL_CONFIG_FROM_ENV.shm_radix_id = node_tag
    GLOBAL_CONFIG_FROM_ENV.radix_world_size = WORLD_SIZE
    GLOBAL_CONFIG_FROM_ENV.radix_registry = registry
    GLOBAL_CONFIG_FROM_ENV.radix_rpc_address = "127.0.0.1"
    GLOBAL_CONFIG_FROM_ENV.radix_rdma_dev = rdma_dev
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    GLOBAL_CONFIG_FROM_ENV.server_recv_port = recv_port

    tag = f"[node r{rank}]"
    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=128,
        dtype=torch.float16, use_mla=False, tp_size=1, dp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=True, enable_remote=False,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        num_ssd_blocks=NUM_SSD_BLOCKS,
        ssd_cache_dir=ssd_dir,
        # The data plane: mooncake for the bytes, Redis for the address book.
        enable_p2p_cpu=True,
        enable_p2p_ssd=True,
        redis_host="127.0.0.1", redis_port=redis_port,
        local_ip="127.0.0.1",
        local_zmq_ip="127.0.0.1", local_zmq_port=zmq_port,
        mooncake_config_path=mooncake_config,
    )

    report = {"rank": rank}
    kvm = None
    tp_proc = None
    try:
        kvm = KVManager(model_config, cache_config, dp_client_id=0)
        kvm.start()

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        tp_proc = ctx.Process(
            target=_tp_client_proc,
            args=(kvm.gpu_register_port, model_config, cache_config,
                  NUM_GPU_BLOCKS, child_conn),
            daemon=True,
        )
        tp_proc.start()
        handles = parent_conn.recv()
        gpu_tensors = [handle.get_tensor() for handle in handles]

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not kvm.is_ready():
            time.sleep(0.1)
        if not kvm.is_ready():
            raise RuntimeError("KVManager not ready in 120s")
        report["node_id"] = int(cache_config.distributed_node_id)
        print(f"{tag} READY as FlexKV node id {report['node_id']}", flush=True)

        peer_win, spliced_win = _windows()
        peer_tokens, peer_slots, peer_blocks = peer_win
        tokens, slots, blocks = spliced_win

        if rank == 1:
            # This node's two layers, before node 0 publishes anything: the
            # reset takes the RHT shard this node hosts down with it.
            _write_pattern(gpu_tensors, blocks[:LOCAL_SSD_BLOCKS], writer=1)
            ok = _put_prefix(kvm, tokens, slots, LOCAL_SSD_BLOCKS)
            kvm._clear_cpu_cache()
            report["layers_put_ok"] = _put_prefix(
                kvm, tokens, slots, LOCAL_CPU_BLOCKS) and ok
            print(f"{tag} layers ok={report['layers_put_ok']}", flush=True)
            reader_ready.set()
            if not written.wait(180):
                raise TimeoutError("writer did not publish in 180s")

            # 1) Entirely off the peer: this node holds nothing of this window.
            _clear_blocks(gpu_tensors, peer_blocks)
            hit = _get_until_full_hit(kvm, peer_tokens, peer_slots,
                                      NUM_REQUEST_BLOCKS)
            report["peer_hit_blocks"] = hit
            report["peer_mismatched"] = _mismatched_blocks(gpu_tensors,
                                                           peer_blocks)
            print(f"{tag} peer GET: hit_blocks={hit} "
                  f"mismatched={len(report['peer_mismatched'])}", flush=True)

            # 2) The four-way splice, in one shot: a GET promotes what it staged
            # into this node's CPU tree, so a retry would not see the layering.
            _clear_blocks(gpu_tensors, blocks)
            report["spliced_hit_blocks"] = _get_once(kvm, tokens, slots)
            for label, first, last, writer in SPLICED_SPANS:
                report[f"spliced_{label}_mismatched"] = _mismatched_blocks(
                    gpu_tensors, blocks[first:last], writer=writer)
            print(f"{tag} spliced GET: hit_blocks={report['spliced_hit_blocks']}"
                  + "".join(f" {label}_mismatched="
                            f"{len(report[f'spliced_{label}_mismatched'])}"
                            for label, *_ in SPLICED_SPANS), flush=True)
            read_done.set()
        else:
            if not reader_ready.wait(180):
                raise TimeoutError("reader did not lay down its layers in 180s")
            # A PUT is write-through, so this leaves both tiers holding the whole
            # window; the reset plus the shorter re-PUT is what parts them.
            _write_pattern(gpu_tensors, blocks)
            ok = _put_prefix(kvm, tokens, slots, NUM_REQUEST_BLOCKS)
            kvm._clear_cpu_cache()
            ok = _put_prefix(kvm, tokens, slots, PEER_CPU_BLOCKS) and ok
            # After the reset, so the reset cannot take it with it.
            _write_pattern(gpu_tensors, peer_blocks)
            report["put_ok"] = _put_prefix(kvm, peer_tokens, peer_slots,
                                           NUM_REQUEST_BLOCKS) and ok
            print(f"{tag} put ok={report['put_ok']}", flush=True)
            written.set()
            # Stay up: the reader reads THIS process's registered buffer and its
            # SSD files, and shutdown would drop both from the address book.
            if not read_done.wait(240):
                raise TimeoutError("reader did not finish in 240s")
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        print(f"{tag} FAILED\n{report['error']}", flush=True)
        reader_ready.set()
        written.set()
        read_done.set()
    finally:
        if tp_proc is not None:
            tp_proc.terminate()
            tp_proc.join(timeout=10)
        if kvm is not None:
            with contextlib.suppress(Exception):
                kvm.shutdown()
        result_q.put(report)


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        print(f"SKIP: need >={WORLD_SIZE} CUDA devices")
        return 0
    registry = os.getenv("FLEXKV_TEST_RADIX_REGISTRY", "")
    if not registry:
        print("SKIP: set FLEXKV_TEST_RADIX_REGISTRY=<etcd endpoint>; etcd is "
              "the radix cluster's only rendezvous")
        return 0
    devices = _active_rdma_devices()
    if not devices:
        print("SKIP: no ACTIVE RDMA port (checked /sys/class/infiniband/*)")
        return 0
    if shutil.which("redis-server") is None:
        print("SKIP: redis-server not on PATH; it carries the data plane's "
              "address book")
        return 0

    rdma_dev = devices[0]
    protocol = os.getenv("FLEXKV_TEST_MOONCAKE_PROTOCOL", "tcp")
    run_id = f"peerdata{os.getpid()}"
    cluster_id = f"flexkv_peer_data_{os.getpid()}"
    redis_port = _free_port()
    redis_dir = f"/tmp/flexkv_peer_data_{os.getpid()}"
    os.makedirs(redis_dir, exist_ok=True)
    ssd_dirs = []
    for rank in range(WORLD_SIZE):
        ssd_dir = f"{redis_dir}/ssd_r{rank}"
        os.makedirs(ssd_dir, exist_ok=True)
        ssd_dirs.append(ssd_dir)

    redis_proc = subprocess.Popen(
        ["redis-server", "--port", str(redis_port), "--save", "",
         "--appendonly", "no", "--dir", redis_dir],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    meta_url = f"redis://127.0.0.1:{redis_port}"
    configs = []
    for rank in range(WORLD_SIZE):
        path = f"{redis_dir}/mooncake_r{rank}.json"
        _write_mooncake_config(path, _free_port(), meta_url, protocol,
                               rdma_dev)
        configs.append(path)

    ctx = mp.get_context("spawn")
    reader_ready, written, read_done = ctx.Event(), ctx.Event(), ctx.Event()
    result_q = ctx.Queue()
    procs = []
    try:
        import redis as redis_py
        client = redis_py.Redis(host="127.0.0.1", port=redis_port)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                if client.ping():
                    break
            time.sleep(0.2)
        else:
            print("FAIL: redis-server did not come up")
            return 1

        print(f"redis on 127.0.0.1:{redis_port}, "
              f"mooncake metadata {meta_url}, etcd {registry}, "
              f"rdma {rdma_dev}, mooncake protocol {protocol}", flush=True)
        # Two consecutive ports per rank, non-overlapping across ranks.
        zmq_base = _free_port_block(WORLD_SIZE * 2)
        for rank in range(WORLD_SIZE):
            proc = ctx.Process(
                target=_node_proc,
                args=(rank, rank, run_id, registry, cluster_id, rdma_dev,
                      redis_port, configs[rank], zmq_base + rank * 2,
                      ssd_dirs[rank], reader_ready, written, read_done,
                      result_q),
                daemon=False,
            )
            proc.start()
            procs.append(proc)

        reports = {}
        deadline = time.monotonic() + 420
        while len(reports) < WORLD_SIZE and time.monotonic() < deadline:
            try:
                report = result_q.get(timeout=5)
                reports[report["rank"]] = report
            except Exception:
                if not any(proc.is_alive() for proc in procs):
                    break
    finally:
        for proc in procs:
            proc.join(timeout=20)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        redis_proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            redis_proc.wait(timeout=10)
        from flexkv.common.transfer import DeviceType
        from flexkv.server.shm_radix_bootstrap import shm_name_for
        for rank in range(WORLD_SIZE):
            for device in (DeviceType.CPU, DeviceType.SSD):
                name = shm_name_for(device, node_tag_for(run_id, rank))
                # A distributed region appends shmradix's node identity and its RHT
                # shards extend that again, so sweep both backings by prefix.
                for root in ("/dev/shm", "/dev/hugepages"):
                    for stale in glob.glob(f"{root}{name}*"):
                        with contextlib.suppress(OSError):
                            os.unlink(stale)
        shutil.rmtree(redis_dir, ignore_errors=True)

    print("\n=== RESULTS ===")
    for rank in sorted(reports):
        printable = {k: v for k, v in reports[rank].items() if k != "error"}
        print(rank, printable)
        if "error" in reports[rank]:
            print(reports[rank]["error"])

    if len(reports) < WORLD_SIZE:
        print(f"FAIL: only {len(reports)}/{WORLD_SIZE} nodes reported")
        return 1
    if any("error" in report for report in reports.values()):
        print("FAIL: a node raised")
        return 1
    writer, reader = reports[0], reports[1]
    if writer.get("node_id") == reader.get("node_id"):
        print(f"FAIL: both nodes registered as node id {writer.get('node_id')}")
        return 1
    if not writer.get("put_ok"):
        print("FAIL: writer's put did not complete")
        return 1
    if not reader.get("layers_put_ok"):
        print("FAIL: reader's puts of its own two layers did not complete")
        return 1
    for key, what in (("peer", "the whole window off node 0"),
                      ("spliced", "the four-way spliced window")):
        hit = reader.get(f"{key}_hit_blocks", 0)
        if hit < NUM_REQUEST_BLOCKS:
            print(f"FAIL: {what}: matched {hit}/{NUM_REQUEST_BLOCKS} blocks")
            return 1
    bad = reader.get("peer_mismatched")
    if bad:
        print(f"FAIL: the whole window off node 0: {len(bad)} (layer, block) "
              f"pairs hold the wrong bytes, e.g. {bad[:5]}")
        return 1
    for label, first, last, writer in SPLICED_SPANS:
        bad = reader.get(f"spliced_{label}_mismatched")
        if bad:
            print(f"FAIL: the spliced window's {label} span (blocks {first}-"
                  f"{last - 1}) does not hold node {writer}'s bytes: "
                  f"{len(bad)} (layer, block) pairs, e.g. {bad[:5]}")
            return 1

    print(f"PASS: node 1 pulled {NUM_REQUEST_BLOCKS} blocks off node 0 byte for "
          f"byte, and on a second window a four-way splice put local CPU / peer "
          f"CPU / local SSD / peer SSD blocks "
          f"({' / '.join(str(last - first) for _, first, last, _ in SPLICED_SPANS)}"
          f") each at the right GPU offset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
