# SPDX-License-Identifier: Apache-2.0
"""
Bootstrap for radixshmem-backed CacheEngine.

Per-device-type RadixTree (CPU / SSD / REMOTE) lives in its own POSIX shm
region. The first DP process (instance 0, dp_client_id 0) creates the regions
via `shmradix.RadixServer`; all others attach via `shmradix.RadixClient`.

Naming convention:
    /flexkv_radix_{server_id}_{cpu|ssd|remote}[_r{rank}]

`server_id` defaults to a fixed token but is overridable so multiple FlexKV
instances on the same host don't collide. The `_r{rank}` suffix is appended only
when `world_size > 1`, so two simulated ranks can share one host without
colliding. The name is purely LOCAL: peers exchange shm base address, rkey and
segment offsets (and even each RHT shard's shm name) through the cluster
manifest at bootstrap, so it never has to agree across ranks.

In distributed mode `RadixServer` builds the name itself as
`config.name + "_" + config.node_name`, so this module sets `node_name = r{rank}`
to make the result deterministic — every DP process has to compute the same name
to attach as a client, and only the bootstrap process holds the server object.

Cluster membership goes through etcd, which is shmradix's only bootstrap path in
an RDMA build. Each tier bootstraps as an INDEPENDENT cluster (its own RHT and
its own rank space), so each gets its own etcd sub-namespace. Cluster rank is an
OUTPUT: etcd assigns dense ranks by sorted `/peers` key order, and callers read
it back via `RadixServer.rank()` / `RadixClient.rank()` rather than assuming the
configured `rank` survived.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType

try:
    import shmradix
except ImportError:  # pragma: no cover
    shmradix = None


_DEVICE_KIND_NAMES = {
    DeviceType.CPU: "cpu",
    DeviceType.SSD: "ssd",
    DeviceType.REMOTE: "remote",
}


def shm_radix_prefix(device_type: DeviceType, server_id: str) -> str:
    """The shm-name prefix shared by a region's owner and its clients.

    In distributed mode ``RadixServer`` appends ``"_" + node_name`` to this
    prefix itself, which is why the bootstrap hands it the bare prefix and sets
    ``node_name`` to the deterministic label ``node_name_for`` returns.
    """
    return f"/flexkv_radix_{server_id}_{_DEVICE_KIND_NAMES[device_type]}"


def node_name_for(rank: int) -> str:
    """This node's shmradix identity label.

    It has to be set explicitly: left empty, ``RadixServer`` derives it from the
    resolved bind IP (``node<ip>``), which two ranks on one host would collide on
    and which no other process can predict. The label names the shm region and
    the etcd ``/peers`` key, so it must be unique per node — but it is NOT the
    cluster rank, which etcd assigns during bootstrap.
    """
    return f"r{rank}"


def shm_name_for(device_type: DeviceType,
                 server_id: str,
                 *,
                 rank: int = 0,
                 world_size: int = 1) -> str:
    """POSIX shm name for the radix region of one device type.

    Mirrors what ``RadixServer.bootstrap()`` computes, so a client process can
    derive the name without talking to the owner.
    """
    prefix = shm_radix_prefix(device_type, server_id)
    if world_size <= 1:
        return prefix
    return f"{prefix}_{node_name_for(rank)}"


def cluster_id_for(device_type: DeviceType,
                   server_id: str,
                   cluster_id: str) -> str:
    """etcd namespace for one tier's cluster.

    Tiers bootstrap independently — separate RHT, separate rank space — so they
    must not share a ``/peers`` prefix or the membership gate would count a
    peer's CPU node toward the SSD cluster.
    """
    return f"{cluster_id}/{server_id}/{_DEVICE_KIND_NAMES[device_type]}"


def device_blocks_from_config(device_type: DeviceType,
                              cache_config: CacheConfig) -> int:
    if device_type == DeviceType.CPU:
        return cache_config.num_cpu_blocks
    if device_type == DeviceType.SSD:
        return cache_config.num_ssd_blocks
    if device_type == DeviceType.REMOTE:
        return cache_config.num_remote_blocks or 0
    return 0


def enabled_devices(cache_config: CacheConfig) -> Tuple[DeviceType, ...]:
    out = []
    if cache_config.enable_cpu:
        out.append(DeviceType.CPU)
    if cache_config.enable_ssd:
        out.append(DeviceType.SSD)
    if cache_config.enable_remote:
        out.append(DeviceType.REMOTE)
    return tuple(out)


class ShmRadixOwners:
    """Holder for the `RadixServer` instances created in the bootstrap process.

    The owner process must keep these alive for the lifetime of the FlexKV
    server (otherwise the shm regions are torn down). We attach them to a
    long-lived object (e.g. KVManager) so Python's GC doesn't reap them.
    """

    def __init__(self) -> None:
        self.servers: Dict[DeviceType, shmradix.RadixServer] = {}
        # Cluster rank etcd assigned this node, read back after bootstrap. All
        # tiers land in the same rank because they see the same sorted peer key
        # order; mismatch is checked and rejected in create_shm_radix_regions.
        self.cluster_rank: int = 0

    def add(self, device_type: DeviceType, server: shmradix.RadixServer) -> None:
        self.servers[device_type] = server

    def shutdown(self) -> None:
        # RadixServer destructor releases the shm region.
        self.servers.clear()


def create_shm_radix_regions(model_config: ModelConfig,
                             cache_config: CacheConfig,
                             server_id: str,
                             rank: int = 0,
                             world_size: int = 1,
                             registry: str = "",
                             cluster_id: str = "flexkv",
                             rpc_address: str = "",
                             rpc_interface: str = "",
                             rdma_dev: str = "",
                             gid_idx: int = 3,
                             bootstrap_timeout_sec: int = 120,
                             remote_op_transport: str = "dc",
                             data_pool_ratio: int = 8,
                             evict_ratio: float = 0.05,
                             background_evict: bool = True) -> ShmRadixOwners:
    """Called by the bootstrap (instance 0, dp 0) process to create the
    shm regions. Returns an owner handle that callers must keep alive.

    ``world_size > 1`` creates *distributed* regions: every node registers itself
    under the tier's etcd namespace, the leader waits for all ``world_size`` of
    them and assigns dense ranks, RDMA connection info is exchanged, and from
    then on a query that outruns the local tree continues on a peer's tree over
    RDMA. The gate is collective — each node blocks in ``bootstrap()`` until the
    whole cluster has arrived.

    ``rank`` is only this node's local identity label. Read the cluster rank back
    off the returned handle (``owners.cluster_rank``); that is the number the peer
    data path addresses."""
    if shmradix is None:
        raise ImportError("shmradix not installed")

    if world_size > 1:
        if not registry:
            raise ValueError(
                "radixshmem distributed mode needs an etcd registry "
                "(FLEXKV_RADIX_REGISTRY, e.g. 'etcd://10.0.0.1:2379'): it is "
                "the only cluster bootstrap path shmradix has"
            )
        if not rpc_address and not rpc_interface:
            raise ValueError(
                "radixshmem distributed mode needs FLEXKV_RADIX_RPC_ADDRESS or "
                "FLEXKV_RADIX_RPC_INTERFACE — the bootstrap IP peers dial "
                "(use 0.0.0.0 for a single-host test)"
            )

    owners = ShmRadixOwners()
    cluster_ranks: Dict[DeviceType, int] = {}
    for dt in enabled_devices(cache_config):
        n_blocks = device_blocks_from_config(dt, cache_config)
        if n_blocks <= 0:
            continue
        cfg = shmradix.ShmConfig(
            # A radix node holds >= 1 block, so node count won't exceed the
            # block count — size the node pool to n_blocks.
            max_nodes=n_blocks,
            max_blocks=n_blocks,
            # Persist tokens_per_block into the region so any attacher (e.g. an
            # external prefetch controller) can recover it via
            # RadixClient.block_size() instead of being told out-of-band.
            block_size=cache_config.tokens_per_block,
            data_pool_ratio=data_pool_ratio,
            evict_ratio=evict_ratio,
            background_evict=background_evict,
        )
        name = shm_name_for(dt, server_id, rank=rank, world_size=world_size)
        flexkv_logger.info(
            f"creating shm radix region {name} "
            f"(max_nodes={cfg.max_nodes}, max_blocks={cfg.max_blocks}, "
            f"rank={rank}/{world_size})"
        )
        if world_size > 1:
            server_cfg = shmradix.RadixServerConfig()
            # RadixServer appends "_" + node_name to this prefix.
            server_cfg.name = shm_radix_prefix(dt, server_id)
            server_cfg.shm = cfg
            # Local identity, not the cluster rank. Set explicitly so the name is
            # deterministic for the client processes that must attach to it.
            server_cfg.node_name = node_name_for(rank)
            server_cfg.rank = rank
            server_cfg.world_size = world_size
            # The membership gate is max(num_shards, expected_min_nodes) and
            # expected_min_nodes is not exposed to Python — so num_shards is the
            # only way to make bootstrap wait for the whole cluster instead of
            # completing with one node. world_size is also what the shard map
            # defaults to, so this changes nothing else.
            server_cfg.num_shards = world_size
            # Each tier is its own cluster with its own rank space.
            server_cfg.registry = registry
            server_cfg.cluster_id = cluster_id_for(dt, server_id, cluster_id)
            server_cfg.rpc_address = rpc_address
            server_cfg.rpc_interface = rpc_interface
            server_cfg.remote_op_transport = remote_op_transport
            server_cfg.rdma_dev = rdma_dev
            server_cfg.gid_idx = gid_idx
            server_cfg.bootstrap_timeout_sec = bootstrap_timeout_sec
            # Unlike the single-node overload, this ctor does NOT create the shm
            # region — bootstrap() does, and it is not idempotent.
            server = shmradix.RadixServer(server_cfg)
            if not server.bootstrap():
                raise RuntimeError(
                    f"radixshmem cluster bootstrap failed for {name} "
                    f"(node_name={server_cfg.node_name}, "
                    f"world_size={world_size}, registry={registry}, "
                    f"cluster_id={server_cfg.cluster_id}); check that etcd is "
                    f"reachable, that all {world_size} nodes joined within "
                    f"{bootstrap_timeout_sec}s, and that shmradix was built "
                    f"with RDMA + etcd support"
                )
            if not server.is_distributed():
                raise RuntimeError(
                    f"radixshmem region {name} bootstrapped but reports "
                    f"world_size={server.world_size()}; the installed shmradix "
                    f"extension was built without RDMA (FLEXKV_NO_RDMA)"
                )
            actual_name = server.shm_name()
            if actual_name != name:
                # Our clients attach by the computed name, so a drift in
                # RadixServer's naming rule would leave them polling forever.
                raise RuntimeError(
                    f"radixshmem named the region {actual_name!r} but FlexKV "
                    f"clients will look for {name!r}; shm_name_for() no longer "
                    f"mirrors RadixServer.bootstrap()"
                )
            cluster_ranks[dt] = int(server.rank())
        else:
            server = shmradix.RadixServer(name, cfg)
            cluster_ranks[dt] = int(server.rank())
        owners.add(dt, server)

    # Every tier sees the same sorted peer key order, so they must agree. If they
    # don't, one node id can't stand for all tiers and the peer data path would
    # read from the wrong node.
    distinct = set(cluster_ranks.values())
    if len(distinct) > 1:
        raise RuntimeError(
            f"radixshmem assigned different cluster ranks per tier: "
            f"{ {_DEVICE_KIND_NAMES[k]: v for k, v in cluster_ranks.items()} }; "
            f"FlexKV needs one node id for all tiers"
        )
    owners.cluster_rank = distinct.pop() if distinct else rank
    if world_size > 1:
        flexkv_logger.info(
            f"radixshmem cluster bootstrap done: local label "
            f"{node_name_for(rank)} -> cluster rank {owners.cluster_rank} "
            f"(world_size={world_size})"
        )
    return owners


def cluster_ready_timeout_s() -> float:
    """How long an attacher waits for the owner's bootstrap to settle."""
    return float(os.getenv("FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC", "120"))


def attach_radix_client(shm_name: str,
                        expect_distributed: bool,
                        wait_timeout_s: Optional[float] = None
                        ) -> shmradix.RadixClient:
    """Attach a RadixClient to `shm_name`, waiting for cluster readiness.

    The owner creates the region BEFORE it calls `bootstrap()` and stamps the
    cluster manifest (world_size, rank, RDMA plane) into the header only once
    bootstrap settles. A client that attaches in the window between the two sees
    a standalone header: it reports `world_size=1`, `rank()=0` and skips the RDMA
    plane permanently, no matter what the header says later. So when the caller
    expects a cluster, waiting for the shm file to appear is not enough — attach
    is retried until the region reports itself distributed.

    Returns the client. If the region never became distributed in time the
    standalone client is returned anyway, leaving the degrade-or-fail decision to
    the caller; if nothing could be attached at all, raises TimeoutError.
    """
    if shmradix is None:
        raise ImportError("shmradix not installed")
    if wait_timeout_s is None:
        wait_timeout_s = cluster_ready_timeout_s()

    shm_path = f"/dev/shm{shm_name}"
    deadline = time.monotonic() + wait_timeout_s
    client = None
    while True:
        if os.path.exists(shm_path):
            try:
                client = shmradix.RadixClient(shm_name)
                if not expect_distributed or client.is_distributed():
                    return client
            except Exception as e:
                flexkv_logger.debug(
                    f"attach to {shm_name} failed (will retry): {e}")
        if time.monotonic() >= deadline:
            if client is None:
                raise TimeoutError(
                    f"Timed out attaching to shm radix region: {shm_name}"
                )
            return client
        time.sleep(0.01)


def attach_shm_radix_clients(cache_config: CacheConfig,
                             server_id: str,
                             rank: int = 0,
                             world_size: int = 1,
                             wait_timeout_s: Optional[float] = None):
    """Attach a RadixClient per enabled device type. Polls until the owner has
    created the shm files AND (for world_size > 1) finished bootstrap, since
    `rank()` on a client that attached mid-bootstrap reads back 0 on every
    node."""
    if shmradix is None:
        raise ImportError("shmradix not installed")
    if wait_timeout_s is None:
        wait_timeout_s = cluster_ready_timeout_s()

    clients: Dict[DeviceType, shmradix.RadixClient] = {}
    deadline = time.monotonic() + wait_timeout_s
    pending = list(enabled_devices(cache_config))
    while pending and time.monotonic() < deadline:
        next_pending = []
        for dt in pending:
            name = shm_name_for(dt, server_id, rank=rank, world_size=world_size)
            remaining = max(0.0, deadline - time.monotonic())
            client = attach_radix_client(name,
                                         expect_distributed=world_size > 1,
                                         wait_timeout_s=remaining)
            if client is None or (world_size > 1 and not client.is_distributed()):
                next_pending.append(dt)
                continue
            clients[dt] = client
        pending = next_pending
        if pending:
            time.sleep(0.01)
    if pending:
        names = [
            shm_name_for(dt, server_id, rank=rank, world_size=world_size)
            for dt in pending
        ]
        raise TimeoutError(
            f"Timed out attaching to shm radix regions: {names}"
        )
    return clients
