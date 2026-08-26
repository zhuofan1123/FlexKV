# SPDX-License-Identifier: Apache-2.0
"""
Bootstrap for radixshmem-backed CacheEngine.

Per-device-type RadixTree (CPU / SSD) lives in its own POSIX shm
region. The first DP process (instance 0, dp_client_id 0) creates the regions
via `shmradix.RadixServer`; every other process attaches by name with
`shmradix.RadixClient` (`attach_radix_client`, called from
`CacheEngineRadixShmem`).

Naming convention:
    /shmradix_{shm_radix_id}_{cpu|ssd|remote}

`shm_radix_id` (FLEXKV_SHM_RADIX_ID, default `flexkv`) tells apart several
FlexKV instances on one node. A distributed region appends this node's shmradix
identity (`node_name_for`: SHMRADIX_NODE_NAME, else `node<bind-ip>`), which FlexKV
resolves itself and passes to `RadixServerConfig.node_name` so `shm_name_for` is
the one name both the owner and every attacher use.

Cluster membership goes through etcd, which is shmradix's only bootstrap path in
an RDMA build. Each tier rendezvouses in its OWN namespace (`cluster_id_for`),
since a peer entry is keyed by node identity alone yet carries that tier's shm
registration. The identity has to be unique per node, so nodes co-located on one
host need a distinct bind IP (or a distinct `SHMRADIX_NODE_NAME`) AND a distinct
FLEXKV_SHM_RADIX_ID for their local region and TE channel names.
Cluster rank is an OUTPUT: etcd assigns dense ranks by sorted `/peers` key order,
and callers read it back off `ShmRadixOwners.cluster_rank` (`RadixServer.rank()`)
rather than configuring one.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import socket
import struct
import time
from typing import Dict, Generator, Optional, Tuple

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV, CacheConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType

try:
    import shmradix
except ImportError:  # pragma: no cover
    shmradix = None


_SHM_PREFIX = "/shmradix"

_SIOCGIFADDR = 0x8915

_DEVICE_KIND_NAMES = {
    DeviceType.CPU: "cpu",
    DeviceType.SSD: "ssd",
    DeviceType.REMOTE: "remote",
}


def _shm_base(device_type: DeviceType, shm_radix_id: str) -> str:
    """Tier name without the node identity — what a distributed owner passes as
    ``RadixServerConfig.name`` for ``bootstrap()`` to extend."""
    return f"{_SHM_PREFIX}_{shm_radix_id}_{_DEVICE_KIND_NAMES[device_type]}"


def _interface_ipv4(iface: str) -> str:
    """IPv4 bound to ``iface``, or "" — mirrors shmradix's ``interface_ipv4``.

    shmradix walks ``getifaddrs`` and takes the interface's first AF_INET address;
    SIOCGIFADDR reports the same primary address.
    """
    with contextlib.closing(socket.socket(socket.AF_INET,
                                          socket.SOCK_DGRAM)) as sock:
        try:
            res = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR,
                              struct.pack("256s", iface.encode()[:15]))
        except OSError:
            return ""
    return socket.inet_ntoa(res[20:24])


def resolve_bind_ip() -> str:
    """Bind IP shmradix resolves for this node, or "" if it resolves none.

    Mirrors ``resolve_bind_ip`` in shmradix's ``net_util.hpp``: the interface wins
    when both it and an address are configured, and an address is used verbatim.
    """
    env = GLOBAL_CONFIG_FROM_ENV
    if env.radix_rpc_interface:
        return _interface_ipv4(env.radix_rpc_interface)
    return env.radix_rpc_address


def node_name_for() -> str:
    """This node's shmradix identity: ``SHMRADIX_NODE_NAME``, else ``node<bind-ip>``.

    Same rule as ``RadixServer::bootstrap()``, but FlexKV resolves it itself and
    passes it as ``RadixServerConfig.node_name``: shmradix applies the env override
    only AFTER fixing the shm name, so left to shmradix the region would always be
    named after the bind IP no matter what the env says.
    """
    env_name = os.getenv("SHMRADIX_NODE_NAME", "")
    if env_name:
        return env_name
    bind_ip = resolve_bind_ip()
    if not bind_ip:
        raise RuntimeError(
            "cannot derive this node's radixshmem identity: distributed mode "
            "falls back to the bind IP, and none resolved from "
            "FLEXKV_RADIX_RPC_ADDRESS / FLEXKV_RADIX_RPC_INTERFACE"
        )
    return f"node{bind_ip}"


def cluster_id_for(device_type: DeviceType) -> str:
    """etcd namespace of one tier: ``SHMRADIX_CLUSTER_ID`` plus the tier name.

    Keys live under ``radix/<cluster_id>/``, and a peer entry (``peers/<node>``) is
    keyed by node identity alone while carrying THAT tier's shm base/rkey — one
    namespace per tier is what stops a node's tiers from overwriting each other.
    The key set is the same in every namespace, so etcd still hands a node the same
    dense rank in each, which is what ``create_shm_radix_regions`` demands. With the
    env unset the base is "default", shmradix's own ``RadixServerConfig`` default.
    """
    base = os.getenv("SHMRADIX_CLUSTER_ID") or "default"
    return f"{base}_{_DEVICE_KIND_NAMES[device_type]}"


@contextlib.contextmanager
def _pin_cluster_id(cluster_id: str) -> Generator[None, None, None]:
    """Hold ``SHMRADIX_CLUSTER_ID`` at ``cluster_id`` for one bootstrap.

    ``RadixServer::bootstrap`` applies that env ON TOP of the config it was given
    (indexer/server.cpp, "Deployment-level env overrides"), so passing
    ``RadixServerConfig.cluster_id`` alone is silently discarded wherever the env is
    set — the per-tier namespace has to be in the env for exactly that call, and
    restored afterwards so the next tier does not inherit it.
    """
    prev = os.environ.get("SHMRADIX_CLUSTER_ID")
    os.environ["SHMRADIX_CLUSTER_ID"] = cluster_id
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("SHMRADIX_CLUSTER_ID", None)
        else:
            os.environ["SHMRADIX_CLUSTER_ID"] = prev


def shm_name_for(device_type: DeviceType, shm_radix_id: str) -> str:
    """POSIX shm name of one tier's region — owner and attachers both use this.

    Standalone regions carry ``shm_radix_id`` verbatim; a distributed one appends
    this node's identity (``node_name_for``), which is what ``bootstrap()`` does
    with the ``node_name`` FlexKV hands it.
    """
    base = _shm_base(device_type, shm_radix_id)
    if GLOBAL_CONFIG_FROM_ENV.radix_world_size <= 1:
        return base
    return f"{base}_{node_name_for()}"


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
        # Cluster rank etcd assigned this node, read back after bootstrap; a
        # per-tier mismatch is rejected in create_shm_radix_regions.
        self.cluster_rank: int = 0

    def add(self, device_type: DeviceType, server: shmradix.RadixServer) -> None:
        self.servers[device_type] = server

    def shutdown(self) -> None:
        # RadixServer destructor releases the shm region.
        self.servers.clear()


def create_shm_radix_regions(cache_config: CacheConfig,
                             shm_radix_id: str,
                             *,
                             data_pool_ratio: int = 8,
                             evict_ratio: float = 0.05,
                             background_evict: bool = True) -> ShmRadixOwners:
    """Called by the bootstrap (instance 0, dp 0) process to create the
    shm regions. Returns an owner handle that callers must keep alive.

    Cluster settings are read from ``GLOBAL_CONFIG_FROM_ENV`` (FLEXKV_RADIX_*).
    ``radix_world_size > 1`` creates *distributed* regions: every node registers
    itself in etcd under its own identity (``SHMRADIX_NODE_NAME``, else derived
    from the bind IP), the leader waits for all ``world_size`` of them and assigns
    dense ranks, RDMA connection info is exchanged, and from then on a query that
    outruns the local tree continues on a peer's tree over RDMA. The gate is
    collective — each node blocks in ``bootstrap()`` until the whole cluster has
    arrived, once per tier, since each tier rendezvouses in its own namespace
    (``cluster_id_for``).

    Read the assigned rank back off the returned handle
    (``owners.cluster_rank``); that is the number the peer data path addresses."""
    if shmradix is None:
        raise ImportError("shmradix not installed")

    env = GLOBAL_CONFIG_FROM_ENV
    world_size = env.radix_world_size
    if world_size > 1:
        if not env.radix_registry:
            raise ValueError(
                "radixshmem distributed mode needs an etcd registry "
                "(FLEXKV_RADIX_REGISTRY, e.g. 'etcd://10.0.0.1:2379'): it is "
                "the only cluster bootstrap path shmradix has"
            )
        if not env.radix_rpc_address and not env.radix_rpc_interface:
            raise ValueError(
                "radixshmem distributed mode needs FLEXKV_RADIX_RPC_ADDRESS or "
                "FLEXKV_RADIX_RPC_INTERFACE — the bootstrap IP peers dial, which "
                "also derives this node's identity: give every node a concrete "
                "address of its own (co-located nodes: 127.0.0.1, 127.0.0.2, "
                "...), never 0.0.0.0"
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
            # Persisted into the region so any attacher can recover it via
            # RadixClient.block_size() instead of being told out-of-band.
            block_size=cache_config.tokens_per_block,
            data_pool_ratio=data_pool_ratio,
            evict_ratio=evict_ratio,
            background_evict=background_evict,
        )
        name = shm_name_for(dt, shm_radix_id)
        flexkv_logger.info(
            f"creating shm radix region {name} "
            f"(max_nodes={cfg.max_nodes}, max_blocks={cfg.max_blocks}, "
            f"world_size={world_size})"
        )
        if world_size > 1:
            server_cfg = shmradix.RadixServerConfig()
            # bootstrap() names the region `name + "_" + node_name`, so it gets the
            # bare tier name plus the identity shm_name_for used. No cluster_id.
            server_cfg.name = _shm_base(dt, shm_radix_id)
            server_cfg.node_name = node_name_for()
            server_cfg.shm = cfg
            # No rank here: etcd assigns it and bootstrap() overwrites it.
            server_cfg.world_size = world_size
            # The membership gate is max(num_shards, expected_min_nodes) and only
            # num_shards is reachable from Python; world_size is its default.
            server_cfg.num_shards = world_size
            server_cfg.registry = env.radix_registry
            server_cfg.cluster_id = cluster_id_for(dt)
            server_cfg.rpc_address = env.radix_rpc_address
            server_cfg.rpc_interface = env.radix_rpc_interface
            server_cfg.remote_op_transport = env.radix_remote_op_transport
            server_cfg.rdma_dev = env.radix_rdma_dev
            server_cfg.gid_idx = env.radix_gid_idx
            server_cfg.bootstrap_timeout_sec = env.radix_bootstrap_timeout_sec
            # Unlike the single-node overload, this ctor does NOT create the shm
            # region — bootstrap() does, and it is not idempotent.
            server = shmradix.RadixServer(server_cfg)
            with _pin_cluster_id(server_cfg.cluster_id):
                ok = server.bootstrap()
            if not ok:
                raise RuntimeError(
                    f"radixshmem cluster bootstrap failed for {name} "
                    f"(world_size={world_size}, registry={env.radix_registry}, "
                    f"cluster_id={server_cfg.cluster_id}); check that etcd is "
                    f"reachable, that all {world_size} nodes joined within "
                    f"{env.radix_bootstrap_timeout_sec}s under a DISTINCT etcd "
                    f"identity (SHMRADIX_NODE_NAME, else derived from the bind "
                    f"IP), and that shmradix was built with RDMA + etcd support"
                )
            if not server.is_distributed():
                raise RuntimeError(
                    f"radixshmem region {name} bootstrapped but reports "
                    f"world_size={server.world_size()}; the installed shmradix "
                    f"extension was built without RDMA (FLEXKV_NO_RDMA)"
                )
            actual_name = server.shm_name()
            if actual_name != name:
                raise RuntimeError(
                    f"radixshmem named the region {actual_name!r} but attachers "
                    f"ask for {name!r}; shm_name_for() no longer mirrors "
                    f"RadixServer.bootstrap()"
                )
        else:
            server = shmradix.RadixServer(name, cfg)
        cluster_ranks[dt] = int(server.rank())
        owners.add(dt, server)

    # Tiers rendezvous independently, so their ranks must agree — one node id has
    # to stand for all of them or the peer data path addresses the wrong node.
    distinct = set(cluster_ranks.values())
    if len(distinct) > 1:
        raise RuntimeError(
            f"radixshmem assigned different cluster ranks per tier: "
            f"{ {_DEVICE_KIND_NAMES[k]: v for k, v in cluster_ranks.items()} }; "
            f"FlexKV needs one node id for all tiers"
        )
    owners.cluster_rank = distinct.pop() if distinct else 0
    if world_size > 1:
        namespaces = ", ".join(cluster_id_for(dt) for dt in cluster_ranks)
        flexkv_logger.info(
            f"radixshmem cluster bootstrap done: {shm_radix_id} -> cluster rank "
            f"{owners.cluster_rank} (world_size={world_size}, "
            f"etcd namespaces: {namespaces})"
        )
    return owners


def attach_radix_client(shm_name: str,
                        expect_distributed: bool,
                        wait_timeout_s: Optional[float] = None
                        ) -> shmradix.RadixClient:
    """Attach a RadixClient to `shm_name`, waiting for cluster readiness.

    `shm_name` is what `shm_name_for` spells out — the full name including the node
    identity, so there is nothing left to resolve here.

    The owner creates the region BEFORE it calls `bootstrap()` and stamps the
    cluster manifest (world_size, rank, RDMA plane) into the header only once
    bootstrap settles. A client that attaches in the window between the two sees
    a standalone header: it reports `world_size=1`, `rank()=0` and skips the RDMA
    plane permanently, no matter what the header says later. So when the caller
    expects a cluster, waiting for the region to appear is not enough — attach is
    retried until it reports itself distributed.

    Returns the client. If the region never became distributed in time the
    standalone client is returned anyway, leaving the degrade-or-fail decision to
    the caller; if nothing could be attached at all, raises TimeoutError.
    """
    if shmradix is None:
        raise ImportError("shmradix not installed")
    if wait_timeout_s is None:
        wait_timeout_s = float(GLOBAL_CONFIG_FROM_ENV.radix_bootstrap_timeout_sec)

    deadline = time.monotonic() + wait_timeout_s
    client = None
    while True:
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
