# SPDX-License-Identifier: Apache-2.0
"""
Bootstrap for radixshmem-backed CacheEngine.

Per-device-type RadixTree (CPU / SSD / REMOTE) lives in its own POSIX shm
region. The first DP process (instance 0, dp_client_id 0) creates the regions
via `shmradix.TreeServer`; all others attach via `shmradix.TreeClient`.

Naming convention:
    /flexkv_radix_{server_id}_{cpu|ssd|remote}

`server_id` defaults to a fixed token but is overridable so multiple FlexKV
instances on the same host don't collide.
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


def shm_name_for(device_type: DeviceType, server_id: str) -> str:
    """POSIX shm name for the radix region of one device type."""
    kind = _DEVICE_KIND_NAMES[device_type]
    return f"/flexkv_radix_{server_id}_{kind}"


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
    """Holder for the `TreeServer` instances created in the bootstrap process.

    The owner process must keep these alive for the lifetime of the FlexKV
    server (otherwise the shm regions are torn down). We attach them to a
    long-lived object (e.g. KVManager) so Python's GC doesn't reap them.
    """

    def __init__(self) -> None:
        self.servers: Dict[DeviceType, shmradix.TreeServer] = {}

    def add(self, device_type: DeviceType, server: shmradix.TreeServer) -> None:
        self.servers[device_type] = server

    def shutdown(self) -> None:
        # TreeServer destructor unlinks the shm region.
        self.servers.clear()


def create_shm_radix_regions(model_config: ModelConfig,
                             cache_config: CacheConfig,
                             server_id: str,
                             max_nodes_per_device: Optional[int] = None,
                             data_pool_ratio: int = 5,
                             evict_ratio: float = 0.05,
                             background_evict: bool = True) -> ShmRadixOwners:
    """Called by the bootstrap (instance 0, dp 0) process to create the
    shm regions. Returns an owner handle that callers must keep alive."""
    if shmradix is None:
        raise ImportError("shmradix not installed")

    owners = ShmRadixOwners()
    for dt in enabled_devices(cache_config):
        n_blocks = device_blocks_from_config(dt, cache_config)
        if n_blocks <= 0:
            continue
        cfg = shmradix.ShmConfig(
            max_nodes=max_nodes_per_device or max(n_blocks * 2, 200_000),
            max_blocks=n_blocks,
            data_pool_ratio=data_pool_ratio,
            evict_ratio=evict_ratio,
            background_evict=background_evict,
        )
        name = shm_name_for(dt, server_id)
        flexkv_logger.info(
            f"creating shm radix region {name} "
            f"(max_nodes={cfg.max_nodes}, max_blocks={cfg.max_blocks})"
        )
        owners.add(dt, shmradix.TreeServer(name, cfg))
    return owners


def attach_shm_radix_clients(cache_config: CacheConfig,
                             server_id: str,
                             wait_timeout_s: float = 60.0):
    """Attach a TreeClient per enabled device type. Polls until the owner
    has created the shm files (since multiple processes may race startup)."""
    if shmradix is None:
        raise ImportError("shmradix not installed")

    clients: Dict[DeviceType, shmradix.TreeClient] = {}
    deadline = time.monotonic() + wait_timeout_s
    pending = list(enabled_devices(cache_config))
    while pending and time.monotonic() < deadline:
        next_pending = []
        for dt in pending:
            name = shm_name_for(dt, server_id)
            shm_path = f"/dev/shm{name}"
            if not os.path.exists(shm_path):
                next_pending.append(dt)
                continue
            try:
                clients[dt] = shmradix.TreeClient(name)
            except Exception as e:
                flexkv_logger.debug(f"attach to {name} failed (will retry): {e}")
                next_pending.append(dt)
        pending = next_pending
        if pending:
            time.sleep(0.01)
    if pending:
        names = [shm_name_for(dt, server_id) for dt in pending]
        raise TimeoutError(
            f"Timed out attaching to shm radix regions: {names}"
        )
    return clients
