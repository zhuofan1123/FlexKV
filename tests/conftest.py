"""
Pytest configuration file for FlexKV tests.
This file contains shared fixtures and setup code for all tests.
"""
import contextlib
import os
import shutil

import pytest

# Import fixtures from common_utils so pytest can discover them
from common_utils import model_config, cache_config, test_config

# Geometry for `radix_shmem_env`. Small on purpose: the tests below assert on
# exact free-block counts, and a 64-block pool makes an accidental leak obvious.
RADIX_TOKENS_PER_BLOCK = 16
RADIX_NUM_BLOCKS = 64


class RadixShmemEnv:
    """A live `GlobalCacheEngine` on radixshmem regions, plus its geometry."""

    def __init__(self, engine, tokens_per_block: int, num_blocks: int):
        self.engine = engine
        self.cpu = engine.cpu_cache_engine
        self.ssd = engine.ssd_cache_engine
        self.tokens_per_block = tokens_per_block
        self.num_blocks = num_blocks

    def seq(self, token_ids):
        from flexkv.common.block import SequenceMeta
        import numpy as np
        return SequenceMeta(token_ids=np.asarray(token_ids).copy(),
                            tokens_per_block=self.tokens_per_block)

    def hit(self, tier, token_ids) -> int:
        """Local match length, dropping the query's refs before returning."""
        match = tier.match(self.seq(token_ids), with_peer=False)
        match.release()
        return match.num_local_blocks

    def slots(self, tier, token_ids, num_blocks: int):
        """The tier's own slot ids for the first `num_blocks` matched blocks."""
        import numpy as np
        match = tier.match(self.seq(token_ids), with_peer=False)
        slots = np.asarray(match.local_range(0, num_blocks), dtype=np.int64)
        match.release()
        return slots

    def free(self) -> tuple:
        return self.cpu.num_free_blocks, self.ssd.num_free_blocks


@pytest.fixture(scope="module")
def radix_shmem_env(request):
    """CPU+SSD radixshmem regions and a `GlobalCacheEngine` bound to them.

    Module-scoped because the regions are cheap to create but the tests mutate
    the trees: each module gets its own region names (derived from the module and
    the pid) so a parallel or repeated run never attaches to a live one. Skips
    when shmradix is missing/stale or when `flexkv.c_ext` cannot load.
    """
    try:
        import shmradix  # noqa: F401
    except ImportError as exc:
        # ImportError, not ModuleNotFoundError, is what a stale `_core.so` next to
        # a newer `__init__.py` raises.
        pytest.skip(f"shmradix unusable ({exc}); rebuild the extension")
    try:
        import torch
        from flexkv.cache.cache_engine import GlobalCacheEngine
    except Exception as exc:  # c_ext links libcudart
        pytest.skip(f"GlobalCacheEngine unavailable: {exc}")

    from flexkv.common.config import (CacheConfig, GLOBAL_CONFIG_FROM_ENV,
                                      ModelConfig)
    from flexkv.common.transfer import DeviceType
    from flexkv.server.shm_radix_bootstrap import (create_shm_radix_regions,
                                                   shm_name_for)

    shm_radix_id = f"{request.module.__name__.replace('_', '')}{os.getpid()}"
    ssd_dir = f"/tmp/flexkv_radix_env_{shm_radix_id}"

    saved = {name: getattr(GLOBAL_CONFIG_FROM_ENV, name)
             for name in ("radix_shmem", "shm_radix_id",
                          "radix_world_size")}
    GLOBAL_CONFIG_FROM_ENV.radix_shmem = True
    GLOBAL_CONFIG_FROM_ENV.shm_radix_id = shm_radix_id
    GLOBAL_CONFIG_FROM_ENV.radix_world_size = 1

    model_config = ModelConfig(num_layers=2, num_kv_heads=4, head_size=64,
                               dtype=torch.float16, use_mla=False,
                               tp_size=1, dp_size=1)
    shutil.rmtree(ssd_dir, ignore_errors=True)
    os.makedirs(ssd_dir, exist_ok=True)
    cache_config = CacheConfig(
        tokens_per_block=RADIX_TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=True, enable_remote=False,
        num_cpu_blocks=RADIX_NUM_BLOCKS, num_ssd_blocks=RADIX_NUM_BLOCKS,
        ssd_cache_dir=[ssd_dir],
    )

    tiers = (DeviceType.CPU, DeviceType.SSD)
    for device_type in tiers:
        with contextlib.suppress(FileNotFoundError):
            os.unlink("/dev/shm" + shm_name_for(device_type, shm_radix_id))
    # Hold the owner handle for the module's lifetime: dropping it unlinks the
    # regions out from under the engine.
    owners = create_shm_radix_regions(cache_config,
                                      shm_radix_id=shm_radix_id)

    engine = GlobalCacheEngine(cache_config, model_config)
    assert engine.use_radix_shmem, "fixture did not get the radixshmem backend"
    try:
        yield RadixShmemEnv(engine, RADIX_TOKENS_PER_BLOCK, RADIX_NUM_BLOCKS)
    finally:
        del owners
        for device_type in tiers:
            with contextlib.suppress(FileNotFoundError):
                os.unlink("/dev/shm" + shm_name_for(device_type, shm_radix_id))
        shutil.rmtree(ssd_dir, ignore_errors=True)
        for name, value in saved.items():
            setattr(GLOBAL_CONFIG_FROM_ENV, name, value)
