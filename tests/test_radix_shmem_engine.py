"""Smoke tests for `CacheEngineRadixShmem`. Skipped if `shmradix` not installed.

Uses a duck-typed fake `SequenceMeta` so the test does not pull in
`flexkv.common.block → flexkv.common.hash_utils → flexkv.c_ext` (which
requires CUDA at import time). On a machine with CUDA + the FlexKV C++
extension built, you can swap in the real `SequenceMeta` — the engine only
calls `seq.gen_hashes()` and reads `seq.block_hashes`.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

shmradix = pytest.importorskip("shmradix")


def _load_module_direct(name: str, path: str):
    """Load a module by file path, bypassing parent package __init__.

    `flexkv/cache/__init__.py` imports `flexkv.c_ext`, which links libcudart.
    Side-load `radix_shmem_engine` directly so the test runs on CPU-only hosts.
    """
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` looks up the module in sys.modules during class
    # construction; register before exec_module.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_FLEXKV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_engine_mod = _load_module_direct(
    "_radix_shmem_engine_test",
    os.path.join(_FLEXKV_ROOT, "flexkv", "cache", "radix_shmem_engine.py"),
)
CacheEngineRadixShmem = _engine_mod.CacheEngineRadixShmem

# `flexkv.common.transfer` is pure-Python (no c_ext). DeviceType is fine.
from flexkv.common.transfer import DeviceType


@dataclass
class FakeSeq:
    """Duck-typed SequenceMeta for the engine API: only `block_hashes` and
    `gen_hashes()` are read by `CacheEngineRadixShmem`."""
    block_hashes: np.ndarray
    tokens_per_block: int = 4

    @property
    def num_blocks(self) -> int:
        return len(self.block_hashes)

    def gen_hashes(self) -> None:
        # already populated
        pass


def _make_engine(name: str, blocks: int = 10000, tokens_per_block: int = 4):
    cfg = shmradix.ShmConfig(max_nodes=blocks * 4, max_blocks=blocks)
    server = shmradix.TreeServer(name, cfg)
    engine = CacheEngineRadixShmem(
        device_type=DeviceType.CPU,
        num_total_blocks=blocks,
        tokens_per_block=tokens_per_block,
        shm_name=name,
    )
    return engine, server


def _hashes(seed: int, num: int) -> np.ndarray:
    """Deterministic, distinct int64 hashes."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=1, high=2**62, size=num, dtype=np.int64)


def test_take_insert_match_recycle():
    engine, _server = _make_engine("/cers_basic")

    seq = FakeSeq(block_hashes=_hashes(seed=1, num=4))
    # Initial match: nothing.
    r = engine.match(seq)
    assert r.num_matched_blocks == 0

    # take 4 slots and insert.
    slots = engine.take(num_required_blocks=4)
    assert len(slots) == 4
    node = engine.insert(seq, slots, num_insert_blocks=4, is_ready=True)
    assert node is not None
    assert node.size() == 4

    # Match should now hit all 4 blocks.
    r2 = engine.match(seq)
    assert r2.num_matched_blocks == 4
    assert r2.num_ready_matched_blocks == 4
    np.testing.assert_array_equal(np.sort(r2.physical_blocks), np.sort(slots))

    # last_node and last_ready_node populated.
    assert r2.last_node is not None
    assert r2.last_ready_node is not None
    assert r2.last_ready_node.node_id == r2.last_node.node_id

    # Recycle a fresh allocation; tree-attached slots are not affected.
    free_slots = engine.take(num_required_blocks=2, strict=False)
    engine.recycle(free_slots)


def test_match_unready_prefix():
    engine, _server = _make_engine("/cers_unready")

    seq = FakeSeq(block_hashes=_hashes(seed=2, num=6))
    slots = engine.take(num_required_blocks=6)
    engine.insert(seq, slots, is_ready=False)

    r = engine.match(seq)
    assert r.num_matched_blocks == 6
    # Whole node is unready, so the ready prefix is 0.
    assert r.num_ready_matched_blocks == 0
    # last_ready_node is None (no ready prefix).
    assert r.last_ready_node is None
    # last_node still set to the unready node.
    assert r.last_node is not None


def test_lock_prevents_eviction():
    # Pool must be big enough for the buddy allocator to initialize
    # (max_blocks * data_pool_ratio * 12 >= 49152). 2000 is comfortable.
    engine, _server = _make_engine("/cers_lock", blocks=2000)

    seq = FakeSeq(block_hashes=_hashes(seed=3, num=10))
    slots = engine.take(num_required_blocks=10)
    node = engine.insert(seq, slots, is_ready=True)
    assert node is not None
    engine.lock_node(node)

    # Drain the rest of the pool. Locked node must not be evicted.
    free_now = engine.num_free_blocks
    drained = engine.take(num_required_blocks=free_now, strict=False)
    assert len(drained) == free_now

    # Original sequence still matchable.
    r = engine.match(seq)
    assert r.num_matched_blocks == 10
    assert r.num_ready_matched_blocks == 10

    engine.unlock(node)
    engine.recycle(drained)


def test_eviction_reclaims_unlocked():
    engine, _server = _make_engine("/cers_evict", blocks=2000)

    seq = FakeSeq(block_hashes=_hashes(seed=4, num=1500))
    s1 = engine.take(num_required_blocks=1500)
    engine.insert(seq, s1, is_ready=True)
    # Don't lock. Allocate enough new blocks that eviction is forced (need >
    # current free 500).
    s2 = engine.take(num_required_blocks=1500, strict=False)
    # On the same shm region, eviction reclaimed the unlocked LRU sequence
    # so we got more than the initial free count.
    assert len(s2) > 500


if __name__ == "__main__":
    test_take_insert_match_recycle()
    print("PASS test_take_insert_match_recycle")
    test_match_unready_prefix()
    print("PASS test_match_unready_prefix")
    test_lock_prevents_eviction()
    print("PASS test_lock_prevents_eviction")
    test_eviction_reclaims_unlocked()
    print("PASS test_eviction_reclaims_unlocked")
