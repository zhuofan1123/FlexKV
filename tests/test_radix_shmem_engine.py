"""Tests for the radixshmem cache path, from the storage engine up to the
transfer graph that a cross-node hit plans. Skipped if `shmradix` is missing.

Three layers, one file:

  Part 1 — `CacheEngineRadixShmem` semantics on a single shm region:
    take / insert / match / recycle, insert-after-transfer publication,
    lock vs. eviction, and the fact that a non-clustered region never reports a
    remote hit.
  Part 2 — `GlobalCacheEngine.get()` planning on the radixshmem backend
    (`_get_impl_radixshmem`), driven by synthetic `ShmRadixMatch`es (no shm
    region, no RDMA): which fragment each transfer type covers, the
    `src_block_node_ids` routing contract, and the staged-insert lifecycle.
  Part 3 — a real 2-rank distributed radix cluster over RDMA in two spawned
    processes: rank 0 publishes a prefix, rank 1 must find it on rank 0.

Parts 2 and 3 both pin the same invariant: a match is a SPLICE — a local head
followed by a tail on ONE peer, with `num_local_blocks` as the boundary — so the
local head is reused as-is and only the tail crosses the wire.

Import-time notes:
  * Part 1 uses a duck-typed fake `SequenceMeta` so the test does not pull in
    `flexkv.common.block → flexkv.common.hash_utils → flexkv.c_ext` (which
    requires CUDA at import time). The engine only calls `seq.gen_hashes()` and
    reads `seq.block_hashes`, so the same fake serves Part 3.
  * Part 2 needs the real `GlobalCacheEngine` (and therefore `c_ext`), so it
    imports it lazily and skips instead of breaking collection for Parts 1/3.
  * Part 3 needs a working RDMA device, a shmradix built WITH RDMA support, and
    a reachable etcd (the only cluster bootstrap path there is), so it is gated
    behind FLEXKV_RUN_RADIX_PEER_TEST=1 and additionally skips when the host
    exposes no ACTIVE RDMA port or FLEXKV_TEST_RADIX_REGISTRY is unset.
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import itertools
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass

import numpy as np
import pytest

try:
    import shmradix
except ImportError as exc:
    # Not importorskip: a shmradix built before the current API (stale `_core.so`
    # next to a newer `__init__.py`) raises ImportError rather than
    # ModuleNotFoundError, and pytest >= 8.2 only skips on the latter — which
    # would abort collection for the whole suite instead of skipping this file.
    pytest.skip(f"shmradix unusable ({exc}); rebuild the extension",
                allow_module_level=True)


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
# The planner only duck-types the match, so the side-loaded class is as good as
# the one `flexkv.cache.cache_engine` imports — and it needs no c_ext.
ShmRadixMatch = _engine_mod.ShmRadixMatch

# `flexkv.common.transfer` is pure-Python (no c_ext).
from flexkv.common.transfer import DeviceType, TransferType


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


# =============================================================================
# Part 1 — local engine semantics on a single (non-clustered) shm region
# =============================================================================


def _make_engine(name: str, blocks: int = 10000, tokens_per_block: int = 4):
    # Fixed region names are reused across runs; drop any leftover so the
    # RadixServer creates a fresh region instead of colliding with stale state.
    with contextlib.suppress(FileNotFoundError):
        os.remove(f"/dev/shm{name}")
    cfg = shmradix.ShmConfig(max_nodes=blocks * 4, max_blocks=blocks)
    server = shmradix.RadixServer(name, cfg)
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
    engine.insert(seq, slots, num_insert_blocks=4)

    # Match should now hit all 4 blocks.
    r2 = engine.match(seq)
    assert r2.num_matched_blocks == 4
    # Non-clustered region: the whole match is the local head.
    assert r2.num_local_blocks == 4
    assert not r2.has_peer_tail
    np.testing.assert_array_equal(np.sort(r2.local_slots), np.sort(slots))
    r2.release()

    # Recycle a fresh allocation; tree-attached slots are not affected.
    free_slots = engine.take(num_required_blocks=2, strict=False)
    engine.recycle(free_slots)


def test_insert_publishes_immediately():
    """There is no ready bit: being in the tree IS being servable.

    Insert runs after the transfer on this backend, so a matched block is
    complete by construction — there is no flag to withhold a span with, and a
    single insert() is the whole publication.
    """
    engine, _server = _make_engine("/cers_unready")

    seq = FakeSeq(block_hashes=_hashes(seed=2, num=6))
    slots = engine.take(num_required_blocks=6)
    engine.insert(seq, slots, num_insert_blocks=6)

    r = engine.match(seq)
    assert r.num_matched_blocks == 6
    assert r.num_local_blocks == 6
    r.release()


def test_recycle_returns_staged_slots():
    """Slots whose transfer never landed are only reachable through recycle().

    They were never attached to the tree, so no query finds them and eviction
    cannot reclaim them — without recycle() they are lost for the life of the
    region.
    """
    engine, _server = _make_engine("/cers_recycle")

    before = engine.num_free_blocks
    slots = engine.take(num_required_blocks=5)
    assert engine.num_free_blocks == before - 5
    engine.recycle(slots)
    assert engine.num_free_blocks == before

    # And nothing was published on the way through.
    seq = FakeSeq(block_hashes=_hashes(seed=22, num=5))
    r = engine.match(seq)
    assert r.num_matched_blocks == 0
    r.release()


def test_eviction_reclaims_inserted():
    """A published span is immediately LRU-evictable.

    insert() runs after the transfer, so the span it attaches has no reader and
    takes no ref — nothing has to be released to make it reclaimable.
    """
    # Pool must be big enough for the buddy allocator to initialize
    # (max_blocks * data_pool_ratio * 12 >= 49152). 2000 is comfortable.
    engine, _server = _make_engine("/cers_evict", blocks=2000)

    seq = FakeSeq(block_hashes=_hashes(seed=4, num=1500))
    s1 = engine.take(num_required_blocks=1500)
    engine.insert(seq, s1, num_insert_blocks=1500)
    # insert() reports nothing, so check the span landed rather than let the
    # eviction assert below pass on an empty tree: an insert that attached
    # nothing would auto-recycle all 1500 slots, and the take() would then be
    # satisfied out of a free pool without evicting anything at all.
    published = engine.match(seq)
    assert published.num_matched_blocks == 1500
    published.release()
    assert engine.num_free_blocks == 500
    # Allocate enough new blocks that eviction is forced (need > current free 500).
    s2 = engine.take(num_required_blocks=1500, strict=False)
    # On the same shm region, eviction reclaimed the now-unlocked LRU sequence
    # so we got more than the initial free count.
    assert len(s2) > 500


def test_match_is_local_without_peer():
    """A single-node region never reports a remote hit, and peer mode is off."""
    engine, _server = _make_engine("/cers_local_only")
    seq = FakeSeq(block_hashes=_hashes(seed=12, num=3))
    slots = engine.take(num_required_blocks=3)
    engine.insert(seq, slots, num_insert_blocks=3)

    assert engine.peer_enabled is False
    result = engine.match(seq)
    # No peer tail => nobody to address, and the whole match is the local head.
    assert not result.has_peer_tail
    assert result.peer_id == _engine_mod.NO_PEER
    assert len(result.peer_slots) == 0
    assert result.num_matched_blocks == 3
    assert result.num_local_blocks == 3
    assert result.finalize is not None
    result.release()
    # release() is what drops the query's refs, and it is idempotent.
    assert result.finalize is None
    result.release()


def test_local_range_intersects_the_window():
    """`local_range` bounds the range on BOTH sides, by slicing alone.

    This is the accessor `_put_impl_radixshmem` leans on instead of clamping the
    match end itself, and `_shm_get_spans` to cut the local CPU/SSD head down to
    the GET window, so the contract both need is that a head stopping short of the
    window contributes nothing and one running past the window end is trimmed to
    it. The overrun case is the one a PUT cannot reach today (the window ends at
    the sequence end, which no match can exceed), so nothing else would notice if
    it stopped clamping.
    """
    match = ShmRadixMatch(
        num_local_blocks=4,
        num_peer_blocks=0,
        local_slots=np.arange(40, 44, dtype=np.int64),
        peer_id=_engine_mod.NO_PEER,
    )
    # Wholly inside the head.
    assert match.local_range(1, 3).tolist() == [41, 42]
    # Head runs PAST the window end -> trimmed to the window.
    assert match.local_range(0, 2).tolist() == [40, 41]
    # Window runs past the head -> trimmed to the head, no error.
    assert match.local_range(2, 99).tolist() == [42, 43]
    # Head stops short of the window start -> nothing of it is ours.
    assert match.local_range(4, 9).tolist() == []
    assert match.local_range(7, 9).tolist() == []
    # Empty and inverted windows name no block; full window is the whole head.
    assert match.local_range(2, 2).tolist() == []
    assert match.local_range(3, 1).tolist() == []
    assert match.local_range(0, 4).tolist() == [40, 41, 42, 43]


# =============================================================================
# Part 2 — GET planning on the radixshmem backend (`_get_impl_radixshmem`)
#
# A match here is a SPLICE: blocks `[0, num_local_blocks)` are held locally and
# `[num_local_blocks, num_matched_blocks)` by the ONE peer named by `peer_id`.
# These tests drive `GlobalCacheEngine.get()` with synthetic match results (no
# RDMA, no shm region) and assert:
#
#   * the local CPU head goes STRAIGHT to GPU — it is already host-resident, so
#     it needs no staging copy,
#   * only the peer tail crosses the wire (PEERH2H / PEERSSD2H), into freshly
#     taken local slots that H2D then reads,
#   * `src_block_node_ids` is sliced to exactly the blocks its own op moves,
#   * the staged span is published to the local tree once the graph completes,
#     and handed back to the mempool if the graph is cancelled instead.
# =============================================================================

TOKENS_PER_BLOCK = 16
PEER_NODE_ID = 77


def _global_cache_engine(enable_ssd: bool = False, enable_gds: bool = False):
    """Build a real `GlobalCacheEngine`.

    Imported here rather than at module scope: `flexkv.cache.__init__` pulls in
    `flexkv.c_ext` (libcudart), which Parts 1 and 3 deliberately avoid. A host
    without CUDA / a built extension skips these tests instead of failing to
    collect the whole file.
    """
    try:
        import torch

        from flexkv.cache.cache_engine import GlobalCacheEngine
        from flexkv.common.config import CacheConfig, ModelConfig
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"GlobalCacheEngine unavailable (needs CUDA + flexkv.c_ext): {exc}")

    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=64,
        dtype=torch.float16, use_mla=False, tp_size=1, dp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True,
        enable_ssd=enable_ssd,
        enable_gds=enable_gds,
        enable_remote=False,
        num_cpu_blocks=256,
        num_ssd_blocks=256,
        ssd_cache_dir=["./ssd_cache_peer_get"],
    )
    return GlobalCacheEngine(cache_config, model_config)


def _spliced_match(local_slots: np.ndarray,
                   peer_slots: np.ndarray) -> ShmRadixMatch:
    """One match with a local head and a tail on a single peer.

    Each side keeps its own slot ids — they resolve against different buffers —
    and `peer_id` names the owner of the tail.
    """
    local_slots = np.asarray(local_slots, dtype=np.int64)
    peer_slots = np.asarray(peer_slots, dtype=np.int64)
    return ShmRadixMatch(
        num_local_blocks=len(local_slots),
        num_peer_blocks=len(peer_slots),
        local_slots=local_slots,
        peer_slots=peer_slots,
        peer_id=PEER_NODE_ID if len(peer_slots) else _engine_mod.NO_PEER,
    )


def _local_match(slots: np.ndarray) -> ShmRadixMatch:
    """A match the local tree serves in full (what a non-clustered region gives)."""
    return _spliced_match(slots, np.array([], dtype=np.int64))


def _peer_match(peer_slots: np.ndarray) -> ShmRadixMatch:
    """A match with no local head at all: the peer owns every block."""
    return _spliced_match(np.array([], dtype=np.int64), peer_slots)


def _force_radixshmem(engine,
                      cpu_result: ShmRadixMatch,
                      ssd_result: ShmRadixMatch | None = None) -> None:
    """Put a real `GlobalCacheEngine` on the radixshmem planners, tree side stubbed.

    The tiers keep their real mempools (so `take` returns honest slot ids) but the
    tree side is faked: the synthetic matches name no real prefix, and the tiers
    here are `CacheEngineAccel`s, whose `insert` signature is a different one.
    Records what a planner published (`engine.inserted_pools`) and what it handed
    back (`engine.aborted_slots`) so completion and cancel paths can be asserted
    without a shm region.

    Only `_match_radixshmem` is stubbed — `match_local_accel` stays untouched,
    since nothing on these paths is supposed to reach it.
    """
    if ssd_result is None:
        ssd_result = ShmRadixMatch()
    engine.use_radix_shmem = True                   # type: ignore[attr-defined]
    engine._match_radixshmem = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (cpu_result, ssd_result)
    )
    inserted = []
    aborted = []
    for tier in (engine.cpu_cache_engine, engine.ssd_cache_engine):
        if tier is None:
            continue
        # No default for num_insert_blocks, mirroring the real signature: a
        # production caller that stopped passing it should fail here, loudly.
        def _insert(sequence_meta, physical_block_ids, num_insert_blocks,
                    _sink=inserted):
            _sink.append((num_insert_blocks, np.asarray(physical_block_ids)))

        # Recording wrapper, not a replacement: the tier's own recycle still runs,
        # so a planner that hands slots back really does free them.
        def _recycle(physical_block_ids, _orig=tier.recycle, _sink=aborted):
            _sink.append(np.asarray(physical_block_ids))
            _orig(np.asarray(physical_block_ids))

        tier.insert = _insert                       # type: ignore[method-assign]
        tier.recycle = _recycle                     # type: ignore[method-assign]
    engine.inserted_pools = inserted                # type: ignore[attr-defined]
    engine.aborted_slots = aborted                  # type: ignore[attr-defined]


def _fake_request(num_blocks: int):
    """(token_ids, token_mask, slot_mapping) for a fully-masked `num_blocks` window."""
    num_tokens = num_blocks * TOKENS_PER_BLOCK
    token_ids = np.arange(num_tokens, dtype=np.int64)
    token_mask = np.ones(num_tokens, dtype=np.bool_)
    # GPU blocks 1000.. so they can't be confused with CPU/SSD slot ids.
    slot_mapping = (
        np.repeat(np.arange(1000, 1000 + num_blocks), TOKENS_PER_BLOCK)
        * TOKENS_PER_BLOCK
        + np.tile(np.arange(TOKENS_PER_BLOCK), num_blocks)
    ).astype(np.int64)
    return token_ids, token_mask, slot_mapping


def _run_get(engine,
             num_blocks: int,
             cpu_result: ShmRadixMatch,
             ssd_result: ShmRadixMatch | None = None):
    """Call get() through `_get_impl_radixshmem` with forced match results."""
    _force_radixshmem(engine, cpu_result, ssd_result)
    token_ids, token_mask, slot_mapping = _fake_request(num_blocks)
    graph, return_mask, callback, _op_cbs, _end = engine.get(
        request_id=1,
        token_ids=token_ids,
        token_mask=token_mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
    )
    ops = {}
    for op in graph._op_map.values():
        ops.setdefault(op.transfer_type, []).append(op)
    engine.get_callback = callback                  # type: ignore[attr-defined]
    return graph, ops, return_mask


def _graph_deps(graph, op_id):
    return set(graph._op_map[op_id].predecessors)


def _abort_hooks(graph) -> list:
    """The cancel hooks that hand staged slots back to the mempool.

    PUT arms one `StagedRadixInsert.abort` per staged insert; GET publishes
    nothing, so it arms its own `_return_staging` for the one staging allocation.
    A graph also carries a `ShmRadixMatch.release` hook per tier (dropping the
    query's refs on a graph that never runs), which says nothing about staging.
    """
    return [hook for hook in graph.on_cancel
            if getattr(hook, "__name__", "") in ("abort", "_return_staging")]


def _assert_peer_routing(op, peer_node_id: int = PEER_NODE_ID) -> None:
    """The routing contract for a peer op's ``src_block_node_ids``.

    The worker zips the ids positionally against ``src_block_ids``, so the array
    must name exactly the blocks THIS op moves. One id too few and the tail blocks
    are dropped; one too many and the ids slide off the slots they belong to.
    """
    ids = np.asarray(op.src_block_node_ids)
    assert len(ids) == op.src_block_ids.size
    assert set(ids.tolist()) == {peer_node_id}


def test_peer_cpu_tail_is_staged_then_h2d():
    """A peer CPU match with no local head: one PEERH2H into staging, then H2D."""
    engine = _global_cache_engine()
    peer_slots = np.arange(50, 54, dtype=np.int64)
    _graph, ops, return_mask = _run_get(engine, 4, _peer_match(peer_slots))

    assert TransferType.PEERH2H in ops
    peer_op = ops[TransferType.PEERH2H][0]
    # Source blocks are the PEER's slot ids, verbatim from the match.
    np.testing.assert_array_equal(peer_op.src_block_ids, peer_slots)
    _assert_peer_routing(peer_op)
    # Destinations are freshly allocated LOCAL cpu blocks, not the peer's.
    assert not set(peer_op.dst_block_ids.tolist()) & set(peer_slots.tolist())

    h2d = ops[TransferType.H2D][0]
    # H2D reads the staged copies, never the peer's slot ids.
    np.testing.assert_array_equal(h2d.src_block_ids, peer_op.dst_block_ids)
    np.testing.assert_array_equal(h2d.dst_block_ids, np.arange(1000, 1004))
    assert peer_op.op_id in _graph_deps(_graph, h2d.op_id)
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK


def test_local_cpu_head_goes_straight_to_gpu():
    """The split point decides what gets staged: only the peer tail does.

    The local head is already in host memory, so copying it into fresh slots
    would waste both a transfer and a CPU block.
    """
    engine = _global_cache_engine()
    local_slots = np.arange(30, 32, dtype=np.int64)
    peer_slots = np.arange(50, 52, dtype=np.int64)
    _graph, ops, return_mask = _run_get(
        engine, 4, _spliced_match(local_slots, peer_slots))

    peer_op = ops[TransferType.PEERH2H][0]
    # Blocks 2-3 only — the head is not re-fetched.
    np.testing.assert_array_equal(peer_op.src_block_ids, peer_slots)
    _assert_peer_routing(peer_op)
    assert len(peer_op.dst_block_ids) == 2

    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids[:2], local_slots)
    np.testing.assert_array_equal(h2d.src_block_ids[2:], peer_op.dst_block_ids)
    np.testing.assert_array_equal(h2d.dst_block_ids, np.arange(1000, 1004))
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK


def test_staged_slots_go_back_on_completion():
    """A GET does not promote: its staging is scratch, never a tree entry.

    Storing the sequence locally is the PUT path's job, so the completion
    callback's only job with the slots is to give them back — and it is the only
    way back, since a slot outside the tree is invisible to eviction.
    """
    engine = _global_cache_engine()
    free_before = engine.cpu_cache_engine.mempool.num_free_blocks
    local_slots = np.arange(30, 32, dtype=np.int64)
    _graph, ops, _mask = _run_get(
        engine, 4, _spliced_match(local_slots, np.arange(50, 52, dtype=np.int64)))
    staged = ops[TransferType.PEERH2H][0].dst_block_ids
    # Held while the graph runs — the H2D reads out of them.
    assert (engine.cpu_cache_engine.mempool.num_free_blocks
            == free_before - len(staged))

    engine.get_callback()                           # type: ignore[attr-defined]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]
    aborted = engine.aborted_slots                  # type: ignore[attr-defined]
    assert len(aborted) == 1
    np.testing.assert_array_equal(aborted[0], staged)
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before


def test_cancelled_graph_returns_the_staged_slots():
    """A graph that never runs must hand its staging back to the mempool.

    The slots are not in the tree, so nothing else can find them and eviction
    cannot reclaim them: the cancel hook is the only way back.
    """
    engine = _global_cache_engine()
    free_before = engine.cpu_cache_engine.mempool.num_free_blocks
    graph, ops, _mask = _run_get(
        engine, 4, _peer_match(np.arange(50, 54, dtype=np.int64)))
    staged = ops[TransferType.PEERH2H][0].dst_block_ids
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - len(staged)

    assert len(_abort_hooks(graph)) == 1
    for cleanup in graph.on_cancel:
        cleanup()
    aborted = engine.aborted_slots                  # type: ignore[attr-defined]
    assert len(aborted) == 1
    np.testing.assert_array_equal(aborted[0], staged)
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before

    # Completion and cancel arm the same hook, so a late completion must be a
    # no-op rather than a second free of slots the pool already took back.
    engine.get_callback()                           # type: ignore[attr-defined]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]
    assert len(engine.aborted_slots) == 1           # type: ignore[attr-defined]
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before


def test_peer_ssd_tail_uses_peerssd2h():
    """A peer SSD tail reads over the wire instead of from local disk."""
    engine = _global_cache_engine(enable_ssd=True)
    ssd_peer_slots = np.arange(80, 84, dtype=np.int64)
    _graph, ops, return_mask = _run_get(
        engine, 4,
        _local_match(np.array([], dtype=np.int64)),
        _peer_match(ssd_peer_slots),
    )

    assert TransferType.DISK2H not in ops     # nothing came from local disk
    peer_ssd = ops[TransferType.PEERSSD2H][0]
    np.testing.assert_array_equal(peer_ssd.src_block_ids, ssd_peer_slots)
    _assert_peer_routing(peer_ssd)

    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids, peer_ssd.dst_block_ids)
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK


def test_peer_ssd_extends_a_local_cpu_hit():
    """CPU (local) covers 0-1, peer SSD reaches 3 → only 2-3 cross the wire."""
    engine = _global_cache_engine(enable_ssd=True)
    cpu_slots = np.arange(20, 22, dtype=np.int64)
    ssd_peer_slots = np.arange(80, 84, dtype=np.int64)
    _graph, ops, return_mask = _run_get(
        engine, 4, _local_match(cpu_slots), _peer_match(ssd_peer_slots))

    assert TransferType.PEERH2H not in ops   # the CPU hit was purely local
    peer_ssd = ops[TransferType.PEERSSD2H][0]
    # Only the 2 blocks beyond the CPU prefix are fetched.
    np.testing.assert_array_equal(peer_ssd.src_block_ids, ssd_peer_slots[2:])
    _assert_peer_routing(peer_ssd)

    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids[:2], cpu_slots)
    np.testing.assert_array_equal(h2d.src_block_ids[2:], peer_ssd.dst_block_ids)
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK


def test_local_ssd_and_peer_ssd_tail_split():
    """The SSD tail splits too: local disk first, then the peer's disk."""
    engine = _global_cache_engine(enable_ssd=True)
    cpu_slots = np.arange(20, 21, dtype=np.int64)          # block 0
    ssd_local = np.arange(60, 63, dtype=np.int64)          # blocks 0-2
    ssd_peer = np.arange(90, 92, dtype=np.int64)           # blocks 3-4
    _graph, ops, _mask = _run_get(
        engine, 5, _local_match(cpu_slots),
        _spliced_match(ssd_local, ssd_peer))

    # Local disk covers what CPU does not, up to the split point.
    disk2h = ops[TransferType.DISK2H][0]
    np.testing.assert_array_equal(disk2h.src_block_ids, ssd_local[1:])
    peer_ssd = ops[TransferType.PEERSSD2H][0]
    np.testing.assert_array_equal(peer_ssd.src_block_ids, ssd_peer)
    _assert_peer_routing(peer_ssd)

    # One contiguous staging span feeds the H2D behind the local CPU head.
    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids[:1], cpu_slots)
    np.testing.assert_array_equal(h2d.src_block_ids[1:3], disk2h.dst_block_ids)
    np.testing.assert_array_equal(h2d.src_block_ids[3:], peer_ssd.dst_block_ids)


def test_local_ssd_tail_stages_through_cpu_under_enable_gds():
    """`enable_gds` is ignored here: the SSD tail always goes via CPU.

    This planner has no DISK2D path, so a config with GDS enabled must still plan
    DISK2H into staging plus one H2D out of it.
    """
    engine = _global_cache_engine(enable_ssd=True, enable_gds=True)
    cpu_slots = np.arange(20, 22, dtype=np.int64)          # blocks 0-1
    ssd_local = np.arange(60, 64, dtype=np.int64)          # blocks 0-3
    graph, ops, return_mask = _run_get(
        engine, 4, _local_match(cpu_slots), _local_match(ssd_local))

    assert TransferType.DISK2D not in ops
    disk2h = ops[TransferType.DISK2H][0]
    np.testing.assert_array_equal(disk2h.src_block_ids, ssd_local[2:])

    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids[:2], cpu_slots)
    np.testing.assert_array_equal(h2d.src_block_ids[2:], disk2h.dst_block_ids)
    np.testing.assert_array_equal(h2d.dst_block_ids, np.arange(1000, 1004))
    assert disk2h.op_id in _graph_deps(graph, h2d.op_id)
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK

    # Staged, not promoted: the tail read off disk does not enter the CPU tree.
    engine.get_callback()                           # type: ignore[attr-defined]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]
    np.testing.assert_array_equal(
        engine.aborted_slots[0], disk2h.dst_block_ids)  # type: ignore[attr-defined]


def test_local_only_get_needs_no_staging():
    """A purely local CPU hit plans exactly one H2D straight from CPU."""
    engine = _global_cache_engine()
    cpu_slots = np.arange(40, 44, dtype=np.int64)
    free_before = engine.cpu_cache_engine.mempool.num_free_blocks
    graph, ops, return_mask = _run_get(engine, 4, _local_match(cpu_slots))
    assert set(ops) == {TransferType.H2D}
    np.testing.assert_array_equal(ops[TransferType.H2D][0].src_block_ids, cpu_slots)
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK
    # No staging taken, so nothing to publish and nothing to give back.
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before
    assert not _abort_hooks(graph)
    engine.get_callback()                           # type: ignore[attr-defined]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]


def test_span_layout_tiles_the_window_for_every_match_pair():
    """Sweep the span layout — every block-range decision the planner makes.

    `_get_impl_radixshmem` does no index arithmetic of its own: `_shm_get_spans`
    decides where the cuts are and resolves each span's slot ids. So this sweep
    over every small (window, CPU match, SSD match) triple is what pins that
    arithmetic down, including the combinations the scenario tests above do not
    reach — a window starting past a match, a match overrunning it, and an SSD
    tier that matched LESS than CPU did.

    What it asserts is exactly what the planner leans on:

      * the spans tile `[lo, end)` with no gap, no overlap, and no reordering;
      * the local CPU head is the only span read in place and always comes
        first, so the staged spans are a contiguous tail that one allocation can
        back, a running offset can walk, and one H2D can read;
      * every span carries as many ids as it has blocks — a peer op's
        `src_block_node_ids` is zipped positionally against its slots, so an
        off-by-one there reads the wrong node's memory instead of failing;
      * no source slot is handed to the H2D twice.
    """
    try:
        from flexkv.cache.cache_engine import _shm_get_spans
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cache_engine unavailable (needs CUDA + flexkv.c_ext): {exc}")

    def _match(num_local: int, num_peer: int, base: int) -> ShmRadixMatch:
        # Distinct id ranges per side and per tier, so a slot appearing in the
        # wrong op is visible rather than coincidentally right.
        return ShmRadixMatch(
            num_local_blocks=num_local,
            num_peer_blocks=num_peer,
            local_slots=np.arange(base, base + num_local, dtype=np.int64),
            peer_slots=np.arange(base + 100, base + 100 + num_peer, dtype=np.int64),
            peer_id=PEER_NODE_ID if num_peer else _engine_mod.NO_PEER,
        )

    span_range = range(5)
    for lo, hi in itertools.product(span_range, repeat=2):
        if lo > hi:
            continue
        for cpu_local, cpu_peer, ssd_local, ssd_peer in itertools.product(
                span_range, repeat=4):
            cpu_match = _match(cpu_local, cpu_peer, base=0)
            ssd_match = _match(ssd_local, ssd_peer, base=500)
            spans = _shm_get_spans(cpu_match, ssd_match, lo, hi)
            case = (f"lo={lo} hi={hi} cpu=({cpu_local},{cpu_peer}) "
                    f"ssd=({ssd_local},{ssd_peer})")

            cursor = lo
            for span in spans:
                assert span.start == cursor, case   # contiguous, in order
                assert span.end > span.start, case  # empty spans are dropped
                cursor = span.end
            assert cursor <= hi, case

            # At most one span is read in place, and it is the local CPU head.
            in_place = [i for i, s in enumerate(spans) if not s.needs_staging]
            assert in_place in ([], [0]), case

            num_staged = sum(len(s) for s in spans if s.needs_staging)
            end = spans[-1].end if spans else lo
            # Mirror the planner: one allocation for the staged tail, walked by a
            # running offset in span order.
            staging = np.arange(9000, 9000 + num_staged, dtype=np.int64)

            h2d_sources = []
            staged = 0
            for span in spans:
                assert len(span.src_block_ids) == len(span), f"{case}: {span}"
                node_ids = span.src_block_node_ids
                if node_ids is not None:
                    assert len(node_ids) == len(span), f"{case}: {span}"
                    assert set(node_ids.tolist()) == {PEER_NODE_ID}, case
                if span.needs_staging:
                    h2d_sources.append(staging[staged:staged + len(span)])
                    staged += len(span)
                else:
                    h2d_sources.append(span.src_block_ids)
            assert staged == num_staged, case

            if spans:
                h2d_src = np.concatenate(h2d_sources)
                assert len(h2d_src) == end - lo, case
                assert len(set(h2d_src.tolist())) == len(h2d_src), case


# =============================================================================
# Part 2b — PUT planning on the radixshmem backend (`_put_impl_radixshmem`)
#
# A PUT match is `with_peer=False` — never spliced — because `transfer_engine` routes
# no H2PEER*, so there is nowhere to write but our own slots. What differs from
# `_put_impl_local` is WHEN the tree learns about them: radixshmem takes a block
# only once it holds data, so both inserts run from graph completion. These tests
# pin the boundary the planner derives from the match (how much of the window is
# already cached, hence what D2H still has to move) and that the insert is armed.
# =============================================================================


def _run_put(engine,
             num_blocks: int,
             cpu_result: ShmRadixMatch,
             ssd_result: ShmRadixMatch | None = None):
    """Call put() through `_put_impl_radixshmem` with forced match results."""
    _force_radixshmem(engine, cpu_result, ssd_result)
    token_ids, token_mask, slot_mapping = _fake_request(num_blocks)
    graph, return_mask, callback, _op_cbs, _end = engine.put(
        request_id=2,
        token_ids=token_ids,
        token_mask=token_mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
    )
    ops = {}
    for op in graph._op_map.values():
        ops.setdefault(op.transfer_type, []).append(op)
    engine.put_callback = callback                   # type: ignore[attr-defined]
    return graph, ops, return_mask


def test_put_with_no_match_stores_the_whole_window():
    """Cold start: nothing cached, so D2H covers every block and the span publishes.

    The load-bearing case for the publish test in `_arm` — a fresh sequence has an
    EMPTY match, and if that read as "the tree does not reach the window start" the
    slots would be handed back and this backend would never cache anything at all.
    """
    engine = _global_cache_engine()
    graph, ops, return_mask = _run_put(engine, num_blocks=4,
                                       cpu_result=_local_match(np.array([])))

    op_d2h = ops[TransferType.D2H][0]
    assert op_d2h.src_block_ids.size == 4          # nothing skipped
    assert op_d2h.dst_block_ids.size == 4
    assert bool(return_mask.all())

    engine.put_callback()                           # graph completion
    # One insert, of the whole window, ending at block 4.
    assert [(n, len(s)) for n, s in engine.inserted_pools] == [(4, 4)]
    assert engine.aborted_slots == []


def test_put_skips_the_cached_prefix():
    """A partial CPU match: D2H moves only the blocks past it, and they publish.

    This is the boundary the planner reads off the match — `num_skipped` comes from
    intersecting the match with the window, and everything else (which GPU blocks
    D2H reads, the returned mask, where the published span starts) hangs off it.
    """
    engine = _global_cache_engine()
    cached = np.arange(20, 23, dtype=np.int64)      # 3 of 5 blocks already in CPU
    graph, ops, return_mask = _run_put(engine, num_blocks=5,
                                       cpu_result=_local_match(cached))

    op_d2h = ops[TransferType.D2H][0]
    assert op_d2h.src_block_ids.size == 2
    # GPU blocks are 1000.., so the skipped prefix is visible in the ids read.
    assert op_d2h.src_block_ids.tolist() == [1003, 1004]
    # The staged slots are fresh — never the ones the match already holds.
    assert not set(op_d2h.dst_block_ids.tolist()) & set(cached.tolist())
    # Only the newly stored blocks come back as stored.
    assert not bool(return_mask[:3 * TOKENS_PER_BLOCK].any())
    assert bool(return_mask[3 * TOKENS_PER_BLOCK:].all())

    engine.put_callback()
    # The span ends at block 5, and carries the 2 new blocks only.
    assert [(n, len(s)) for n, s in engine.inserted_pools] == [(5, 2)]
    assert engine.aborted_slots == []


def test_put_with_fully_cached_window_does_nothing():
    """A match covering the window ends the PUT: nothing to store, nothing to arm."""
    engine = _global_cache_engine()
    graph, ops, return_mask = _run_put(
        engine, num_blocks=3, cpu_result=_local_match(np.arange(20, 23)))
    assert ops == {}
    assert not bool(return_mask.any())
    assert engine.inserted_pools == []
    assert engine.aborted_slots == []


def test_put_cancelled_graph_returns_the_staged_slots():
    """Cancel before launch: the slots never held data, so they go back unpublished."""
    engine = _global_cache_engine()
    graph, _ops, _mask = _run_put(engine, num_blocks=4,
                                  cpu_result=_local_match(np.array([])))
    hooks = _abort_hooks(graph)
    assert len(hooks) == 1                          # one staged CPU insert
    for hook in graph.on_cancel:
        hook()
    assert [len(s) for s in engine.aborted_slots] == [4]
    assert engine.inserted_pools == []


# =============================================================================
# Part 3 — opt-in two-rank radixshmem/FlexKV peer-match integration over RDMA
#
# Boots a real 2-rank distributed radix cluster in two spawned processes: rank 0
# publishes a prefix into its own shm tree, rank 1 queries a prefix it has never
# seen and must find it on rank 0. Verifies the `ShmRadixMatch` such a hit
# produces — a peer tail, the peer's own block ids, and one FlexKV node id (== the
# peer's cluster rank) per peer block. A second case gives rank 1 a partial local
# prefix too and checks the SPLICE: the local head keeps rank 1's own slots, the
# tail past it comes from rank 0.
#
# Cluster membership goes through etcd (shmradix's only bootstrap path), so these
# need FLEXKV_TEST_RADIX_REGISTRY pointing at a reachable etcd endpoint on top of
# an ACTIVE RDMA device. Cluster rank is an OUTPUT of bootstrap, not the `rank`
# label handed in, so the expected node ids are read back off the writer.
# =============================================================================


def _active_rdma_devices() -> list[str]:
    """RDMA devices with at least one ACTIVE port, honoring the env override.

    FLEXKV_TEST_RDMA_DEVICES restricts (and orders) the candidates; anything it
    names that the host does not expose as ACTIVE is dropped, so a machine
    without RDMA yields [] and the tests below skip instead of failing inside a
    spawned rank.
    """
    def _has_active_port(device: str) -> bool:
        for state_file in glob.glob(
                f"/sys/class/infiniband/{device}/ports/*/state"):
            with contextlib.suppress(OSError):
                with open(state_file) as handle:
                    if "ACTIVE" in handle.read():
                        return True
        return False

    present = sorted(os.listdir("/sys/class/infiniband")) \
        if os.path.isdir("/sys/class/infiniband") else []
    requested = [d for d in os.getenv("FLEXKV_TEST_RDMA_DEVICES", "").split(",") if d]
    candidates = requested or present
    return [d for d in candidates if d in present and _has_active_port(d)]


def _require_cluster() -> tuple[str, str]:
    """(rdma device, etcd registry) for the ranks, or skip when unavailable."""
    if os.getenv("FLEXKV_RUN_RADIX_PEER_TEST") != "1":
        pytest.skip("set FLEXKV_RUN_RADIX_PEER_TEST=1 to run the RDMA test")
    devices = _active_rdma_devices()
    if not devices:
        pytest.skip(
            "no ACTIVE RDMA device found (checked "
            f"{os.getenv('FLEXKV_TEST_RDMA_DEVICES') or '/sys/class/infiniband'})"
        )
    registry = os.getenv("FLEXKV_TEST_RADIX_REGISTRY", "")
    if not registry:
        pytest.skip(
            "set FLEXKV_TEST_RADIX_REGISTRY=<etcd endpoint> — etcd is the only "
            "cluster bootstrap path shmradix has"
        )
    return devices[0], registry


def _rank_main(rank, prefix, cluster_id, registry, rdma_dev, ready, done, output,
               local_head_blocks=0, rht_slots_per_bucket=1):
    # Runs in a spawned child, which re-imports this module by name — so
    # `shmradix`, `CacheEngineRadixShmem` (side-loaded) and `DeviceType` are
    # already bound at module scope here, no per-rank imports needed.
    try:
        shm = shmradix.ShmConfig(
            max_nodes=1170,
            max_blocks=1170,
            block_size=16,
            data_pool_ratio=8,
            background_evict=True,
        )
        # Mirrors flexkv.server.shm_radix_bootstrap.create_shm_radix_regions:
        # membership goes through etcd, and RadixServer names the region
        # `name + "_" + node_name` — so node_name is what keeps two ranks on one
        # host from colliding on the same shm file.
        cfg = shmradix.RadixServerConfig()
        cfg.rht_slots_per_bucket = rht_slots_per_bucket
        cfg.name = prefix
        cfg.shm = shm
        cfg.node_name = f"r{rank}"
        cfg.rank = rank
        cfg.world_size = 2
        # The membership gate is max(num_shards, expected_min_nodes), and only
        # num_shards is reachable from Python — without it bootstrap completes
        # with a single node and the peer is never seen.
        cfg.num_shards = 2
        cfg.registry = registry
        cfg.cluster_id = cluster_id
        # Single-host test: peers dial back on the loopback-reachable wildcard.
        cfg.rpc_address = "0.0.0.0"
        cfg.rdma_dev = rdma_dev
        cfg.gid_idx = int(os.getenv("FLEXKV_RADIX_GID_IDX", "3"))
        cfg.bootstrap_timeout_sec = 30

        server = shmradix.RadixServer(cfg)
        # This ctor does NOT create the region; bootstrap() does, collectively —
        # it blocks until both ranks have joined the etcd namespace.
        if not server.bootstrap():
            raise RuntimeError("server bootstrap returned false")
        if not server.is_distributed():
            raise RuntimeError(
                "region bootstrapped but reports world_size="
                f"{server.world_size()}; shmradix was built without RDMA")
        # etcd assigns dense cluster ranks by sorted peer-key order, so the
        # cluster rank is an OUTPUT — it need not equal the label above.
        cluster_rank = int(server.rank())

        engine = CacheEngineRadixShmem(
            device_type=DeviceType.CPU,
            num_total_blocks=1170,
            tokens_per_block=16,
            shm_name=server.shm_name(),
            peer_enabled=True,
        )
        if not engine.peer_enabled:
            raise RuntimeError("engine did not see a distributed region")

        hashes = np.arange(26, dtype=np.uint64) * 104729 + 101
        query_hashes = hashes[:-1]

        def _seq(block_hashes):
            return FakeSeq(block_hashes=block_hashes.view(np.int64),
                           tokens_per_block=16)

        if rank == 0:
            sequence = _seq(hashes)
            slots = engine.take(num_required_blocks=len(hashes), strict=True)
            if len(slots) != len(hashes):
                raise RuntimeError(f"took={len(slots)}")
            # insert() publishes (the data is notionally already there) and, with
            # peer_enabled, flushes the RHT so rank 1 can route to us.
            engine.insert(
                sequence, slots,
                num_insert_blocks=len(hashes),
            )
            output.put({"writer_slots": slots.tolist(),
                        "writer_rank": cluster_rank})
            ready.set()
            if not done.wait(20):
                raise TimeoutError("reader did not complete")
        else:
            head_slots = np.array([], dtype=np.int64)
            if local_head_blocks > 0:
                # Give this rank a shorter local prefix of its own, so the match
                # has a local head to splice the peer's tail onto.
                head = hashes[:local_head_blocks]
                head_slots = engine.take(
                    num_required_blocks=local_head_blocks, strict=True)
                engine.insert(
                    _seq(head), head_slots,
                    num_insert_blocks=local_head_blocks)
            if not ready.wait(20):
                raise TimeoutError("writer did not publish")
            # Poll until the writer's prefix is routable (its RHT publication is
            # asynchronous). The query continues onto the peer past whatever we
            # hold locally, so the full length is reached either way.
            expect = len(query_hashes)
            result = None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                result = engine.match(_seq(query_hashes))
                if result.num_matched_blocks >= expect:
                    break
                result.release()
                time.sleep(0.01)
            if result is None or result.num_matched_blocks < expect:
                raise AssertionError(
                    f"expected >= {expect} matched blocks, got "
                    f"{result.num_matched_blocks if result else None}"
                )
            num_local = result.num_local_blocks
            output.put({
                "has_peer_tail": result.has_peer_tail,
                "num_matched": result.num_matched_blocks,
                "num_local": num_local,
                "local_slots": result.local_slots.tolist(),
                "peer_slots": result.peer_slots.tolist(),
                # What the PEERH2H op would carry for the tail: one owner id per
                # block, sliced to exactly the peer's range.
                "peer_node_ids": result.peer_node_ids(
                    num_local, result.num_matched_blocks).tolist(),
                "local_head_slots": np.asarray(head_slots).tolist(),
            })
            result.release()
            done.set()
    except Exception:
        output.put({"error": traceback.format_exc(), "rank": rank})
        ready.set()
        done.set()


def _run_two_ranks(registry, rdma_dev, local_head_blocks=0,
                   rht_slots_per_bucket=1):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()
    output = ctx.Queue()
    prefix = f"/flexkv_radix_peer_test_{os.getpid()}_{local_head_blocks}"
    # The cluster id is the etcd namespace the two ranks meet in. Carrying the
    # pid and the case keeps a rerun (or a leftover key from a crashed run) from
    # being counted as a third member of this cluster.
    cluster_id = f"flexkv-peer-test-{os.getpid()}-{local_head_blocks}"

    processes = [
        ctx.Process(
            target=_rank_main,
            args=(rank, prefix, cluster_id, registry, rdma_dev, ready, done,
                  output, local_head_blocks, rht_slots_per_bucket),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    messages = []
    while not output.empty():
        messages.append(output.get())
    errors = [message for message in messages if "error" in message]
    assert not errors, errors
    assert all(process.exitcode == 0 for process in processes)
    return (next(m for m in messages if "num_matched" in m),
            next(m for m in messages if "writer_slots" in m))


def test_radixshmem_single_peer_match_over_rdma():
    """Rank 1's tree is empty → the splice is all tail, no head."""
    rdma_dev, registry = _require_cluster()
    reader, writer = _run_two_ranks(registry, rdma_dev)

    assert reader["has_peer_tail"]
    assert reader["num_matched"] == 25
    # Nothing local to splice onto, so the boundary sits at 0 and every block is
    # the peer's.
    assert reader["num_local"] == 0
    assert reader["local_slots"] == []
    # Peer-owned blocks carry rank 0's node id, which IS its cluster rank.
    assert reader["peer_node_ids"] == [writer["writer_rank"]] * 25
    # The reported slots are rank 0's block ids, resolved against ITS buffer.
    assert reader["peer_slots"] == writer["writer_slots"][:25]


def test_radixshmem_peer_tail_extends_a_local_prefix():
    """A local head plus a peer tail come back as ONE spliced match.

    Rank 1 holds blocks 0-9 locally while rank 0 holds 0-25. The match walks the
    local tree as far as it goes (10 blocks, rank 1's own slots), then continues
    on the peer for 10-25. `num_local_blocks` is where one becomes the other, and
    the two slot arrays stay apart because they mean different buffers.
    """
    rdma_dev, registry = _require_cluster()
    reader, writer = _run_two_ranks(
        registry, rdma_dev, local_head_blocks=10, rht_slots_per_bucket=4)

    assert reader["has_peer_tail"]
    assert reader["num_matched"] == 25
    assert reader["num_local"] == 10
    # 15 blocks of tail, every one addressed to rank 0 — no -1 leaks in, which is
    # what would happen if the head were included in the op's node ids.
    assert reader["peer_node_ids"] == [writer["writer_rank"]] * 15
    # The head resolves against rank 1's own buffer, the tail against rank 0's.
    assert reader["local_slots"] == reader["local_head_slots"]
    assert reader["peer_slots"] == writer["writer_slots"][10:25]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
