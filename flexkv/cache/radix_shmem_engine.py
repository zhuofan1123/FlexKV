# SPDX-License-Identifier: Apache-2.0
"""
RadixShmem-backed CacheEngine.

A drop-in replacement for `flexkv.cache.cache_engine.CacheEngineAccel` whose
RadixTree + slot Mempool live in POSIX shared memory (via the `shmradix`
package, https://github.com/.../radixshmem). Every DP scheduler process can
attach to the same shm region and run prefix queries / inserts in parallel,
serialised only by a process-shared rwlock.

Public surface mirrors `CacheEngineAccel` so that `GlobalCacheEngine` and the
`_get_impl_local`/`_get_impl_global`/`_put_*` helpers in `cache_engine.py` can
treat both backends uniformly: `match()` returns a `MatchResultAccel`, `insert`
returns an opaque "node" handle that has `.size()`, etc.

Differences from `CacheEngineAccel`:
- The slot mempool is owned by radixshmem (one mempool per shm region). The
  cache engine no longer holds its own `flexkv.cache.mempool.Mempool`. `take()`
  forwards to `tree.allocate_slots()` and `recycle()` forwards to
  `tree.recycle_slots()`.
- `lock_node`/`unlock`/`set_ready(node, ...)` use the radixshmem node_id stored
  on the `ShmRadixNode` wrapper.
- `evict()` is performed implicitly by radixshmem's auto-evict during
  `allocate_slots`. The standalone `evict()` API is not used by FlexKV's
  `take()` path on this backend.

Slot IDs returned by radixshmem are `int32`; FlexKV expects `int64`. We cast at
the boundary (`np.asarray(..., dtype=np.int64)`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType
from flexkv.common.type import MatchResultAccel

if TYPE_CHECKING:
    # `flexkv.common.block` (and `flexkv.integration.dynamo.collector`) pull in
    # the FlexKV C++ extension transitively. Keep them out of import-time so
    # this module can be loaded for unit tests without CUDA/libtorch.
    from flexkv.common.block import SequenceMeta
    from flexkv.integration.dynamo.collector import KVEventCollector

try:
    import shmradix
except ImportError as e:  # pragma: no cover
    shmradix = None
    _SHMRADIX_IMPORT_ERROR = e
else:
    _SHMRADIX_IMPORT_ERROR = None


_DEVICE_TYPE_NAMES = ['CPU', 'GPU', 'SSD', 'REMOTE']

# Sentinel for "no node" — radixshmem uses uint32 max.
INVALID_NODE_ID = 0xFFFFFFFF


@dataclass
class ShmRadixNode:
    """Lightweight handle around a radixshmem node_id.

    Mirrors the small subset of `CRadixNode` semantics that FlexKV's
    `cache_engine.py` actually uses: `.size()` (number of blocks contributed by
    this node) and pass-through to `lock_node`/`unlock`/`set_ready`.

    `ready_length` records how many blocks of this node are ready when the
    handle was created — FlexKV calls `set_ready(node, ready, ready_length)`
    later to flip them.
    """
    node_id: int
    num_blocks: int

    def size(self) -> int:
        return self.num_blocks

    def is_valid(self) -> bool:
        return self.node_id != INVALID_NODE_ID


def _ensure_shmradix():
    if shmradix is None:
        raise ImportError(
            "shmradix is not installed; install it from radixshmem repo "
            "(pip install -e radixshmem/python). Original error: "
            f"{_SHMRADIX_IMPORT_ERROR}"
        )


class CacheEngineRadixShmem:
    """Radixshmem-backed cache engine for one device (CPU / SSD / REMOTE).

    Multiple instances (one per DP scheduler process) attach to the same shm
    region by name and concurrently query / insert.
    """

    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 shm_name: str,
                 evict_ratio: float = 0.05,
                 evict_start_threshold: float = 1.0,
                 hit_reward_seconds: int = 0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector=None,
                 protected_threshold: int = 2):
        """Attach to an existing radix shm region by name. The TreeServer
        owning the region must have been created elsewhere (e.g. by
        `flexkv.server.shm_radix_bootstrap.create_shm_radix_regions`)."""
        _ensure_shmradix()

        if eviction_policy != "lru":
            flexkv_logger.warning(
                f"radixshmem only supports LRU eviction; ignoring "
                f"eviction_policy={eviction_policy!r}"
            )
        # `hit_reward_seconds` and `protected_threshold` aren't expressible in
        # radixshmem yet — we accept them for ABI compatibility with
        # CacheEngineAccel and warn if non-default values are requested.
        if hit_reward_seconds != 0 or protected_threshold != 2:
            flexkv_logger.debug(
                "radixshmem ignores hit_reward_seconds and protected_threshold"
            )

        self.device_type = device_type
        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold
        self.shm_name = shm_name

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # TreeClient always; TreeServer is owned by the bootstrap process.
        self._tree = shmradix.TreeClient(shm_name)
        # node_id -> (hashes_uint64_copy, matched_prefix, inserted_count).
        # FlexKV calls insert(is_ready=False) then set_ready(node, True, length)
        # later; radixshmem set_ready needs hashes, so remember them here.
        self._pending_ready: dict = {}

    # ---------- Mempool view (compatibility shims for CacheEngineAccel API) ----------

    @property
    def mempool(self) -> _MempoolView:
        return _MempoolView(self._tree)

    # ---------- Lifecycle ----------

    def reset(self) -> None:
        # radixshmem has no public reset; closest is removing all entries one
        # by one. For now, we only support reset by destroying and re-creating
        # the shm region. Used in tests; not exercised in production.
        flexkv_logger.warning(
            "CacheEngineRadixShmem.reset(): radixshmem has no in-place reset; "
            "tear down and recreate the shm region instead."
        )

    def close(self) -> None:
        self._tree = None

    # ---------- Match / insert / lock ----------

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        # SequenceMeta.block_hashes is int64; radixshmem expects uint64. They
        # share the same byte width, so view-cast is safe.
        hashes = sequence_meta.block_hashes.view(np.uint64)

        detail = self._tree.query_detail(hashes)

        # Build MatchResultAccel mirroring CacheEngineAccel.match().
        last_node_id = detail.last_node_id
        last_ready_id = detail.last_ready_node_id

        if last_node_id != INVALID_NODE_ID and last_node_id != 0:
            # last_node carries the entire node it represents (or its full size
            # if we descended into it via match-then-stop). For the unified
            # cache_engine code we expose `.size()` = matched length within node.
            last_node = ShmRadixNode(node_id=int(last_node_id),
                                      num_blocks=int(detail.last_node_matched_blocks))
        else:
            last_node = None

        if last_ready_id != INVALID_NODE_ID and last_ready_id != 0:
            # FlexKV uses last_ready_node primarily to lock the path against
            # eviction. The exact size attached to it is informational; cap by
            # ready_prefix_len.
            last_ready_node = ShmRadixNode(node_id=int(last_ready_id),
                                            num_blocks=int(detail.ready_prefix_len))
        else:
            last_ready_node = None

        physical = np.asarray(detail.slots, dtype=np.int64)

        return MatchResultAccel(
            num_ready_matched_blocks=int(detail.ready_prefix_len),
            num_matched_blocks=int(detail.matched_blocks),
            last_ready_node=last_ready_node,
            last_node=last_node,
            last_node_matched_length=int(detail.last_node_matched_blocks),
            physical_blocks=physical,
            block_node_ids=None,
            matched_pos="local",
        )

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Optional[ShmRadixNode]:
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)

        # FlexKV pattern: `physical_block_ids` are slots that the caller
        # reserved via `take()` for the SUFFIX (the new portion beyond the
        # existing matched prefix). Length of `physical_block_ids` may be the
        # full prefix or just the suffix depending on call site; radixshmem
        # walks the prefix internally and only consumes as many supplied slots
        # as it actually attaches.
        suffix_slots = np.asarray(physical_block_ids, dtype=np.int32)

        if num_insert_blocks > 0:
            # Caller wants to register only the first `num_insert_blocks`
            # blocks of the prefix. Trim the hash array to that length so
            # radixshmem doesn't try to attach beyond.
            target_hashes = hashes[:num_insert_blocks]
        else:
            target_hashes = hashes

        result = self._tree.insert_with_slots(target_hashes, suffix_slots, is_ready)

        if self.event_collector is not None and result.inserted_count > 0:
            attached_hashes = sequence_meta.block_hashes[
                result.matched_prefix : result.matched_prefix + result.inserted_count
            ]
            self.event_collector.publish_stored(
                block_hashes=attached_hashes,
                block_size=self.tokens_per_block,
                medium=_DEVICE_TYPE_NAMES[self.device_type]
            )

        if result.last_node_id == INVALID_NODE_ID or result.inserted_count <= 0:
            return None
        # Remember the hash path so set_ready(node, ...) can walk it later.
        # Stores (hashes up to end of inserted suffix, matched_prefix offset,
        # inserted_count). set_ready marks only the inserted suffix ready —
        # ancestor nodes are the responsibility of their own insert/set_ready.
        matched_prefix = int(result.matched_prefix)
        inserted_count = int(result.inserted_count)
        total_path_len = matched_prefix + inserted_count
        self._pending_ready[int(result.last_node_id)] = (
            np.ascontiguousarray(target_hashes[:total_path_len]).copy(),
            matched_prefix,
            inserted_count,
        )
        return ShmRadixNode(node_id=int(result.last_node_id),
                             num_blocks=int(result.inserted_count))

    def lock_node(self, node: ShmRadixNode) -> None:
        if node is None or not node.is_valid():
            return
        self._tree.inc_ref_node(node.node_id)

    def unlock(self, node: ShmRadixNode) -> None:
        if node is None or not node.is_valid():
            return
        self._tree.dec_ref_node(node.node_id)

    def set_ready(self, node: ShmRadixNode, ready: bool, ready_length: int) -> None:
        if node is None or not node.is_valid():
            return
        entry = self._pending_ready.pop(int(node.node_id), None)
        if entry is None:
            flexkv_logger.debug(
                f"CacheEngineRadixShmem.set_ready: no pending entry for "
                f"node_id={node.node_id} (insert from another process or "
                f"already set_ready)."
            )
            return
        hashes, matched_prefix, inserted_count = entry
        # Mark only the NEWLY inserted suffix ready. Marking the ancestor
        # blocks ready here would expose them before their own D2H finishes,
        # serving stale data -> garbled inference output.
        self._tree.set_ready(hashes, matched_prefix, inserted_count, bool(ready))

    def set_ready_path(self,
                       sequence_meta: SequenceMeta,
                       start: int,
                       length: int,
                       ready: bool) -> None:
        """Path-based set_ready that walks a SequenceMeta hash path."""
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)
        self._tree.set_ready(hashes, start, length, ready)

    # ---------- Mempool ops (take/recycle) ----------

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[ShmRadixNode] = None,
             strict: bool = True) -> np.ndarray:
        """Allocate `num_required_blocks` slots from radixshmem's mempool.

        radixshmem's `allocate_slots` will auto-evict LRU entries to satisfy
        the request. `protected_node` is locked across the call to prevent it
        from being evicted; we wrap inc_ref/dec_ref around the call.
        """
        if protected_node is not None and protected_node.is_valid():
            self._tree.inc_ref_node(protected_node.node_id)
        try:
            slots_i32 = self._tree.allocate_slots(num_required_blocks)
        finally:
            if protected_node is not None and protected_node.is_valid():
                self._tree.dec_ref_node(protected_node.node_id)

        slots = np.asarray(slots_i32, dtype=np.int64)

        if strict and len(slots) < num_required_blocks:
            # Caller will recycle whatever we returned; mirror CacheEngineAccel
            # by raising on shortfall.
            self._tree.recycle_slots(np.asarray(slots, dtype=np.int32))
            raise RuntimeError(
                f"radixshmem: not enough free blocks to take, required: "
                f"{num_required_blocks}, available: {len(slots)}"
            )

        if self._metrics_collector is not None and len(slots) > 0:
            self._metrics_collector.record_allocation(
                _DEVICE_TYPE_NAMES[self.device_type].lower(), len(slots)
            )
        return slots

    def recycle(self, physical_blocks: np.ndarray) -> None:
        if physical_blocks is None or len(physical_blocks) == 0:
            return
        slots_i32 = np.asarray(physical_blocks, dtype=np.int32)
        self._tree.recycle_slots(slots_i32)

    # ---------- Stats passthrough ----------

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())

    @property
    def total_nodes(self) -> int:
        return int(self._tree.total_nodes())


@dataclass
class _MempoolView:
    """Read-only mempool view that lets `cache_engine.py` query free/used
    counts via `engine.mempool.num_free_blocks` etc."""
    _tree: object

    @property
    def num_total_blocks(self) -> int:
        return int(self._tree.mempool_total())

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())
