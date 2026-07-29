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
- radixshmem exposes no node_id (nodes split on later inserts), so writes are
  addressed by the hash path + (start, length). `insert` carries the
  InsertResult.finalize on the `ShmRadixNode`; `lock_node`/`unlock` are no-ops
  (insert already inc_ref'd via lock=True) and `release_node` calls finalize to
  flip the suffix ready and drop the ref in one shot.
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


@dataclass
class ShmRadixNode:
    """Handle around a radixshmem insert result. radixshmem no longer exposes
    node_id (nodes split on later inserts), so state is addressed by the hash
    path + (matched_prefix, inserted_count). Carries the InsertResult.finalize,
    which packs "set_ready(inserted suffix) + dec_ref" into one idempotent call.

    Mirrors the `CRadixNode` subset cache_engine.py uses: `.size()` = the
    inserted suffix length that set_ready(node, True, ready_length) flips.
    """
    finalize: object  # InsertResult.finalize — one-shot set_ready + dec_ref
    hashes: np.ndarray
    matched_prefix: int
    inserted_count: int

    def size(self) -> int:
        return self.inserted_count

    def is_valid(self) -> bool:
        return self.inserted_count > 0


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

        # RadixClient attaches to a shm region created by a RadixServer owned
        # by the bootstrap process.
        self._tree = shmradix.RadixClient(shm_name)

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

        # lock=True inc_ref's the matched ready prefix so it can't be evicted
        # (and its slots recycled) before the transfer consumes them; release
        # via qr.finalize, owned by the cache_engine layer. No node_id handles.
        qr = self._tree.query(hashes, lock=True)

        physical = np.asarray(qr.ready_prefix_slots, dtype=np.int64)

        return MatchResultAccel(
            num_ready_matched_blocks=int(qr.ready_prefix_len),
            num_matched_blocks=int(qr.total_hit_length),
            last_ready_node=None,
            last_node=None,
            last_node_matched_length=0,
            physical_blocks=physical,
            block_node_ids=None,
            matched_pos="local",
            finalize=qr.finalize,
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

        # lock=True inc_ref's the newly inserted leaf so it can't be evicted
        # before the transfer writes its KV; released via result.finalize (which
        # also flips the suffix ready). Even is_ready=True needs it — lock_node
        # is a no-op on this backend, so finalize is the only protection.
        result = self._tree.insert_with_slots(target_hashes, suffix_slots, is_ready, lock=True)

        if self.event_collector is not None and result.inserted_count > 0:
            attached_hashes = sequence_meta.block_hashes[
                result.matched_prefix : result.matched_prefix + result.inserted_count
            ]
            self.event_collector.publish_stored(
                block_hashes=attached_hashes,
                block_size=self.tokens_per_block,
                medium=_DEVICE_TYPE_NAMES[self.device_type]
            )

        matched_prefix = int(result.matched_prefix)
        inserted_count = int(result.inserted_count)
        if inserted_count <= 0:
            # Nothing new attached (full prefix already present, or no slots);
            # finalize is empty/no-op, so there is no node to release later.
            return None
        # Keep the hash path so set_ready / finalize can re-walk it (radixshmem
        # is split-invariant, addressed by hashes + (start, length)).
        total_path_len = matched_prefix + inserted_count
        return ShmRadixNode(
            finalize=result.finalize,
            hashes=np.ascontiguousarray(target_hashes[:total_path_len]).copy(),
            matched_prefix=matched_prefix,
            inserted_count=inserted_count,
        )

    def lock_node(self, node: ShmRadixNode) -> None:
        # No-op: insert already inc_ref'd the leaf (lock=True). Protection is
        # released by release_node -> node.finalize().
        pass

    def unlock(self, node: ShmRadixNode) -> None:
        # No standalone unlock: the insert-time ref is released only through
        # release_node (node.finalize), which also flips the suffix ready.
        pass

    def set_ready(self, node: ShmRadixNode, ready: bool, ready_length: int) -> None:
        if node is None or not node.is_valid():
            return
        # Idempotent, does NOT release the ref (unlike finalize). Used by
        # _op_callback to expose the suffix as soon as its own op completes,
        # while the node stays locked until the whole graph finishes. Marks
        # only the NEWLY inserted suffix — flipping ancestors early would serve
        # data before their own transfer lands.
        self._tree.set_ready(node.hashes, node.matched_prefix,
                             node.inserted_count, bool(ready))

    def release_node(self, node: ShmRadixNode, ready_length: int) -> None:
        """Transfer-complete release. finalize packs set_ready(suffix) +
        dec_ref; ``ready_length`` is implicit (== inserted_count) so it is
        unused here. Idempotent — safe if set_ready already ran."""
        if node is None or not node.is_valid():
            return
        node.finalize()

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
        the request. `protected_node` is ignored: on this backend match() sets
        last_node=None and protects the matched prefix via its own finalize ref
        instead, so there is nothing extra to inc_ref here.
        """
        slots_i32 = self._tree.allocate_slots(num_required_blocks)

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
        return int(self._tree.total_radix_nodes())


@dataclass
class _MempoolView:
    """Read-only mempool view that lets `cache_engine.py` query free/used
    counts via `engine.mempool.num_free_blocks` etc."""
    _tree: object

    @property
    def num_total_blocks(self) -> int:
        return int(self._tree.total_blocks())

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())
