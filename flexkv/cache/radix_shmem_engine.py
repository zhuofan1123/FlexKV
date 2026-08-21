# SPDX-License-Identifier: Apache-2.0
"""
RadixShmem-backed CacheEngine.

A drop-in replacement for `flexkv.cache.cache_engine.CacheEngineAccel` whose
RadixTree + slot Mempool live in POSIX shared memory (via `shmradix`). Every DP
scheduler process attaches to the same region by name and runs prefix queries /
inserts in parallel, serialised only by a process-shared rwlock.

Differences from `CacheEngineAccel`:

- The slot mempool is owned by radixshmem (one per region), so there is no
  `flexkv.cache.mempool.Mempool` here: `take()`/`recycle()` forward to
  `allocate_slots()`/`recycle_slots()`. Slot ids come back int32, cast to int64
  at the boundary. Eviction is implicit inside `allocate_slots`.
- No node_id (nodes split on later inserts), so everything is addressed by hash
  path + (start, length): `insert()` hands nothing back and `take()` accepts no
  `protected_node`. There is no lock/unlock on the write path either -- insert
  runs after the transfer, so the span has no reader to protect. The one refcount
  FlexKV holds is the READ side's: `query(lock=True)`, released via
  `QueryResult.finalize`.
- `match()` returns a `ShmRadixMatch`, not a `MatchResultAccel`, and there is no
  `match_all`/`match_local` pair -- local-vs-cluster is one `with_peer=` flag.

Insert happens AFTER transfer. There is no "ready" bit; a block is published by
being in the tree at all, so the order is `take() -> transfer -> insert()`.
Consequences: `insert()` must be called from the completion path, not while
building the graph; a failed or cancelled transfer leaves slots attached to
nothing (neither reachable nor evictable) which leak unless `recycle()`d; and
`insert(auto_recycle=True)` takes slot ownership, so the caller must not recycle
the same slots again.

Distributed (peer) reuse: with `peer_enabled` on a clustered region, `match()`
walks the local tree, routes through the cluster's router hash table, and
CONTINUES on a peer's tree over RDMA. The two spans are SPLICED, not ranked:

    prefix_slots  covers blocks [0, local_hit_length)          -> local slots
    remote_slots  covers [local_hit_length, total_hit_length)  -> peer slots

`query(lock=True)` inc_refs both sides and `finalize` drops both; cache_engine
defers finalizers to graph completion so the peer cannot evict mid-PEERH2H.

Known gap: the router is probed only past the local hit, so a peer holding a
longer prefix that diverges INSIDE the local hit is never found.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType

if TYPE_CHECKING:
    # These pull in the FlexKV C++ extension transitively; keep them out of
    # import-time so this module loads without CUDA/libtorch.
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

# `ShmRadixMatch.peer_id` when the match has no peer tail. Also what shmradix
# itself reports as `remote_node_id` for a purely local query.
NO_PEER = -1


@dataclass
class ShmRadixMatch:
    """One radixshmem prefix query: a local head plus a tail on at most one peer.

    `MatchResultAccel`'s single `matched_pos` can say "local" or "remote" but not
    "local up to here, then that peer", which is exactly what a radixshmem query
    is -- hence this backend's own type.

        block index   0          num_local_blocks        num_matched_blocks
                      |  local_slots (our mempool)  |  peer_slots (peer_id's)  |

    Slot ids are per-owner, so the two arrays must never be concatenated without
    carrying the owner along. Accessors take ABSOLUTE block indices, and the peer
    ones reject a range crossing the boundary: the transfer worker zips
    `src_block_node_ids` positionally against `src_block_ids`, so an off-by-one
    reads the wrong node's memory rather than failing.

    The query ran with `lock=True`, so refs on both sides live until `release()`,
    which must run on every path or the prefix is pinned for the region's life.
    """
    num_local_blocks: int = 0
    num_peer_blocks: int = 0
    local_slots: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    peer_slots: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))
    peer_id: int = NO_PEER
    finalize: Optional[Callable[[], None]] = None

    @property
    def num_matched_blocks(self) -> int:
        return self.num_local_blocks + self.num_peer_blocks

    @property
    def has_peer_tail(self) -> bool:
        return self.num_peer_blocks > 0

    def local_range(self, first: int, last: int) -> np.ndarray:
        """Whatever part of absolute block range [first, last) the local head holds.

        The slice does all the bounding both ways, so callers need no boundary
        arithmetic; overrunning the head is not an error, those blocks simply are
        not ours. Both bounds must be non-negative.
        """
        return self.local_slots[first:last]

    def peer_range(self, first: int, last: int) -> np.ndarray:
        """The peer's slot ids for absolute block range [first, last)."""
        if first >= last:
            return self.peer_slots[:0]
        base = self.num_local_blocks
        assert base <= first <= last <= self.num_matched_blocks, (
            f"[{first}, {last}) is not inside the peer tail "
            f"[{base}, {self.num_matched_blocks})"
        )
        return self.peer_slots[first - base:last - base]

    def peer_node_ids(self, first: int, last: int) -> np.ndarray:
        """Owner ids to pair with `peer_range(first, last)`, one per block.

        A match has a single peer, so this is a constant run — but the worker
        wants it per block, and building it here keeps the length tied to the
        same range check as the slots it accompanies.
        """
        if first >= last:
            return np.empty(0, dtype=np.int64)
        assert self.has_peer_tail, "no peer tail to address"
        base = self.num_local_blocks
        assert base <= first <= last <= self.num_matched_blocks, (
            f"[{first}, {last}) is not inside the peer tail "
            f"[{base}, {self.num_matched_blocks})"
        )
        return np.full(last - first, self.peer_id, dtype=np.int64)

    def release(self) -> None:
        """Drop the query's refs on both sides. Idempotent."""
        finalize, self.finalize = self.finalize, None
        if finalize is not None:
            finalize()


class StagedRadixInsert:
    """Attach staged slots to a radixshmem tree once their data has landed.

    radixshmem admits a block only when it already holds data, so the insert has
    to run from a completion callback. Until then the slots are owned by nobody
    but this object, and exactly one of its two exits must run or they are lost
    for the life of the region:

      ``publish`` -- transfer landed; hand the slots to the tree, which takes
                     ownership and recycles whatever did not attach
      ``abort``   -- data never landed; give them straight back to the mempool

    Mutually exclusive and each idempotent, so completion and cancel can both be
    armed. ``publish`` takes no ref on what it attached (the transfer is over, so
    the span has no reader), but it IS only legal while the ref keeping the local
    tree reaching the span's start is held -- pass that as ``holds`` and both exits
    drop it afterwards.
    """

    def __init__(self,
                 engine: "CacheEngineRadixShmem",
                 sequence_meta: "SequenceMeta",
                 slots: np.ndarray,
                 path_end: int,
                 label: str,
                 holds: Sequence[Callable[[], None]] = ()) -> None:
        self._engine = engine
        self._sequence_meta = sequence_meta
        self._slots = slots
        self._path_end = path_end
        self._label = label
        self._holds = list(holds)
        self._settled = False

    def publish(self) -> None:
        if self._settled:
            return
        self._settled = True
        try:
            # The staged span always ENDS at path_end, so insert() derives
            # start = path_end - len(slots) on its own.
            self._engine.insert(
                self._sequence_meta,
                self._slots,
                num_insert_blocks=self._path_end,
            )
        except Exception as e:
            flexkv_logger.error(
                f"radixshmem {self._label}: insert of {len(self._slots)} "
                f"staged slots failed: {e}; returning them to the mempool"
            )
            self._engine.recycle(self._slots)
        finally:
            self._release_holds()

    def abort(self) -> None:
        if self._settled:
            return
        self._settled = True
        flexkv_logger.debug(
            f"radixshmem {self._label}: returning {len(self._slots)} staged "
            f"slots, their data never landed"
        )
        try:
            self._engine.recycle(self._slots)
        finally:
            self._release_holds()

    def _release_holds(self) -> None:
        # In a finally, and one try each: a ref left taken pins its prefix for
        # the life of the shm region, and no attached process can undo that.
        holds, self._holds = self._holds, []
        for release in holds:
            try:
                release()
            except Exception as e:  # keep dropping the rest
                flexkv_logger.error(
                    f"radixshmem {self._label}: ref release failed: {e}")


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
                 tokens_per_block: int,  # -1 => recover from region via block_size()
                 shm_name: str,
                 evict_ratio: float = 0.05,
                 evict_start_threshold: float = 1.0,
                 hit_reward_seconds: int = 0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector=None,
                 protected_threshold: int = 2,
                 peer_enabled: bool = False):
        """Attach to an existing radix shm region by name; the owning RadixServer
        must already have been created (see `shm_radix_bootstrap`).

        `peer_enabled` turns on cross-node reuse: GET matches query the whole
        cluster and may come back spliced local-head + peer-tail. The cluster RANK
        shmradix reports IS the FlexKV node id the peer data path addresses.
        """
        _ensure_shmradix()

        if eviction_policy != "lru":
            flexkv_logger.warning(
                f"radixshmem only supports LRU eviction; ignoring "
                f"eviction_policy={eviction_policy!r}"
            )
        # Not expressible in radixshmem yet; accepted for ABI compatibility with
        # CacheEngineAccel.
        if hit_reward_seconds != 0 or protected_threshold != 2:
            flexkv_logger.debug(
                "radixshmem ignores hit_reward_seconds and protected_threshold"
            )

        self.device_type = device_type
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold
        self.shm_name = shm_name

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        self.peer_enabled = peer_enabled
        self._trace_peer = os.getenv("FLEXKV_TRACE_RADIX_PEER", "0") == "1"

        from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
        from flexkv.server.shm_radix_bootstrap import attach_radix_client
        expect_cluster = self.peer_enabled and getattr(
            GLOBAL_CONFIG_FROM_ENV, "radix_world_size", 1) > 1
        self._tree = attach_radix_client(shm_name,
                                         expect_distributed=expect_cluster)
        if self.peer_enabled and not self._tree.is_distributed():
            flexkv_logger.warning(
                f"radixshmem peer matching is enabled for {shm_name} but the "
                f"attached region has world_size=1; GETs stay local-only"
            )
            self.peer_enabled = False

        # -1 => recover tokens_per_block from the region itself
        # (RadixClient.block_size(), written by the owner on create).
        if tokens_per_block is None or tokens_per_block < 0:
            tokens_per_block = int(self._tree.block_size())
        self.tokens_per_block = tokens_per_block

    # ---------- Mempool view (compatibility shims for CacheEngineAccel API) ----------

    @property
    def mempool(self) -> _MempoolView:
        return _MempoolView(self._tree)

    # ---------- Lifecycle ----------

    def reset(self) -> None:
        """Clear this node's cached data in place.

        Invalidates every slot id and match result taken before the call, so it is
        only safe with no transfer in flight.
        """
        self._tree.reset()

    def close(self) -> None:
        self._tree = None

    def start(self) -> None:
        """No-op; the peer-capable cache engine lifecycle calls this."""

    # ---------- Match ----------

    def match(self,
              sequence_meta: SequenceMeta,
              *,
              with_peer: bool = True,
              gpu_matched_blocks: int = 0) -> ShmRadixMatch:
        """Prefix-match against the shared (and maybe peer) index.

        `with_peer=False` (or a non-distributed region) restricts the walk to the
        local tree -- what PUT wants, since PUT only ever writes locally.
        `gpu_matched_blocks` is accepted for parity with the accel/hie engines.
        The result is a SINGLE spliced match: a local head and, past it, a tail on
        one peer. See `ShmRadixMatch`.
        """
        local_only = not (with_peer and self.peer_enabled)
        sequence_meta.gen_hashes()
        # SequenceMeta.block_hashes is int64; radixshmem expects uint64. They
        # share the same byte width, so view-cast is safe.
        hashes = sequence_meta.block_hashes.view(np.uint64)

        # lock=True inc_ref's [0, total_hit_length) across BOTH sides; qr.finalize,
        # owned by the cache_engine layer, is the only thing that releases them.
        qr = self._tree.query(hashes, local_only=local_only, lock=True)

        # The two spans are contiguous halves of one match, not alternatives.
        total_hit = int(qr.total_hit_length)
        local_hit = min(int(qr.local_hit_length), total_hit)
        peer_rank = int(qr.remote_node_id)
        peer_hit = (total_hit - local_hit) if peer_rank >= 0 else 0
        # A peer_rank with no tail contributes nothing; treat it as local-only so
        # downstream never builds an empty peer op.
        if peer_hit <= 0:
            peer_rank = NO_PEER
            total_hit = local_hit

        if self._trace_peer:
            flexkv_logger.info(
                f"[RADIX PEER QUERY] shm={self.shm_name} "
                f"local_only={local_only} blocks={len(hashes)} "
                f"local_hit={local_hit} peer_rank={peer_rank} "
                f"peer_hit={peer_hit} total_hit={total_hit} "
                f"status={qr.status} "
                f"rdma_reads={int(qr.rdma_read_count)} "
                f"rdma_atomics={int(qr.rdma_atomic_count)}"
            )

        local_slots = np.asarray(qr.prefix_slots, dtype=np.int64)
        if len(local_slots) < local_hit:
            self._finalize_and_raise(
                qr,
                f"radixshmem returned {len(local_slots)} local slots for a "
                f"{local_hit}-block local hit"
            )

        if peer_hit <= 0:
            return ShmRadixMatch(
                num_local_blocks=local_hit,
                local_slots=local_slots[:local_hit],
                finalize=qr.finalize,
            )

        peer_slots = np.asarray(qr.remote_slots, dtype=np.int64)
        if len(peer_slots) < peer_hit:
            self._finalize_and_raise(
                qr,
                f"radixshmem returned {len(peer_slots)} remote slots for a "
                f"{peer_hit}-block peer tail (local_hit={local_hit}, "
                f"total_hit={total_hit})"
            )

        # The cluster rank IS the FlexKV node id the data path addresses. Liveness
        # is checked at the read, not here: the transfer worker validates
        # node:<id> on every get_node_meta before issuing an RDMA read.
        return ShmRadixMatch(
            num_local_blocks=local_hit,
            num_peer_blocks=peer_hit,
            local_slots=local_slots[:local_hit],
            peer_slots=peer_slots[:peer_hit],
            peer_id=peer_rank,
            finalize=qr.finalize,
        )

    @staticmethod
    def _finalize_and_raise(qr, message: str) -> None:
        """Drop the query's refs before propagating a consistency failure —
        otherwise the locked prefix stays pinned for the region's lifetime."""
        if qr.finalize is not None:
            qr.finalize()
        raise RuntimeError(message)

    # ---------- Publish (insert) ----------

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int) -> None:
        """Attach already-written slots to the tree. Call ONLY after the transfer
        into `physical_block_ids` has landed -- an entry in the tree is by
        definition complete and servable.

        `physical_block_ids[i]` holds logical block `start + i`, with
        `start = num_insert_blocks - len(physical_block_ids)`. `num_insert_blocks`
        is required and has no default: the two lengths visible here give the
        span's extent, never its POSITION, and assuming it reaches the path end
        fails silently for any caller whose window stops short. radixshmem owns
        the disagreement between `start` and its own matched prefix, so nothing is
        re-derived from a (by now stale) match result.

        Slot ownership transfers (`auto_recycle=True`); the caller must not
        recycle them again. No ref is taken on what landed -- the transfer is over,
        so the span has no reader -- and nothing is returned, since what did not
        attach was already recycled internally.
        """
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)

        slots = np.ascontiguousarray(physical_block_ids, dtype=np.int32)
        num_slots = len(slots)
        if num_slots == 0:
            return

        # Full logical path this insert reaches the end of. Overshooting the
        # sequence is clamped; undershooting the staged span trips start < 0.
        path_end = min(int(num_insert_blocks), len(hashes))
        start = path_end - num_slots
        if start < 0:
            # A caller bug, not a race: radixshmem absorbs a matched prefix that
            # moved under us. Raising keeps slot ownership with the caller.
            raise ValueError(
                f"radixshmem insert of {num_slots} slots overruns the "
                f"{path_end}-block path on {self.shm_name}"
            )

        target_hashes = hashes[:path_end]
        result = self._tree.insert(target_hashes, slots, start=start,
                                   auto_recycle=True)

        unused = len(result.unused_slots)
        landed = num_slots - unused
        if result.error != shmradix.InsertError.OK:
            flexkv_logger.warning(
                f"radixshmem insert on {self.shm_name} returned "
                f"{result.error}: {landed}/{num_slots} blocks landed at "
                f"start={start} (unused slots were auto-recycled)"
            )

        if landed <= 0:
            # Nothing new attached (concurrent writer, or pool exhausted);
            # auto_recycle already returned the slots.
            return

        if self.peer_enabled:
            # Until the RHT publication drains, this node's new blocks are
            # invisible cluster-wide. Not gated on PUT: a GET that stages a peer
            # hit inserts too.
            self._tree.flush()

        if self.event_collector is not None and result.error == shmradix.InsertError.OK:
            # `landed` is a COUNT, not a range -- unused_slots merges leading
            # redundancy with an unlanded tail. Only on the error-free path can it
            # name blocks: we always supply exactly `path_end - start` slots, so
            # insert cannot run short, leaving redundancy as the only cause and
            # putting what landed at the END of the path.
            attached_hashes = sequence_meta.block_hashes[path_end - landed : path_end]
            self.event_collector.publish_stored(
                block_hashes=attached_hashes,
                block_size=self.tokens_per_block,
                medium=_DEVICE_TYPE_NAMES[self.device_type]
            )

    # ---------- Mempool ops (take/recycle) ----------

    def take(self,
             num_required_blocks: int,
             strict: bool = True) -> np.ndarray:
        """Allocate slots, auto-evicting LRU as needed.

        No `protected_node` to accept: the prefix a caller would want pinned is
        already held by its match's query ref, which auto-evict honours. The
        returned slots are un-evictable AND un-reclaimable until `insert()` or
        `recycle()`.
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
        """Give slots back to the mempool.

        Also the only way back for slots whose transfer never completed: outside
        the tree they are unreachable by any query and invisible to eviction.
        """
        if physical_blocks is None or len(physical_blocks) == 0:
            return
        slots_i32 = np.ascontiguousarray(physical_blocks, dtype=np.int32)
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
    _tree: Any  # shmradix.RadixClient; the extension ships no stubs

    @property
    def num_total_blocks(self) -> int:
        return int(self._tree.total_blocks())

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())
