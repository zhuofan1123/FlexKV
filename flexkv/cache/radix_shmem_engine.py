# SPDX-License-Identifier: Apache-2.0
"""
RadixShmem-backed CacheEngine.

A drop-in replacement for `flexkv.cache.cache_engine.CacheEngineAccel` whose
RadixTree + slot Mempool live in POSIX shared memory (via the `shmradix`
package, https://github.com/.../radixshmem). Every DP scheduler process can
attach to the same shm region and run prefix queries / inserts in parallel,
serialised only by a process-shared rwlock.

Public surface mirrors `CacheEngineAccel` where it can — `take()` / `recycle()`
move slots, `match()` walks the tree — so `GlobalCacheEngine` can hold either
backend. Two deliberate divergences keep the radixshmem planners the only
consumers of the rest: `match()` returns a `ShmRadixMatch` (see below) rather
than a `MatchResultAccel`, and there is no `match_all` / `match_local` pair in
front of it — local-vs-cluster is one flag on one walk, so the callers that
care say `with_peer=` and the ones that don't get the cluster; and `insert()`
returns nothing rather than a node handle, because there is no handle to hand
out.

Differences from `CacheEngineAccel`:
- The slot mempool is owned by radixshmem (one mempool per shm region). The
  cache engine no longer holds its own `flexkv.cache.mempool.Mempool`. `take()`
  forwards to `tree.allocate_slots()` and `recycle()` forwards to
  `tree.recycle_slots()`.
- radixshmem exposes no node_id (nodes split on later inserts), so everything is
  addressed by the hash path + (start, length): `insert` hands nothing back and
  `take` accepts no `protected_node`, there being no handle either to hand out or
  to be handed. There is no lock/unlock pair on the write path at all: insert runs after
  the transfer, so a span reaching the tree has no reader left and nothing to be
  protected from. The one refcount FlexKV does hold is the READ side's --
  `query(lock=True)`, released through `QueryResult.finalize`.
- `evict()` is performed implicitly by radixshmem's auto-evict during
  `allocate_slots`. The standalone `evict()` API is not used by FlexKV's
  `take()` path on this backend.

Slot IDs returned by radixshmem are `int32`; FlexKV expects `int64`. We cast at
the boundary (`np.asarray(..., dtype=np.int64)`).

=========================  Insert happens AFTER transfer  ====================

radixshmem has no "ready" bit any more. A block is published by being in the
tree at all, so the order is:

    take() -> allocate slots (out of the tree, evictable by nobody)
    transfer bytes into those slots
    insert()  -> attach the slots to the tree, then flush() to publish

Consequences FlexKV has to honour:
- `insert()` must be called from the graph-completion path, not while building
  it. There is no ready flag to pass or clear -- and, for the same reason, no ref
  to take on what it published: the transfer is over, so the span has no reader.
- A transfer that fails or is cancelled leaves allocated slots attached to
  nothing. They are NOT reachable and NOT evictable, so they leak permanently
  unless the caller `recycle()`s them.
- `insert(auto_recycle=True)` hands slot ownership over: whatever it does not
  attach (redundant prefix a concurrent writer already landed, or an unlanded
  suffix on MEMPOOL_FULL) it recycles itself. The caller must not recycle the
  same slots again.

==========================  Distributed (peer) reuse  =======================

With `peer_enabled` on a clustered region (`world_size > 1`), `match()` runs a
DISTRIBUTED query: shmradix walks the local tree, then routes through the
cluster's router hash table and CONTINUES the walk on a peer's tree over RDMA.
The two spans are **spliced, not ranked** — this is a single match with a local
head and a peer tail:

    prefix_slots  covers blocks [0, local_hit_length)          -> local slots
    remote_slots  covers blocks [local_hit_length, total_hit_length)
                                                              -> peer slots

`match()` returns a `ShmRadixMatch` — this backend's own result type, not the
shared `MatchResultAccel`, whose single `matched_pos` label cannot express "local
up to here, then that peer". It keeps the two spans and their owner apart and
hands out slots by absolute block range.

`query(lock=True)` inc_ref's the whole span on both sides and
`QueryResult.finalize` drops both refs; cache_engine defers finalizers to graph
completion, so the peer cannot evict mid-PEERH2H.

Known gap: the router is probed only past the local hit, so a peer holding a
longer prefix that diverges from ours *inside* the local hit is never found.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType

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

# `ShmRadixMatch.peer_id` when the match has no peer tail. Also what shmradix
# itself reports as `remote_node_id` for a purely local query.
NO_PEER = -1


@dataclass
class ShmRadixMatch:
    """One radixshmem prefix query: a local head plus a tail on at most one peer.

    `MatchResultAccel` describes where a match lives with a single `matched_pos`
    label, which can say "local" or "remote" but not "local up to here, then that
    peer". A radixshmem query is exactly that split, so this backend returns its
    own type and the shared one keeps its meaning for every other engine.

        block index   0          num_local_blocks        num_matched_blocks
                      |  local_slots (our mempool)  |  peer_slots (peer_id's)  |

    Slot ids are per-owner, so the two arrays are not interchangeable and must
    never be concatenated without carrying the owner along. Every accessor takes
    ABSOLUTE block indices, and the peer ones reject a range that crosses the
    boundary, which is what stops a peer op from being addressed to the local
    node: the transfer worker zips `src_block_node_ids` positionally against
    `src_block_ids`, so a range off by one block reads the wrong node's memory
    rather than failing. `local_range` needs no such check — a range the head does
    not cover is not ours to begin with, so clamping it is the answer, not an
    error.

    The underlying query ran with `lock=True`, so the refs on both sides stay
    alive until `release()`. That must happen on every path — success, early
    return, cancel — or the matched prefix is pinned for the life of the region.
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

        The slice does all the bounding, in both directions, so a caller asking
        "how much of my window is already here?" needs no boundary arithmetic of
        its own: a head stopping short of `first` yields nothing, one running past
        `last` is trimmed to it, and `first >= last` names no block. Overrunning
        the head is not an error to begin with -- the blocks past it are simply
        not ours -- so there is nothing here to reject.

        Both bounds must be non-negative: a negative one would count from the end
        of the head instead of being clamped to its start. Every caller derives
        them from block indices, which start at 0.
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

    radixshmem admits a block to the tree only when it already holds real data,
    so the build-time insert the ready-bit backends do has no equivalent here --
    the insert has to run from a completion callback. That leaves the staged
    slots owned by nobody but this object in the meantime, and exactly one of its
    two exits must run or they are lost from the mempool for the life of the
    region:

      ``publish`` -- the transfer landed; hand the slots to the tree, which takes
                     ownership and internally recycles whatever did not attach
      ``abort``   -- the data never landed (cancel, or an unusable local prefix);
                     give the slots straight back to the mempool

    They are mutually exclusive and each idempotent, so the completion and cancel
    paths can both be armed without racing to a double free. Neither leaves
    anything to release afterwards -- ``publish`` takes no ref on what it
    attached, because the transfer that filled those slots is already over and
    the published span has no reader to protect it for.

    ``publish`` is only legal while the ref that keeps the local tree reaching
    the span's start is still held -- radixshmem rejects a span whose start it
    cannot reach. Pass that ref as ``holds`` and both exits drop it once they are
    done, which puts the ordering in the object that depends on it instead of in
    the order a caller happened to append two callbacks in.
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
            # Every way insert() can raise — argument validation here, argument
            # casting in the binding — fires BEFORE radixshmem takes the slots,
            # and a rejected span comes back as `error == OK` + unused_slots
            # rather than an exception. So the slots are still ours to return.
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
        """Attach to an existing radix shm region by name. The RadixServer
        owning the region must have been created elsewhere (e.g. by
        `flexkv.server.shm_radix_bootstrap.create_shm_radix_regions`).

        `peer_enabled` turns on cross-node reuse: GET matches query the whole
        cluster and a match may be spliced local-head + peer-tail. This needs no
        control-plane service of its own — prefixes are discovered over RDMA
        through the cluster's router hash table, and the cluster RANK shmradix
        reports IS the FlexKV node id the peer data path (PEERH2H / PEERSSD2H)
        addresses."""
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
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold
        self.shm_name = shm_name

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        self.peer_enabled = peer_enabled
        self._trace_peer = os.getenv("FLEXKV_TRACE_RADIX_PEER", "0") == "1"

        # RadixClient attaches to a shm region created by a RadixServer owned
        # by the bootstrap process. With peer matching on, the attach has to wait
        # out that server's bootstrap: the region exists before the cluster
        # manifest is stamped into it, and a client that gets in early is stuck
        # standalone for its whole life -- it would publish nothing to the
        # cluster router table and query only locally.
        # Only wait when a cluster is actually configured -- with world_size=1 a
        # standalone header is the final answer, not a transient state.
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

        Invalidates every slot id and match result obtained before the call, so
        it is only safe with no transfer in flight. Used by tests and by
        `_clear_cpu_cache`.
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
        """Prefix-match the sequence against the shared (and maybe peer) index.

        `with_peer=False` (or a non-distributed region) restricts the walk to the
        local tree — that is what PUT wants, since PUT only ever writes locally.
        `gpu_matched_blocks` is accepted for interface parity with the accel/hie
        engines; radixshmem always matches the full sequence.

        The result is a SINGLE spliced match — a local head and, past it, a tail
        on one peer. See `ShmRadixMatch`.
        """
        local_only = not (with_peer and self.peer_enabled)
        sequence_meta.gen_hashes()
        # SequenceMeta.block_hashes is int64; radixshmem expects uint64. They
        # share the same byte width, so view-cast is safe.
        hashes = sequence_meta.block_hashes.view(np.uint64)

        # lock=True inc_ref's [0, total_hit_length) across BOTH sides so nothing
        # is evicted before the transfer consumes it; qr.finalize, owned by the
        # cache_engine layer, is the only thing that releases them.
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

        # The cluster rank IS the FlexKV node id the data path addresses, so no
        # lookup is needed. Liveness is checked where it matters — the transfer
        # worker validates node:<id> on every get_node_meta before issuing an
        # RDMA read, far closer to the read than match time.
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
        """Attach already-written slots to the tree.

        **Call this only after the transfer into `physical_block_ids` has
        completed** — an entry in the radixshmem tree is by definition complete
        and immediately servable to this node and its peers.

        `physical_block_ids[i]` holds the KV for logical block `start + i`, with
        `start = num_insert_blocks - len(physical_block_ids)`: a staged span
        always ends at the path end. Nothing is re-derived from a match result
        here — radixshmem owns the disagreement between `start` and its own
        matched prefix (`matched < start` rejects the whole segment, `matched >
        start` drops just the leading redundancy), and a match taken when the
        plan was built is exactly the stale input that decision must not use.

        `num_insert_blocks` is REQUIRED, and deliberately has no default. The
        two lengths this method can see — the hash path's and the slot array's —
        give the path's extent and the span's, never the span's POSITION on it;
        that last bit is a fact about the transfer that just ran, so only the
        caller holds it. Defaulting to `len(hashes)` would not derive the end,
        it would assume the span reaches it, which is false for every caller
        whose window stops short (a masked PUT, an SSD-hit-bounded prefetch),
        and the assumption fails silently in both directions: `start` too far
        along is rejected wholesale while radixshmem still reports `error ==
        OK`, and a concurrent writer that extended the prefix past it turns the
        same call into a publish of the right slots under the wrong hashes.

        Slot ownership transfers to radixshmem (`auto_recycle=True`): anything
        not attached is recycled internally, so the caller must not recycle the
        same slots afterwards.

        No ref is taken on what landed, so there is nothing to release: the
        transfer that filled these slots is already over, which leaves the
        published span without a reader, and eviction reclaiming it is just the
        cache dropping a cold entry.

        Nothing comes back. How many blocks landed is a COUNT, not a range (see
        the event-collector note below), so it can feed a log or a metric but can
        never tell a caller which blocks to act on — and there is nothing left to
        act on anyway: what did not attach was recycled internally, and what did
        needs no release.
        """
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)

        slots = np.ascontiguousarray(physical_block_ids, dtype=np.int32)
        num_slots = len(slots)
        if num_slots == 0:
            return

        # Full logical path this insert reaches the end of. A caller overshooting
        # the sequence is clamped; one that undershoots the span it staged (or
        # passes the old -1) lands on start < 0 below and is told off.
        path_end = min(int(num_insert_blocks), len(hashes))
        start = path_end - num_slots
        if start < 0:
            # A caller bug, not a race: radixshmem itself absorbs a matched
            # prefix that moved under us. Raising keeps slot ownership with the
            # caller, which recycles on the way out.
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
            # Nothing new attached (a concurrent writer already published this
            # path, or a pool is exhausted). auto_recycle already returned the
            # slots, so there is nothing left to do.
            return

        if self.peer_enabled:
            # The insert's RHT publication is only routable from peers once
            # drained — and until it is, this node's new blocks are invisible
            # cluster-wide. Not gated on PUT: a GET that stages a peer hit
            # inserts too.
            self._tree.flush()

        if self.event_collector is not None and result.error == shmradix.InsertError.OK:
            # `landed` is a COUNT, not a range: radixshmem merges the leading
            # redundancy a concurrent writer caused with any tail that never
            # landed into one `unused_slots`, so on its own it cannot say WHICH
            # blocks attached. It pins one down on the error-free path only. There
            # we always supply exactly `path_end - start` slots, so insert can
            # never run short and drop a tail, which leaves leading redundancy as
            # the only way to leave a slot unused — and that puts what landed at
            # the END of the path. A pool running out mid-insert is the other way
            # to end up with a gap, and it always reports an error, so that path
            # publishes nothing rather than name blocks that may not be there.
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
        """Allocate `num_required_blocks` slots from radixshmem's mempool.

        radixshmem's `allocate_slots` will auto-evict LRU entries to satisfy the
        request. There is no `protected_node` to accept, and nothing is lost by
        that: this backend's match() hands out no node handle, and the prefix a
        caller would want pinned is already held by that match's own query ref,
        which auto-evict honours.

        The returned slots are outside the tree until `insert()` attaches them,
        which makes them un-evictable but also un-reclaimable — the caller owns
        them until it calls `insert()` or `recycle()`.
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

        Also how slots whose transfer never completed are given back: on this
        backend a failed or cancelled op leaves them attached to nothing --
        unreachable by any query and invisible to eviction -- so without this
        call they leak for the lifetime of the shm region. Attached slots need
        no such call, being evictable once they are in the tree.
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
