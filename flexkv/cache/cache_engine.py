# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
from functools import partial
from queue import Queue
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field

import numpy as np
import nvtx
import torch
from flexkv.c_ext import CRadixNode, CRadixTreeIndex, CMatchResult
from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.cache.redis_meta import RedisMeta, dist_available

from flexkv.cache.mempool import Mempool
from flexkv.cache.radix_shmem_engine import (
    ShmRadixMatch,
    StagedRadixInsert,
)
from flexkv.cache.radixtree import RadixTreeIndex, RadixNode, MatchResult
from flexkv.cache.swa_cache_engine import SWAOpConstructor
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import (
    DeviceType,
    TransferOpGraph,
    TransferOp,
    TransferType,
    add_virtual_op_for_multiple_finished_ops,
)
from flexkv.common.debug import flexkv_logger, summarize_id_tensor
from flexkv.common.type import MatchResultAccel
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics import FlexKVMetricsCollector, init_global_collector, get_global_collector

DEVICE_TYPE: List[str] = ['CPU', 'GPU', 'SSD', 'REMOTE']
_VALID_EVICTION_POLICIES = {'lru', 'lfu', 'slru', 'fifo', 'mru', 'filo'}


@dataclass
class GetTransferPlan:
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    op_callback_dict: Dict[int, Callable]
    num_gpu_blocks_to_transfer: int
    # Deferred actions run once the transfer graph completes: node unlock /
    # set_ready / publish, buffer recycle, and radixshmem source-ref release.
    on_complete: List[Callable[[], None]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "GetTransferPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            op_callback_dict={},
            num_gpu_blocks_to_transfer=0,
        )


@dataclass
class PutTransferPlan:
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    op_callback_dict: Dict[int, Callable]
    num_gpu_blocks_to_transfer: int
    skipped_gpu_blocks: int
    # Deferred actions run once the transfer graph completes: node unlock /
    # set_ready / publish, buffer recycle, and radixshmem source-ref release.
    on_complete: List[Callable[[], None]] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "PutTransferPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            op_callback_dict={},
            num_gpu_blocks_to_transfer=0,
            skipped_gpu_blocks=0,
        )


@dataclass
class SWAReadSource:
    hit_blocks: int = 0
    host_slot: int = -1
    node: Optional[object] = None
    device_type: Optional[DeviceType] = None
    engine: Optional[object] = None
    # Key-addressed REMOTE tier (mooncake-store): the tail hash of the hit
    # block is the sole remote handle — no radix node / host slot exists on
    # that tier, so pin / unlock / evict do not apply to the source.
    mooncake_tail_hash: Optional[str] = None

    @property
    def is_mooncake(self) -> bool:
        return self.mooncake_tail_hash is not None

    @property
    def found(self) -> bool:
        if self.hit_blocks <= 0 or self.device_type is None:
            return False
        if self.is_mooncake:
            return self.device_type == DeviceType.REMOTE
        return self.host_slot >= 0 and self.node is not None


@dataclass(frozen=True)
class SWAReadReservation:
    """Pinned SWA source plus any transient CPU staging slot and graph op."""
    source: SWAReadSource
    staging_slot: int
    h2d_id: int


class CacheEngineAccel:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None,
                 protected_threshold = 2,
                 swa_config: Optional["SWAPoolConfig"] = None):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks, hit_reward_seconds, eviction_policy,
                                     protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # SWA (Sliding Window Attention) — NODE-MOUNTED on the Full-KV radix
        # tree (hicache / sglang style), NOT a standalone index. The radix nodes
        # carry the SWA slot / tombstone / lock (see csrc/radix_tree.h and
        # flexkv/cache/radixtree.py); this engine only owns the SWA host-pool
        # (slot bytes + free-list) and the slot alloc/free/drain plumbing. SWA
        # and Full eviction are UNIFIED through the one tree so the two pools
        # never drift. Thisengine owns SWA initialization for its tier; init_swa()
        #  remains public for tests and explicit embedding.
        self.swa_pool = None
        tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if tier_swa_config is not None:
            self.init_swa(tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine."""
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.swa_pool is not None

    def _alloc_swa_slot(self, protected_node=None) -> int:
        """Allocate one SWA slot; evict SWA-LRU once when the pool is full."""
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        # can not allocate SWA slot, evict SWA-LRU once
        if protected_node is not None:
            self.lock_node(protected_node)
        try:
            self._evict_swa_slots(1)
        finally:
            if protected_node is not None:
                self.unlock(protected_node)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _free_swa_slot(self, slot: int) -> None:
        """Return one detached SWA slot to this tier's pool."""
        self.swa_pool.free(int(slot))

    def _drain_unmounted_swa_slots(self) -> None:
        """Return slots detached by radix-tree structural changes to the pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            self._free_swa_slot(slot)

    def _pin_swa_node(self, node) -> None:
        self.index.lock(node)
        try:
            node.inc_swa_lock_ref()
        except Exception:
            self.index.unlock(node)
            raise

    def _evict_swa_slots(self, num_swa_evicted: int) -> int:
        """Evict node-mounted SWA slots through the C++ radix tree."""
        if self.swa_pool is None:
            return 0
        evicted_full = torch.zeros(0, dtype=torch.int64)
        num_freed = self.index.evict_swa(evicted_full, num_swa_evicted)
        if evicted_full.numel() > 0:
            self.mempool.recycle_blocks(evicted_full.numpy())
        self._drain_unmounted_swa_slots()
        return num_freed

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()
        # The tree reset bulk-deletes all nodes (their SWA slots are not
        # buffered), so re-arm the SWA pool as fully free to avoid a leak.
        if self.swa_pool is not None:
            self.swa_pool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        match_result = self.index.match_prefix(torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                              sequence_meta.num_blocks, True)
        # physical blocks (torch.Tensor -> numpy, zero-copy on CPU)
        phys = match_result.physical_blocks.cpu().numpy()
        # optional block_node_ids
        try:
            bnis = getattr(match_result, "block_node_ids", None)
            if isinstance(bnis, torch.Tensor) and bnis.numel() > 0:
                bnids_np = bnis.cpu().numpy()
            else:
                bnids_np = None
        except Exception:
            bnids_np = None
        return MatchResultAccel(
            num_ready_matched_blocks=match_result.num_ready_matched_blocks,
            num_matched_blocks=match_result.num_matched_blocks,
            last_ready_node=match_result.last_ready_node,
            last_node=match_result.last_node,
            last_node_matched_length=match_result.last_node_matched_length,
            physical_blocks=phys,
            block_node_ids=bnids_np,
            matched_pos="remote" if self.device_type == DeviceType.REMOTE else "local",
            # SWA node-mount: carry the SWA hit found on the SAME forward pass so
            # the SWA-aware get can reuse it (no second match_prefix walk).
            last_swa_node=getattr(match_result, "last_swa_node", None),
            swa_hit_blocks=int(getattr(match_result, "swa_hit_blocks", 0) or 0),
        )

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Optional[CRadixNode]:
        sequence_meta.gen_hashes()
        if match_result is None:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready)
        else:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready,
                                     match_result.last_node,
                                     match_result.num_matched_blocks,
                                     match_result.last_node_matched_length)

        if self.event_collector is not None:
            self.event_collector.publish_stored(
                block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                block_size=self.tokens_per_block,
                medium=DEVICE_TYPE[self.device_type]
            )

        return node

    def lock_node(self, node: CRadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: CRadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: CRadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    def release_node(self, node: CRadixNode, ready_length: int) -> None:
        """Transfer-complete release: unlock the node and mark it ready. Paired
        with the lock_node taken at insert/match protection time."""
        self.unlock(node)
        self.set_ready(node, True, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> np.ndarray:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0  # Or minimum evict_ratio
            )

            if evict_block_num > 0:
                target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
                evicted_block_hashes = torch.zeros(evict_block_num, dtype=torch.int64)
                # evict() resizes both tensors in-place to the actual freed count
                # (which may EXCEED evict_block_num when the I2 tombstone cascade
                # frees ancestors) and returns that count. Trust it, don't assume
                # evict_block_num.
                num_evicted = self.index.evict(target_blocks, evicted_block_hashes, evict_block_num)
                if target_blocks.numel() != num_evicted:
                    target_blocks.resize_(num_evicted)
                    evicted_block_hashes.resize_(num_evicted)
                target_blocks = target_blocks.numpy()
                self.mempool.recycle_blocks(target_blocks)

                # SWA node-mount: full eviction may have connected-freed SWA
                # slots (record_freed_swa_slot in split/evict). Return them to the
                # SWA host pool so the two pools stay in lock-step (I1). No-op when
                # SWA is disabled.
                self._drain_unmounted_swa_slots()

                # Record eviction metrics
                if self._metrics_collector is not None and num_evicted > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), num_evicted)

                if self.event_collector is not None:
                    self.event_collector.publish_removed(
                        block_hashes=evicted_block_hashes.numpy(),
                        medium=DEVICE_TYPE[self.device_type]
                    )
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError(f"Not enough free blocks to take, "
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)
        self._drain_unmounted_swa_slots()

class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None,
                 protected_threshold = 2,
                 swa_config: Optional["SWAPoolConfig"] = None):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block, hit_reward_seconds=hit_reward_seconds, eviction_policy=eviction_policy,
                                       protected_threshold=protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # Legacy Python mirror. Keep the SWA helpers local to this class; the
        # C++ CacheEngineAccel path is the maintained path.
        self.swa_pool = None
        self.tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if self.tier_swa_config is not None:
            self.init_swa(self.tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine."""
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.tier_swa_config is not None and self.tier_swa_config.enabled \
               and self.swa_pool is not None

    def _alloc_swa_slot(self, protected_node=None) -> int:
        """Allocate one SWA slot; evict SWA-LRU once when the pool is full."""
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        if protected_node is not None:
            self.lock_node(protected_node)
        try:
            self._evict_swa_slots(1)
        finally:
            if protected_node is not None:
                self.unlock(protected_node)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _free_swa_slot(self, slot: int) -> None:
        """Return one detached SWA slot to this tier's pool."""
        self.swa_pool.free(int(slot))

    def _drain_unmounted_swa_slots(self) -> None:
        """Return slots detached by radix-tree structural changes to the pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            self._free_swa_slot(slot)

    def _pin_swa_node(self, node) -> None:
        self.index.lock(node)
        try:
            node.inc_swa_lock_ref()
        except Exception:
            self.index.unlock(node)
            raise

    def _evict_swa_slots(self, num_swa_evicted: int) -> int:
        """Evict node-mounted SWA slots through the Python radix tree."""
        if self.swa_pool is None:
            return 0
        evicted_full, num_freed = self.index.evict_swa(num_swa_evicted)
        if evicted_full.size > 0:
            self.mempool.recycle_blocks(evicted_full)
        self._drain_unmounted_swa_slots()
        return num_freed

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()
        if self.swa_pool is not None:
            self.swa_pool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResult:
        match_result = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True)
        return match_result

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResult] = None) -> Optional[RadixNode]:
        node = self.index.insert(sequence_meta,
                                 physical_block_ids,
                                 num_insert_blocks=num_insert_blocks,
                                 is_ready=is_ready,
                                 match_result=match_result)
        if self.event_collector is not None:
            self.event_collector.publish_stored(block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                                                block_size=self.tokens_per_block,
                                                medium=DEVICE_TYPE[self.device_type])
        return node

    def lock_node(self, node: RadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: RadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: RadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[RadixNode] = None,
             strict: bool = True) -> np.ndarray:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0  # Or minimum evict_ratio
            )
            if evict_block_num > 0:
                evicted_blocks, evicted_block_hashes = self.index.evict(evict_block_num)
                self.mempool.recycle_blocks(evicted_blocks)

                # SWA node-mount: return connected-freed SWA slots to the pool (I1).
                self._drain_unmounted_swa_slots()

                # Record eviction metrics
                if self._metrics_collector is not None and len(evicted_blocks) > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), len(evicted_blocks))

                if self.event_collector is not None:
                    self.event_collector.publish_removed(block_hashes=evicted_block_hashes,
                                                         medium=DEVICE_TYPE[self.device_type])
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError("Not enough free blocks to take, ",
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)
        self._drain_unmounted_swa_slots()

@dataclass
class CacheStrategy:
    # if True, will not put or get blocks from GPU
    ignore_gpu: bool = False
    # if True, will not put or get blocks from SSD
    ignore_ssd: bool = False
    # if True, will not get blocks from REMOTE
    ignore_remote: bool = False
    # if True, will not use GDS
    ignore_gds: bool = False

DEFAULT_CACHE_STRATEGY = CacheStrategy()

CPUONLY_CACHE_STRATEGY = CacheStrategy(ignore_gpu=False, ignore_ssd=True, ignore_remote=True, ignore_gds=True)


@dataclass(frozen=True)
class _ShmGetSpan:
    """Blocks ``[start, end)`` of a radixshmem GET window, and where to read them.

    ``src_block_ids`` is already resolved: a local head and a peer tail count from
    different origins inside their match, and rebasing happens once, here, at
    construction. Everything downstream only ever talks in absolute blocks.

    ``transfer_type`` is None for the local CPU head -- those blocks are already
    in host memory, so they are read where they lie rather than copied anywhere.
    """
    tier: str  # metrics label for the tier that served the blocks
    transfer_type: Optional[TransferType]
    start: int
    end: int
    src_block_ids: np.ndarray
    src_block_node_ids: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.end - self.start

    @property
    def needs_staging(self) -> bool:
        """True when these blocks have to be copied into local slots first."""
        return self.transfer_type is not None


def _shm_get_spans(cpu_match: ShmRadixMatch,
                   ssd_match: ShmRadixMatch,
                   lo: int,
                   hi: int) -> List[_ShmGetSpan]:
    """Cut the GET window ``[lo, hi)`` into one span per source, nearest first.

        lo            cpu local     cpu matched   ssd local     ssd matched   hi
        |  local CPU  |  peer CPU   |  local SSD  |  peer SSD   |   (miss)    |
        | read where  |   PEERH2H   |   DISK2H    |  PEERSSD2H  |
        | it lies     +------------- staged in fresh local slots -------------+
        +------------------------------ one H2D ------------------------------+

    A source can only ever EXTEND the coverage of the ones ahead of it, so the
    layout is just a cursor walking from ``lo`` towards ``hi`` and it takes one
    clamp and one test to place every boundary: ``min(limit, hi)`` trims a match
    that overruns the window, and ``end > cursor`` is what makes a tier claim only
    ground no nearer tier already covers -- a tier that matched LESS than an
    earlier one reaches no further than the cursor and so contributes nothing.

    Empty spans are dropped, so a purely local hit comes back as a single span and
    no caller needs a per-tier special case. The last span's ``end`` is the end of
    the whole hit, and the spans that need staging are always its tail -- which is
    what lets one contiguous allocation back all of them and one H2D read the lot.
    """
    # (tier, transfer type, first block NOT covered, slots, peer node ids).
    sources = (
        ("cpu", None, cpu_match.num_local_blocks,
         cpu_match.local_range, None),
        ("cpu", TransferType.PEERH2H, cpu_match.num_matched_blocks,
         cpu_match.peer_range, cpu_match.peer_node_ids),
        ("ssd", TransferType.DISK2H, ssd_match.num_local_blocks,
         ssd_match.local_range, None),
        ("ssd", TransferType.PEERSSD2H, ssd_match.num_matched_blocks,
         ssd_match.peer_range, ssd_match.peer_node_ids),
    )
    spans: List[_ShmGetSpan] = []
    cursor = lo
    for tier, transfer_type, limit, slots, node_ids in sources:
        end = min(limit, hi)
        if end > cursor:
            spans.append(_ShmGetSpan(
                tier, transfer_type, cursor, end,
                src_block_ids=slots(cursor, end),
                src_block_node_ids=None if node_ids is None else node_ids(cursor, end)))
            cursor = end
    return spans


class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig, redis_meta: RedisMeta = None,
                 event_collector: Optional[KVEventCollector] = None):
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None
        self.use_mooncake_store_backend = cache_config.use_mooncake_store_backend

        self.index_accel = GLOBAL_CONFIG_FROM_ENV.index_accel
        # When True, replace per-device CacheEngine{,Accel} with the radixshmem-
        # backed engine so multiple DP processes share a single index in shm.
        self.use_radix_shmem = bool(getattr(GLOBAL_CONFIG_FROM_ENV, "radix_shmem", False))
        self._shm_radix_server_id = getattr(
            GLOBAL_CONFIG_FROM_ENV, "shm_radix_server_id", "default"
        )
        if cache_config.enable_kv_sharing:
            assert redis_meta is not None
            self.redis_meta = redis_meta
            self.node_id = self.redis_meta.get_node_id()
            self.enable_kv_sharing = True
        else:
            self.enable_kv_sharing = False
        self.cache_engines = {}

        self.evict_ratio = GLOBAL_CONFIG_FROM_ENV.evict_ratio
        self.evict_start_threshold = GLOBAL_CONFIG_FROM_ENV.evict_start_threshold
        self.hit_reward_seconds = GLOBAL_CONFIG_FROM_ENV.hit_reward_seconds
        self.eviction_policy = GLOBAL_CONFIG_FROM_ENV.eviction_policy
        self.protected_threshold = GLOBAL_CONFIG_FROM_ENV.slru_protected_threshold

        # Initialize metrics collector for cache engine monitoring (before creating CacheEngines)
        self._metrics_collector = get_global_collector()
        if self._metrics_collector is None:
            self._metrics_collector = init_global_collector()

        need_dist = (
            (cache_config.enable_cpu and cache_config.enable_p2p_cpu)
            or (cache_config.enable_ssd and cache_config.enable_p2p_ssd)
            or (cache_config.enable_remote and cache_config.enable_kv_sharing)
        )
        if need_dist and not dist_available():
            raise RuntimeError(
                "Config enables distributed KV cache (P2P/Redis), but FlexKV was built without it. "
                "Rebuild with FLEXKV_ENABLE_P2P=1 and install Redis dependencies "
                "(e.g. libhiredis-dev, redis-tools). See README for full list."
            )

        if cache_config.enable_cpu:
            # radix_shmem owns the index outright — including the distributed
            # (peer) case, where the shared radix tree replaces the Redis-backed
            # HierarchyLRCacheEngine rather than layering on top of it. So it must
            # be checked BEFORE enable_p2p_cpu, which both backends read as
            # "peer reuse is on".
            if self.use_radix_shmem:
                self.cpu_cache_engine = self._build_radix_shmem_engine(
                    DeviceType.CPU, cache_config.num_cpu_blocks, event_collector,
                    peer_enabled=cache_config.enable_p2p_cpu,
                )
            elif cache_config.enable_p2p_cpu:
                self.cpu_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.CPU, meta=self.redis_meta) #TODO
            elif self.index_accel:
                self.cpu_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.cpu_cache_engine = CacheEngine(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
        if cache_config.enable_ssd:
            # radix_shmem owns the index outright — including the distributed
            # (peer) case, where the shared radix tree replaces the Redis-backed
            # HierarchyLRCacheEngine rather than layering on top of it. So it must
            # be checked BEFORE enable_p2p_ssd, which both backends read as
            # "peer reuse is on".
            if self.use_radix_shmem:
                self.ssd_cache_engine = self._build_radix_shmem_engine(
                    DeviceType.SSD, cache_config.num_ssd_blocks, event_collector,
                    peer_enabled=cache_config.enable_p2p_ssd,
                )
            elif cache_config.enable_p2p_ssd:
                self.ssd_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.SSD, meta=self.redis_meta) #TODO
            elif self.index_accel:
                self.ssd_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.ssd_cache_engine = CacheEngine(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine
        if cache_config.enable_remote:
            if self.use_mooncake_store_backend:
                from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine
                self.remote_cache_engine = MooncakeStoreCacheEngine(
                    cache_config=cache_config,
                )
            elif cache_config.enable_kv_sharing:
                # Build PCFSCacheEngine from CacheConfig directly (replacing RemotePCFSCacheEngine) TODO
                self.remote_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.REMOTE, meta=self.redis_meta)
            elif self.use_radix_shmem:
                self.remote_cache_engine = self._build_radix_shmem_engine(
                    DeviceType.REMOTE, cache_config.num_remote_blocks, None
                )
            elif self.index_accel:
                self.remote_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.REMOTE,
                    num_total_blocks=cache_config.num_remote_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.remote_cache_engine = CacheEngine(
                    device_type=DeviceType.REMOTE,
                    num_total_blocks=cache_config.num_remote_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.REMOTE] = self.remote_cache_engine

        # SWA peer-op builder. Per-tier match/slot resolution is fused into the
        # Full-KV get/put implementations; this helper only appends SWA ops.
        self.swa_op_constructor = SWAOpConstructor(self)

        #TODO move this to kvmanager.start()
        self.start()

        # Update initial mempool stats
        self._update_mempool_metrics()

    def _build_radix_shmem_engine(self,
                                   device_type: DeviceType,
                                   num_blocks: int,
                                   event_collector,
                                   peer_enabled: bool = False) -> "object":
        """Attach to a pre-created radixshmem region as a RadixClient.

        The shm region itself (RadixServer) is owned by the KVManager bootstrap
        process via `flexkv.server.shm_radix_bootstrap.create_shm_radix_regions`.
        Non-bootstrap procs poll for region availability before reaching this
        point, so the attach is unconditional here.
        """
        from flexkv.cache.radix_shmem_engine import CacheEngineRadixShmem
        from flexkv.server.shm_radix_bootstrap import shm_name_for

        return CacheEngineRadixShmem(
            device_type=device_type,
            num_total_blocks=num_blocks,
            tokens_per_block=self.cache_config.tokens_per_block,
            shm_name=shm_name_for(
                device_type,
                self._shm_radix_server_id,
                rank=getattr(GLOBAL_CONFIG_FROM_ENV, "radix_rank", 0),
                world_size=getattr(
                    GLOBAL_CONFIG_FROM_ENV, "radix_world_size", 1
                ),
            ),
            evict_ratio=self.evict_ratio,
            evict_start_threshold=self.evict_start_threshold,
            hit_reward_seconds=self.hit_reward_seconds,
            eviction_policy=self.eviction_policy,
            event_collector=event_collector,
            metrics_collector=self._metrics_collector,
            protected_threshold=self.protected_threshold,
            peer_enabled=peer_enabled,
        )

    def start(self) -> None:
        if self.cpu_cache_engine and self.cache_config.enable_p2p_cpu:
            self.cpu_cache_engine.start()
        if self.ssd_cache_engine and self.cache_config.enable_p2p_ssd:
            self.ssd_cache_engine.start()
        if self.remote_cache_engine and self.cache_config.enable_3rd_remote:
            self.remote_cache_engine.start()

    def reset(self) -> None:
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.remote_cache_engine:
            self.remote_cache_engine.reset()

    def _update_mempool_metrics(self) -> None:
        """Update memory pool metrics for all cache engines."""
        if self._metrics_collector is None:
            return
        for device_type, engine in self.cache_engines.items():
            if hasattr(engine, 'mempool'):
                device_label = DEVICE_TYPE[device_type].lower()
                self._metrics_collector.update_mempool_stats(
                    device_label,
                    engine.mempool.num_total_blocks,
                    engine.mempool.num_free_blocks
                )

    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None,
            swa_aware: bool = False) \
                 -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        self._check_input(token_ids, token_mask, slot_mapping)

        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block

        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False

        if aligned_length == 0 or not token_mask.any():
            transfer_graph = TransferOpGraph.create_empty_graph()
            return_mask = np.zeros_like(token_mask, dtype=np.bool_)
            callback = partial(self._transfer_callback, on_complete=[])
            return transfer_graph, return_mask, callback, {}, -1

        block_start_idx, block_end_idx = self._get_block_range(token_mask)
        # block_end_idx is the block just past the LAST True in token_mask. On the
        # plain path the caller marks every non-resident token up to the aligned
        # end, so this equals aligned_length // tokens_per_block. On the SWA-aware
        # path (swa_aware=True) _get_impl_* clamps the window to usable = min(full,
        # swa) after matching, which can end before the aligned length. So the
        # invariant is <= (can never exceed the aligned length), not ==. Nothing
        # below uses aligned_length; all downstream sizing keys off block_end_idx.
        assert block_end_idx <= aligned_length // self.tokens_per_block
        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        if self.use_radix_shmem:
            # Dispatched ahead of the enable_remote branch, which KVManager
            # already rules out for this backend: radixshmem needs the spliced
            # local+peer match and the insert-after-transfer order, neither of
            # which _get_impl_local can express.
            plan = self._get_impl_radixshmem(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )
        elif not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            # from this entrance, we will also handle the case of peer_cpu and peer_ssd
            plan = self._get_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )
        else:
            #TODO pcfs will be supported later
            plan = self._get_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph,
            plan.finished_ops_ids,
            dp_client_id,
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        if temp_cache_strategy.ignore_gpu and temp_cache_strategy.ignore_gds:
            # Prefetch return_mask covers full-KV tokens only. SWA REMOTE2H ops
            # (is_swa=True) live in a separate slot space and must not be summed
            # into prefetch_blocks, or the mask over-extends by SWA slot count.
            prefetch_blocks = 0
            for op in transfer_graph._op_map.values():
                if op.transfer_type == TransferType.REMOTE2H and not op.is_swa:
                    prefetch_blocks += len(op.src_block_ids)
            if prefetch_blocks > 0:
                return_mask[block_start_idx * self.tokens_per_block:
                            (block_start_idx + prefetch_blocks) * self.tokens_per_block] = True
        else:
            return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + plan.num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        # if layer_num // layer_granularity != 1:
        #     transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
        #                                                                         finished_ops_ids=finished_ops_ids,
        #                                                                         layer_num=layer_num,
        #                                                                         layer_granularity=layer_granularity)

        callback = partial(self._transfer_callback, on_complete=plan.on_complete)

        op_callback_dict = plan.op_callback_dict

        # Update mempool metrics after GET operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _build_op_callback_dict(self, op_node_to_ready: Dict) -> Dict[int, Callable]:
        op_callback_dict = {}
        for op_id, (device_type, node_to_ready, ready_length) in op_node_to_ready.items():
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=device_type,
                                              node_to_ready=node_to_ready,
                                              ready_length=ready_length)
        return op_callback_dict

    @staticmethod
    def _append_op_callback(op_callback_dict: Dict[int, Callable],
                            op_id: int,
                            callback: Callable) -> None:
        """Append ``callback`` without overwriting another completion action."""
        previous = op_callback_dict.get(op_id)
        if previous is None:
            op_callback_dict[op_id] = callback
            return

        def combined_callback() -> None:
            previous()
            callback()

        op_callback_dict[op_id] = combined_callback

    def _publish_swa_put_slot(self,
                              device_type: DeviceType,
                              node,
                              slot: int) -> None:
        """Make a reserved PUT slot readable after its tier transfer completes.

        The slot id is allocated before graph construction so the data plane can
        address it, but it is deliberately not mounted on the radix node until
        this callback.  Full-KV and SWA publication therefore remain independent:
        either transfer may finish first without exposing unfilled SWA bytes.
        """
        assert node is not None
        assert slot >= 0
        engine = self.cache_engines[device_type]
        engine.index.set_swa(node, int(slot))

    def _empty_get_return(self, request_id: int) -> GetTransferPlan:
        return GetTransferPlan.empty()

    def _empty_put_return(self, request_id: int) -> PutTransferPlan:
        return PutTransferPlan.empty()

    def _fail_put_before_insert(
            self,
            request_id: int,
            reason: str,
            cpu_blocks: np.ndarray,
            cpu_swa_slot: int = -1,
            ssd_blocks: Optional[np.ndarray] = None,
            ssd_swa_slot: int = -1,
            remote_blocks: Optional[np.ndarray] = None,
            remote_swa_slot: int = -1,
            match_finalizers: Optional[List[Callable]] = None) -> PutTransferPlan:
        flexkv_logger.warning(
            "[FlexKV-SWA] PUT request failed before radix insert; "
            f"request_id={request_id}, reason={reason}, "
            f"cpu_blocks={len(cpu_blocks)}, ssd_blocks={0 if ssd_blocks is None else len(ssd_blocks)}, "
            f"remote_blocks={0 if remote_blocks is None else len(remote_blocks)}, "
            f"cpu_swa_slot={cpu_swa_slot}, ssd_swa_slot={ssd_swa_slot}, "
            f"remote_swa_slot={remote_swa_slot}"
        )
        if cpu_swa_slot >= 0:
            self.cpu_cache_engine._free_swa_slot(cpu_swa_slot)
        if ssd_swa_slot >= 0:
            self.ssd_cache_engine._free_swa_slot(ssd_swa_slot)
        if remote_swa_slot >= 0:
            self.remote_cache_engine._free_swa_slot(remote_swa_slot)
        self.cpu_cache_engine.recycle(cpu_blocks)
        if ssd_blocks is not None:
            self.ssd_cache_engine.recycle(ssd_blocks)
        if remote_blocks is not None:
            self.remote_cache_engine.recycle(remote_blocks)
        # No transfer will consume the matched prefix; release its ref now.
        for fn in match_finalizers or []:
            fn()
        return self._empty_put_return(request_id)

    def _get_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int,
            swa_aware: bool = False) \
                 -> GetTransferPlan:
        """
        transfer pattern:

        GPU: (gpu cached) | fragment1 | fragment2      | fragment3      | (need compute)
                               ↑          ↑               ↑
        CPU:     ...      | fragment1 | fragment2(new) | fragment3(new) ← (from REMOTE)
                                          ↑               ↓
        SSD:     ...      | fragment1 | fragment2      | fragment3(new)

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd
        enable_remote = self.cache_config.enable_remote and not temp_cache_strategy.ignore_remote
        assert enable_cpu and enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None
        if self.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta)
        match_finalizers = self._collect_finalizers(
            cpu_matched_result, ssd_matched_result, remote_matched_result)
        transfer_graph = TransferOpGraph()
        swa_reservation: Optional[SWAReadReservation] = None
        if swa_aware:
            block_mask_end, swa_read_source = self._select_swa_read_source(
                block_mask_start,
                block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result,
                 DeviceType.REMOTE: remote_matched_result},
                sequence_meta=sequence_meta,
            )
            protected_cpu_node = (
                cpu_matched_result.last_ready_node
                if cpu_matched_result.num_ready_matched_blocks > block_mask_start
                else None
            )
            if enable_gpu:
                swa_reservation = self._reserve_swa_read_source(
                    transfer_graph, swa_read_source, protected_cpu_node, dp_client_id)
            if swa_read_source.found and swa_reservation is None:
                block_mask_end = block_mask_start
            if (enable_gpu and swa_read_source.found and swa_reservation is None
                    and self._metrics_collector is not None):
                self._metrics_collector.record_allocation_failure("global")
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        shared_pcfs_read = (self.cache_config.enable_kv_sharing and self.index_accel
                            and not self.use_mooncake_store_backend)
        remote_file_nodeids = None
        if shared_pcfs_read:
            remote_file_nodeids = remote_matched_result.block_node_ids
        fragment123_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks), len(remote_matched_blocks))
        #early return if no blocks to transfer
        if fragment123_num_blocks == 0:
            self._release_swa_read_reservation(swa_reservation)
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_get_return(request_id)
        assert fragment123_num_blocks <= len(gpu_block_ids)

        finished_ops_ids = []

        fragment1_num_blocks = len(cpu_matched_blocks)
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment3_num_blocks = max(len(remote_matched_blocks) - fragment12_num_blocks, 0)
        fragment23_num_blocks = fragment2_num_blocks + fragment3_num_blocks

        fragment123_gpu_blocks = gpu_block_ids[:fragment123_num_blocks]
        fragment123_cpu_blocks = cpu_matched_blocks
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment3_remote_blocks = remote_matched_blocks[-fragment3_num_blocks:]
        fragment3_remote_file_nodeids = None
        if shared_pcfs_read:
            fragment3_remote_file_nodeids = remote_file_nodeids[-fragment3_num_blocks:]
        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node
        remote_node_to_unlock = remote_matched_result.last_ready_node
        cpu_blocks_to_free = np.array([], dtype=np.int64)

        if fragment23_num_blocks > 0:
            num_extra_required_blocks = fragment23_num_blocks
            try:
                fragment23_cpu_blocks = self.cpu_cache_engine.take(
                    num_required_blocks=num_extra_required_blocks,
                    protected_node=cpu_matched_result.last_node,
                    strict=True
                )
            except RuntimeError:
                self._release_swa_read_reservation(swa_reservation)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("global")
                # No transfer will consume the matched prefix; release its ref now.
                for fn in match_finalizers or []:
                    fn()
                return self._empty_get_return(request_id)
            if len(fragment23_cpu_blocks) < num_extra_required_blocks:
                self.cpu_cache_engine.recycle(fragment23_cpu_blocks)
                self._release_swa_read_reservation(swa_reservation)
                # Record allocation failure (resource unavailable, not cache miss)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("global")
                # No transfer will consume the matched prefix; release its ref now.
                for fn in match_finalizers or []:
                    fn()
                return self._empty_get_return(request_id)
            fragment123_cpu_blocks = np.concatenate([fragment123_cpu_blocks, fragment23_cpu_blocks])
            # we only insert the buffer blocks to cpu cache engine only:
            # 1. the cpu cache engine satisfies prefix cache after insertion
            # 2. the sequence is all ready blocks
            if (cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  fragment23_cpu_blocks,
                                                                  num_insert_blocks=fragment123_num_blocks + \
                                                                    block_mask_start,
                                                                  is_ready=False,
                                                                  match_result=cpu_matched_result)
            else:
                cpu_blocks_to_free = fragment23_cpu_blocks

        # Record cache hit/miss metrics after confirming successful allocation
        if self._metrics_collector is not None:
            total_query_blocks = block_mask_end - block_mask_start
            # CPU hit blocks (directly from CPU cache)
            self._metrics_collector.record_cache_hit("cpu", fragment1_num_blocks)
            # SSD hit blocks (blocks loaded from SSD)
            self._metrics_collector.record_cache_hit("ssd", fragment2_num_blocks)
            # Remote hit blocks (blocks loaded from remote)
            self._metrics_collector.record_cache_hit("remote", fragment3_num_blocks)
            # Miss blocks (not in any cache)
            miss_blocks = total_query_blocks - fragment123_num_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        op_disk2h = None
        if fragment2_num_blocks > 0:
            op_disk2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.DISK2H,
                src_block_ids = fragment2_ssd_blocks,
                dst_block_ids = fragment123_cpu_blocks[fragment1_num_blocks:fragment12_num_blocks],
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_disk2h)

        op_remote2h = None
        if fragment3_num_blocks > 0:
            mooncake_block_hashes = None
            if self.use_mooncake_store_backend:
                mooncake_block_hashes = sequence_meta.block_hashes[
                    block_mask_start + fragment12_num_blocks:
                    block_mask_start + fragment12_num_blocks + fragment3_num_blocks
                ]
            op_remote2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.REMOTE2H,
                src_block_ids = fragment3_remote_blocks,
                dst_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                src_block_node_ids = fragment3_remote_file_nodeids,
                dp_client_id = dp_client_id,
                mooncake_store_block_hashes = mooncake_block_hashes,
            )
            transfer_graph.add_transfer_op(op_remote2h)

        # prepare ssd blocks to transfer
        write_ssd_blocks_from_remote = False
        if (enable_ssd and
            op_remote2h is not None and
            ssd_matched_result.num_ready_matched_blocks >= block_mask_start and
            ssd_matched_result.num_ready_matched_blocks == ssd_matched_result.num_matched_blocks):
            # only when the above all are satisfied, we load data back from cpu to ssd
            write_ssd_blocks_from_remote = True
            fragment3_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node=ssd_matched_result.last_node,
                strict=False
            )
            if len(fragment3_ssd_blocks) < fragment3_num_blocks:
                self.ssd_cache_engine.recycle(fragment3_ssd_blocks)
                write_ssd_blocks_from_remote = False
            if write_ssd_blocks_from_remote:
                op_h2disk = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.H2DISK,
                    src_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                    dst_block_ids = fragment3_ssd_blocks,
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_h2disk)
                transfer_graph.add_dependency(op_h2disk.op_id, op_remote2h.op_id)

                ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                                fragment3_ssd_blocks,
                                                                num_insert_blocks=fragment123_num_blocks + \
                                                                    block_mask_start,
                                                                is_ready=False,
                                                                match_result=ssd_matched_result)
        if enable_gpu:
            op_h2d = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2D,
                src_block_ids = fragment123_cpu_blocks,
                dst_block_ids = fragment123_gpu_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            if op_disk2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
            if op_remote2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_remote2h.op_id)
            finished_ops_ids.append(op_h2d.op_id)

        on_complete: List[Callable[[], None]] = []
        if cpu_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.CPU,
                                     cpu_node_to_unlock, cpu_node_to_unlock.size(), is_put=False))
        if ssd_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.SSD,
                                     ssd_node_to_unlock, ssd_node_to_unlock.size(), is_put=False))
        if remote_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.REMOTE,
                                     remote_node_to_unlock, remote_node_to_unlock.size(), is_put=False))
        recycle = self._defer_recycle(DeviceType.CPU, cpu_blocks_to_free)
        if recycle is not None:
            on_complete.append(recycle)
        on_complete.extend(match_finalizers)

        num_gpu_blocks_to_transfer = len(fragment123_gpu_blocks) if enable_gpu else 0
        op_callback_dict = {}
        if swa_reservation is not None:
            assert num_gpu_blocks_to_transfer > 0
            finished_ops_ids.append(swa_reservation.h2d_id)
            op_callback_dict[swa_reservation.h2d_id] = partial(
                self._swa_release_load_lock,
                node=swa_reservation.source.node,
                staging_slot=swa_reservation.staging_slot,
                engine=swa_reservation.source.engine,
            )

        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            op_callback_dict=op_callback_dict,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
            on_complete=on_complete,
        )

    def _get_impl_local(self,
                        request_id: int,
                        sequence_meta: SequenceMeta,
                        block_mask_start: int,
                        block_mask_end: int,
                        gpu_block_ids: np.ndarray,
                        temp_cache_strategy: CacheStrategy,
                        dp_client_id: int,
                        swa_aware: bool = False) \
                            -> GetTransferPlan:
        """
        transfer pattern:

        GPU          : (gpu cached) | fragment1 | fragment2      | (need compute)
                               ↑          ↑
        CPU(+peerCPU):     ...      | fragment1 | fragment2(new) | (uncached)
                                          ↑
        SSD(+peerSSD):     ...      | fragment1 | fragment2      | (uncached)

        """
        nvtx_range = nvtx.start_range(message=f"CacheEngine.get_impl_local[{request_id}]", color="cyan")
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta, temp_cache_strategy, is_put=False, gpu_matched_blocks=block_mask_start)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta, temp_cache_strategy)
        match_finalizers = self._collect_finalizers(cpu_matched_result, ssd_matched_result)

        transfer_graph = TransferOpGraph()
        swa_reservation: Optional[SWAReadReservation] = None
        if swa_aware:
            block_mask_end, swa_read_source = self._select_swa_read_source(
                block_mask_start,
                block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result},
                sequence_meta=sequence_meta,
            )
            protected_cpu_node = (
                cpu_matched_result.last_ready_node
                if cpu_matched_result.num_ready_matched_blocks > block_mask_start
                else None
            )
            if enable_gpu:
                swa_reservation = self._reserve_swa_read_source(
                    transfer_graph, swa_read_source, protected_cpu_node, dp_client_id)
            if swa_read_source.found and swa_reservation is None:
                block_mask_end = block_mask_start
            if (enable_gpu and swa_read_source.found and swa_reservation is None
                    and self._metrics_collector is not None):
                self._metrics_collector.record_allocation_failure("local")

        # DEBUG: Log GET operation with hash info
        #if len(sequence_meta.block_hashes) > 0:
        #    print(f"[GET {request_id}] hash[0]={sequence_meta.block_hashes[0]}, CPU={cpu_matched_result.num_matched_blocks}/{cpu_matched_result.num_ready_matched_blocks}, SSD={ssd_matched_result.num_matched_blocks}/{ssd_matched_result.num_ready_matched_blocks}, pos_CPU={cpu_matched_result.matched_pos}, pos_SSD={ssd_matched_result.matched_pos}")

        # tailor the blocks to assure:
        # the blocks are needed by the mask & the blocks are ready
        cpu_matched_blocks = cpu_matched_result.physical_blocks[:cpu_matched_result.num_ready_matched_blocks]
        cpu_matched_blocks = cpu_matched_blocks[block_mask_start:block_mask_end]
        # if ssd disabled, len(ssd_physical_blocks) is 0
        ssd_matched_blocks = ssd_matched_result.physical_blocks[:ssd_matched_result.num_ready_matched_blocks]
        ssd_matched_blocks = ssd_matched_blocks[block_mask_start:block_mask_end]

        # TODO: is this possible?
        if len(cpu_matched_blocks) > len(ssd_matched_blocks):
            ssd_matched_blocks = np.array([], dtype=np.int64)

        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment1_num_blocks = len(cpu_matched_blocks)
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        #early return if no blocks to transfer
        if fragment12_num_blocks == 0:
            self._release_swa_read_reservation(swa_reservation)
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            nvtx.end_range(nvtx_range)
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_get_return(request_id)
        assert fragment12_num_blocks <= len(gpu_block_ids)

        finished_ops_ids = []
        op_node_to_ready = {}

        fragment12_gpu_blocks = gpu_block_ids[:fragment12_num_blocks]
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment1_cpu_blocks = cpu_matched_blocks[:fragment1_num_blocks]

        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node

        # prepare cpu blocks to transfer
        cpu_blocks_to_free = np.array([], dtype=np.int64)
        op_disk2h = None
        op_gds_transfer = None
        fragment2_cpu_blocks = None

        # Allocate CPU blocks only for paths that actually stage data through
        # host memory. GDS moves fragment2 directly from SSD to GPU.
        allocated_cpu_block_num = 0 if enable_gds else fragment2_num_blocks
        # Remote CPU hits still need local CPU blocks for PEERH2H staging,
        # regardless of whether those blocks are inserted into the local index.
        if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            allocated_cpu_block_num += fragment1_num_blocks
        if allocated_cpu_block_num > 0:
            nvtx.push_range(f"take {allocated_cpu_block_num} cpu blocks", color="green")
            allocated_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=allocated_cpu_block_num,
                protected_node=cpu_matched_result.last_node,
                strict=False
            )
            nvtx.pop_range()
        else:
            # take(0) may still trigger proactive eviction at high utilization.
            allocated_cpu_blocks = np.empty(0, dtype=np.int64)
        # NOTE: not enough space to allocate, skip the request
        # there might be a better way to handle this
        if len(allocated_cpu_blocks) < allocated_cpu_block_num:
            self.cpu_cache_engine.recycle(allocated_cpu_blocks)
            self._release_swa_read_reservation(swa_reservation)
            # Record allocation failure (resource unavailable, not cache miss)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            nvtx.end_range(nvtx_range)
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_get_return(request_id)

        # Record cache hit/miss metrics after confirming successful allocation
        if self._metrics_collector is not None:
            total_query_blocks = block_mask_end - block_mask_start
            # CPU hit blocks (directly from CPU cache)
            self._metrics_collector.record_cache_hit("cpu", fragment1_num_blocks)
            # SSD hit blocks (loaded directly to GPU with GDS, otherwise via CPU)
            self._metrics_collector.record_cache_hit("ssd", fragment2_num_blocks)
            # Miss blocks (not in any cache)
            miss_blocks = total_query_blocks - fragment12_num_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            fragment1_cpu_blocks_local = allocated_cpu_blocks[-fragment1_num_blocks:]
            op_peerh2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.PEERH2H,
                src_block_ids = fragment1_cpu_blocks,
                dst_block_ids = fragment1_cpu_blocks_local,
                remote_node_ids = cpu_matched_result.matched_node_ids,
                src_block_node_ids = cpu_matched_result.matched_node_ids,  # Add this for worker
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_peerh2h)
            #TODO here we dont combine peer cpu or local cpu match results, so we can safely add remote results to local cpu
            #TODO here assume all matched blocks are ready blocks for peer cpu
            if (cpu_matched_result.insert_to_local_cpu_index and
                cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  fragment1_cpu_blocks_local,
                                                                  is_ready=False)
                # insert() returns None when nothing was attached (suffix already
                # in the shared tree) — no unready node to flip ready.
                if cpu_node_to_unlock is not None:
                    op_node_to_ready[op_peerh2h.op_id] = (
                        DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
            else:
                cpu_blocks_to_free = np.concatenate([cpu_blocks_to_free, fragment1_cpu_blocks_local])

        if fragment2_num_blocks > 0:
            if enable_gds:
                # For GDS, transfer directly from SSD to GPU using GDS transfer path (DISK2D)
                op_gds_transfer = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.DISK2D,
                    src_block_ids = fragment2_ssd_blocks,
                    dst_block_ids = fragment12_gpu_blocks[-fragment2_num_blocks:],
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_gds_transfer)
                finished_ops_ids.append(op_gds_transfer.op_id)
                if ssd_node_to_unlock is not None:
                    op_node_to_ready[op_gds_transfer.op_id] = (DeviceType.SSD,
                                                               ssd_node_to_unlock,
                                                               ssd_node_to_unlock.size())
            else:
                fragment2_cpu_blocks = allocated_cpu_blocks[:fragment2_num_blocks]

                op_disk2h = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.PEERSSD2H if ssd_matched_result.matched_pos == "remote" else TransferType.DISK2H,
                    src_block_ids = fragment2_ssd_blocks,
                    dst_block_ids = fragment2_cpu_blocks,
                    remote_node_ids = ssd_matched_result.matched_node_ids if ssd_matched_result.matched_pos == "remote" else None,
                    src_block_node_ids = ssd_matched_result.matched_node_ids if ssd_matched_result.matched_pos == "remote" else None,
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_disk2h)
                # we only insert the buffer blocks to cpu cache engine only:
                # 1. the cpu cache engine satisfies prefix cache after insertion
                # 2. the sequence is all ready blocks
                # TODO: for simplicity, if we use peer cpu results, we dont insert the buffer ssd blocks to local cpu any more
                if (cpu_matched_result.matched_pos == "local" and
                    cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                    cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                    cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                    fragment2_cpu_blocks,
                                                                    num_insert_blocks=fragment12_num_blocks + \
                                                                        block_mask_start,
                                                                    is_ready=False,
                                                                    match_result=cpu_matched_result)
                    if cpu_node_to_unlock is not None:
                        op_node_to_ready[op_disk2h.op_id] = (
                            DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
                else:
                    cpu_blocks_to_free = np.concatenate([cpu_blocks_to_free, fragment2_cpu_blocks])
        if self.cache_config.enable_p2p_cpu and cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            fragment1_cpu_blocks = fragment1_cpu_blocks_local

        if fragment2_cpu_blocks is not None:
            fragment12_cpu_blocks = np.concatenate([fragment1_cpu_blocks, fragment2_cpu_blocks])
        else:
            fragment12_cpu_blocks = fragment1_cpu_blocks

        if enable_gpu:
            op_h2d = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2D,
                src_block_ids = fragment12_cpu_blocks if not enable_gds else fragment1_cpu_blocks,
                dst_block_ids = fragment12_gpu_blocks if not enable_gds \
                    else fragment12_gpu_blocks[:fragment1_num_blocks],
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            if op_disk2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
            if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
                transfer_graph.add_dependency(op_h2d.op_id, op_peerh2h.op_id)
            finished_ops_ids.append(op_h2d.op_id)

        on_complete: List[Callable[[], None]] = []
        if cpu_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.CPU,
                                     cpu_node_to_unlock, cpu_node_to_unlock.size(), is_put=False))
        if ssd_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.SSD,
                                     ssd_node_to_unlock, ssd_node_to_unlock.size(), is_put=False))
        recycle = self._defer_recycle(DeviceType.CPU, cpu_blocks_to_free)
        if recycle is not None:
            on_complete.append(recycle)
        on_complete.extend(match_finalizers)
        num_gpu_blocks_to_transfer = len(fragment12_gpu_blocks) if enable_gpu else 0
        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)

        if swa_reservation is not None:
            assert num_gpu_blocks_to_transfer > 0
            finished_ops_ids.append(swa_reservation.h2d_id)
            op_callback_dict[swa_reservation.h2d_id] = partial(
                self._swa_release_load_lock,
                node=swa_reservation.source.node,
                staging_slot=swa_reservation.staging_slot,
                engine=swa_reservation.source.engine,
            )
        nvtx.end_range(nvtx_range)
        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            op_callback_dict=op_callback_dict,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
            on_complete=on_complete,
        )

    def _assert_radixshmem_no_swa(self, phase: str) -> None:
        """radixshmem has no node-mounted SWA slot, so SWA cannot be planned.

        CacheEngineRadixShmem exposes no ``_alloc_swa_slot`` and its match never
        fills ``last_swa_node``: a window written by PUT would be invisible to
        every GET. Fail loudly instead of silently dropping the window.
        """
        if self.swa_op_constructor.enabled:
            raise NotImplementedError(
                f"{phase} on the radixshmem backend does not support SWA "
                f"(CacheEngineRadixShmem mounts no SWA slot on a tree node)"
            )

    def _match_radixshmem(self,
                          sequence_meta: SequenceMeta,
                          temp_cache_strategy: CacheStrategy,
                          is_get: bool) \
                              -> Tuple[ShmRadixMatch, ShmRadixMatch]:
        """CPU + SSD matches for the radixshmem planners.

        Not ``match_local_accel``: that one returns ``MatchResultAccel``, whose
        single ``matched_pos`` cannot carry a local-head/peer-tail splice, and it
        picks local-vs-cluster off ``enable_p2p_*``. Here the choice is structural
        — a GET always asks the cluster (``match`` collapses to a local walk by
        itself when the region has no peers), and a PUT is local-only because
        ``transfer_engine`` routes no write into a peer's slots. ``is_get`` IS
        ``with_peer``, which is why it is phrased from the GET side.

        A tier that is absent or excluded by the strategy yields an empty match,
        so callers can read the pair unconditionally.
        """
        cpu_match = ShmRadixMatch()
        ssd_match = ShmRadixMatch()
        if self.cpu_cache_engine is not None:
            cpu_match = self.cpu_cache_engine.match(sequence_meta, with_peer=is_get)
        if self.ssd_cache_engine is not None and not temp_cache_strategy.ignore_ssd:
            ssd_match = self.ssd_cache_engine.match(sequence_meta, with_peer=is_get)
        return cpu_match, ssd_match

    def _get_impl_radixshmem(self,
                             request_id: int,
                             sequence_meta: SequenceMeta,
                             block_mask_start: int,
                             block_mask_end: int,
                             gpu_block_ids: np.ndarray,
                             temp_cache_strategy: CacheStrategy,
                             dp_client_id: int,
                             swa_aware: bool = False) \
                                 -> GetTransferPlan:
        """GET planner for the radixshmem CPU+SSD tiers.

        Two properties of the shmradix API make this a separate planner rather
        than a branch inside ``_get_impl_local``:

        * A match is SPLICED, not "local OR peer". ``query()`` walks the local
          tree and continues onto one peer's tree, and ``ShmRadixMatch`` keeps
          both halves. So the local head can go straight to GPU and only the peer
          tail needs host staging, where ``_get_impl_local`` pushes the entire CPU
          fragment through PEERH2H the moment a peer wins.
        * Slots join the tree only once they hold data, so the build-time insert
          ``_get_impl_local`` does (ready bit off, flipped on completion) has no
          equivalent here. This planner does not insert at all -- a GET promotes
          nothing -- which leaves the slots reachable by nobody, so they have to
          be handed back explicitly on every exit.

        ``_shm_get_spans`` owns the whole block-range layout, so what is left here
        is one allocation, one op per span, and the cleanups. Every range below is
        an absolute block index.
        """
        nvtx_range = nvtx.start_range(
            message=f"CacheEngine.get_impl_radixshmem[{request_id}]", color="cyan")
        enable_gpu = not temp_cache_strategy.ignore_gpu
        assert self.cache_config.enable_cpu
        assert self.cpu_cache_engine is not None
        self._assert_radixshmem_no_swa("GET")

        cpu_match, ssd_match = self._match_radixshmem(
            sequence_meta, temp_cache_strategy, is_get=True)

        def _release_match() -> GetTransferPlan:
            # Nothing will consume the matched prefix; drop the query's ref now.
            cpu_match.release()
            ssd_match.release()
            if self._metrics_collector is not None and block_mask_end > block_mask_start:
                self._metrics_collector.record_cache_miss(
                    block_mask_end - block_mask_start)
            nvtx.end_range(nvtx_range)
            return self._empty_get_return(request_id)

        spans = _shm_get_spans(cpu_match, ssd_match,
                               block_mask_start, block_mask_end)
        num_staged = sum(len(span) for span in spans if span.needs_staging)

        staging = np.empty(0, dtype=np.int64)
        if num_staged > 0:
            staging = self.cpu_cache_engine.take(num_required_blocks=num_staged,
                                                 strict=False)
            if len(staging) < num_staged:
                # Out of staging room. The local CPU head needs no staging at all,
                # so degrade to serving just that instead of dropping the whole
                # request. Dropping the staged spans re-derives every boundary.
                self.cpu_cache_engine.recycle(staging)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("local")
                spans = [span for span in spans if not span.needs_staging]
                served = spans[-1].end if spans else block_mask_start
                flexkv_logger.warning(
                    f"radixshmem GET {request_id}: only {len(staging)}/{num_staged} "
                    f"staging blocks available; serving the local CPU prefix "
                    f"[{block_mask_start}, {served}) only"
                )
                staging = np.empty(0, dtype=np.int64)

        if not spans:
            return _release_match()
        end = spans[-1].end

        if self._metrics_collector is not None:
            cpu_blocks = sum(len(span) for span in spans if span.tier == "cpu")
            self._metrics_collector.record_cache_hit("cpu", cpu_blocks)
            self._metrics_collector.record_cache_hit(
                "ssd", end - block_mask_start - cpu_blocks)
            if block_mask_end > end:
                self._metrics_collector.record_cache_miss(block_mask_end - end)

        transfer_graph = TransferOpGraph()
        staging_ops_ids: List[int] = []
        h2d_src_block_ids: List[np.ndarray] = []
        staged = 0
        for span in spans:
            if not span.needs_staging:
                # Already in host memory, so H2D reads it straight from the match.
                h2d_src_block_ids.append(span.src_block_ids)
                continue
            # The staged spans are the tail of the window and come in order, so a
            # running offset walks down the single allocation backing them all.
            dst_block_ids = staging[staged:staged + len(span)]
            staged += len(span)
            h2d_src_block_ids.append(dst_block_ids)
            op = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=span.transfer_type,
                src_block_ids=span.src_block_ids,
                dst_block_ids=dst_block_ids,
                src_block_node_ids=span.src_block_node_ids,
                dp_client_id=dp_client_id,
            )
            transfer_graph.add_transfer_op(op)
            staging_ops_ids.append(op.op_id)

        if enable_gpu:
            # One H2D for the whole hit: the staging is contiguous and sits right
            # behind the local head, so the sources concatenate in span order.
            op_h2d = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=TransferType.H2D,
                src_block_ids=np.concatenate(h2d_src_block_ids),
                dst_block_ids=gpu_block_ids[:end - block_mask_start],
                dp_client_id=dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            for op_id in staging_ops_ids:
                transfer_graph.add_dependency(op_h2d.op_id, op_id)
            finished_ops_ids = [op_h2d.op_id]
        else:
            # No H2D to hang the contract on (prefetch-style GET): the staging
            # ops are the terminals.
            finished_ops_ids = list(staging_ops_ids)

        staging_returned = False

        def _return_staging() -> None:
            # No promotion: what a GET reads off a peer (or off local disk) is not
            # published into the local CPU tree. Nothing is lost by that -- the
            # request's own PUT stores the sequence when it finishes, which is what
            # makes the prefix locally reusable. So the staging is scratch for the
            # H2D above, which has read it by the time the graph completes.
            nonlocal staging_returned
            if staging_returned:
                return
            staging_returned = True
            assert self.cpu_cache_engine is not None
            self.cpu_cache_engine.recycle(staging)

        # Each cleanup has to run whether the graph completes or is cancelled: a
        # cancelled graph never reaches on_complete, and both resources are
        # invisible to reclamation until released -- a slot outside the tree to
        # eviction, and the matched prefix stays pinned for the life of the
        # region. Both are idempotent, so arming the two paths costs nothing.
        cleanups: List[Callable[[], None]] = []
        if len(staging) > 0:
            cleanups.append(_return_staging)
        cleanups += [cpu_match.release, ssd_match.release]
        for cleanup in cleanups:
            transfer_graph.add_cancel_cleanup(cleanup)

        nvtx.end_range(nvtx_range)
        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            op_callback_dict={},
            num_gpu_blocks_to_transfer=(end - block_mask_start) if enable_gpu else 0,
            on_complete=cleanups,
        )

    def put(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        self._check_input(token_ids, token_mask, slot_mapping)
        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False
        block_start_idx, block_end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert block_start_idx == 0

        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        assert not temp_cache_strategy.ignore_gpu
        if self.use_radix_shmem:
            # See the matching branch in get(): insert-after-transfer, and
            # enable_remote is already excluded for this backend.
            plan = self._put_impl_radixshmem(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )
        elif not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            plan = self._put_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )
        else:
            plan = self._put_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph,
            plan.finished_ops_ids,
            dp_client_id,
        )
        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[(block_start_idx + plan.skipped_gpu_blocks)* self.tokens_per_block:
                    (block_start_idx + plan.skipped_gpu_blocks + plan.num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        callback = partial(self._transfer_callback, on_complete=plan.on_complete)

        op_callback_dict = plan.op_callback_dict

        # Update mempool metrics after PUT operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _put_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> PutTransferPlan:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                               ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                ↓
        SSD:          (ssd cached)         | fragment2(new) |

        CPU:            ...           |     fragment3      |
                                               ↓ (from cpu)
        REMOTE:     (remote cached)   |   fragment3(new)   |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_remote = self.cache_config.enable_remote and not temp_cache_strategy.ignore_remote
        assert enable_gpu
        assert enable_cpu
        assert enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta,
                                                                                               temp_cache_strategy=temp_cache_strategy,
                                                                                               is_get=False)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta,
                                                                                           temp_cache_strategy=temp_cache_strategy)
        match_finalizers = self._collect_finalizers(
            cpu_matched_result, ssd_matched_result, remote_matched_result)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0

        # NOTE: to avoid full kv repeating write in mooncake store.
        if self.use_mooncake_store_backend:
            kv_hit = int(getattr(remote_matched_result, "kv_matched_blocks", 0)
                         or remote_matched_result.num_matched_blocks)
            remote_put_hit_blocks = max(0, min(len(gpu_block_ids), kv_hit - block_mask_start))
            fragment3_num_blocks = len(gpu_block_ids) - remote_put_hit_blocks
        else:
            remote_put_hit_blocks = len(remote_matched_blocks)
            fragment3_num_blocks = len(gpu_block_ids) - len(remote_matched_blocks)

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )
        if len(fragment12_cpu_blocks) < fragment12_num_blocks:
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_put_return(request_id)
        put_to_ssd = False
        if enable_ssd and fragment2_num_blocks > 0:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
            if len(fragment2_ssd_blocks) == fragment2_num_blocks:
                put_to_ssd = True
            else:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)
        put_to_remote = False
        if fragment3_num_blocks > 0:
            fragment3_remote_blocks = self.remote_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node = remote_matched_result.last_node,
                strict=False
            )
            if len(fragment3_remote_blocks) == fragment3_num_blocks:
                put_to_remote = True
            else:
                self.remote_cache_engine.recycle(fragment3_remote_blocks)
        else:
            fragment3_remote_blocks = np.array([], dtype=np.int64)

        cpu_swa_slot = -1
        ssd_swa_slot = -1
        remote_swa_slot = -1
        mooncake_swa_tail_hash: Optional[str] = None

        if self.swa_op_constructor.enabled:
            cpu_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                cpu_matched_result.last_node)
            if cpu_swa_slot >= 0 and put_to_ssd:
                ssd_swa_slot = self.ssd_cache_engine._alloc_swa_slot(
                    ssd_matched_result.last_node)
            if (cpu_swa_slot >= 0 and
                    (not put_to_ssd or ssd_swa_slot >= 0) and
                    put_to_remote):
                if self.use_mooncake_store_backend:
                    # Key-addressed store: no remote slot to reserve / mount.
                    # SWA snapshot keyed by the tail hash of the written prefix.
                    tail_idx = block_mask_start + len(gpu_block_ids) - 1
                    mooncake_swa_tail_hash = str(
                        sequence_meta.block_hashes[tail_idx])
                else:
                    remote_swa_slot = self.remote_cache_engine._alloc_swa_slot(
                        remote_matched_result.last_node)
            if (cpu_swa_slot < 0 or
                    (put_to_ssd and ssd_swa_slot < 0) or
                    (put_to_remote and remote_swa_slot < 0
                     and mooncake_swa_tail_hash is None)):
                return self._fail_put_before_insert(
                    request_id=request_id,
                    reason="swa_slot_alloc_failed",
                    cpu_blocks=fragment12_cpu_blocks,
                    cpu_swa_slot=cpu_swa_slot,
                    ssd_blocks=fragment2_ssd_blocks if put_to_ssd else None,
                    ssd_swa_slot=ssd_swa_slot,
                    remote_blocks=fragment3_remote_blocks if put_to_remote else None,
                    remote_swa_slot=remote_swa_slot,
                    match_finalizers=match_finalizers,
                )

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            dp_client_id = dp_client_id,
        )
        flexkv_logger.info(
            "[FlexKV-SEGV-DEBUG] cache_engine create D2H op (global_put) "
            f"request_id={request_id}, op_id={op_d2h.op_id}, "
            f"graph_id={transfer_graph.graph_id}, dp_client_id={dp_client_id}, "
            f"fragment12_num_blocks={fragment12_num_blocks}, "
            f"fragment2_num_blocks={fragment2_num_blocks}, "
            f"fragment3_num_blocks={fragment3_num_blocks}, "
            f"{summarize_id_tensor('gpu_src', fragment12_gpu_blocks)}, "
            f"{summarize_id_tensor('cpu_dst', fragment12_cpu_blocks)}"
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if put_to_ssd:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        if put_to_remote:
            if fragment3_num_blocks > fragment12_num_blocks:
                extra_num_cpu_blocks = fragment3_num_blocks - fragment12_num_blocks
                fragment3_cpu_blocks = np.concatenate([cpu_matched_blocks[-extra_num_cpu_blocks:],
                                                       fragment12_cpu_blocks])
            else:
                fragment3_cpu_blocks = fragment12_cpu_blocks[-fragment3_num_blocks:]
            mooncake_block_hashes = None
            if self.use_mooncake_store_backend:
                mooncake_block_hashes = sequence_meta.block_hashes[
                    block_mask_start + remote_put_hit_blocks:
                    block_mask_start + remote_put_hit_blocks + fragment3_num_blocks
                ]
            op_h2remote = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2REMOTE,
                src_block_ids = fragment3_cpu_blocks,
                dst_block_ids = fragment3_remote_blocks,
                dp_client_id = dp_client_id,
                mooncake_store_block_hashes = mooncake_block_hashes,
            )
            transfer_graph.add_transfer_op(op_h2remote)
            transfer_graph.add_dependency(op_h2remote.op_id, op_d2h.op_id)

        if cpu_swa_slot >= 0:
            empty = np.array([], dtype=np.int64)
            put_remote_via_mooncake = (
                mooncake_swa_tail_hash is not None and cpu_swa_slot >= 0)
            if remote_swa_slot >= 0:
                remote_slot_ids = np.array([remote_swa_slot], dtype=np.int64)
            elif put_remote_via_mooncake:
                remote_slot_ids = np.array([0], dtype=np.int64) # slot 0 will not be used in mooncake store backend.
            else:
                remote_slot_ids = empty
            swa_ops = self.swa_op_constructor.build_put_chain(
                transfer_graph,
                gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
                cpu_slot_ids=np.array([cpu_swa_slot], dtype=np.int64),
                ssd_slot_ids=(np.array([ssd_swa_slot], dtype=np.int64)
                              if ssd_swa_slot >= 0 else empty),
                remote_slot_ids=remote_slot_ids,
                dp_client_id=dp_client_id,
                return_op_ids=True,
                mooncake_tail_hashes=(
                    [mooncake_swa_tail_hash] if put_remote_via_mooncake else None),
            )
            assert swa_ops.d2h_id is not None
            if put_to_ssd:
                assert swa_ops.h2disk_id is not None
            if put_to_remote and (remote_swa_slot >= 0 or put_remote_via_mooncake):
                assert swa_ops.h2remote_id is not None
            finished_ops_ids.append(swa_ops.d2h_id)

        cpu_node_to_unlock = self.cpu_cache_engine.insert(
            sequence_meta,
            fragment12_cpu_blocks,
            is_ready=False,
            match_result=cpu_matched_result,
        )
        # insert() returns None when nothing was attached (the whole suffix was
        # already present in the shared tree) — then there is no unready node to
        # flip ready after the transfer, so skip the ready-callback bookkeeping.
        if cpu_node_to_unlock is not None:
            op_node_to_ready[op_d2h.op_id] = (
                DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if put_to_ssd:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(
                sequence_meta,
                fragment2_ssd_blocks,
                is_ready=False,
                match_result=ssd_matched_result,
            )
            if ssd_node_to_unlock is not None:
                op_node_to_ready[op_h2disk.op_id] = (
                    DeviceType.SSD, ssd_node_to_unlock, ssd_node_to_unlock.size())
        remote_node_to_unlock = None
        if put_to_remote:
            remote_node_to_unlock = self.remote_cache_engine.insert(
                sequence_meta,
                fragment3_remote_blocks,
                is_ready=False,
                match_result=remote_matched_result,
            )
            if remote_node_to_unlock is not None:
                op_node_to_ready[op_h2remote.op_id] = (
                    DeviceType.REMOTE,
                    remote_node_to_unlock,
                    remote_node_to_unlock.size(),
                )
        on_complete: List[Callable[[], None]] = []
        if cpu_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.CPU,
                                     cpu_node_to_unlock, cpu_node_to_unlock.size(), is_put=True))
        if ssd_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.SSD,
                                     ssd_node_to_unlock, ssd_node_to_unlock.size(), is_put=True))
        if remote_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.REMOTE,
                                     remote_node_to_unlock, remote_node_to_unlock.size(), is_put=True))
        on_complete.extend(match_finalizers)

        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)
        if cpu_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.d2h_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.CPU, cpu_node_to_unlock, cpu_swa_slot),
            )
        if ssd_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2disk_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.SSD, ssd_node_to_unlock, ssd_swa_slot),
            )
        if remote_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2remote_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.REMOTE, remote_node_to_unlock, remote_swa_slot),
            )
        skipped_gpu_blocks = len(cpu_matched_blocks)
        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            op_callback_dict=op_callback_dict,
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
            skipped_gpu_blocks=skipped_gpu_blocks,
            on_complete=on_complete,
        )

    def _put_impl_local(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> PutTransferPlan:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                                ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                 ↓
        SSD:          (ssd cached)         | fragment2(new) |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_gpu
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta,
                                                                            temp_cache_strategy=temp_cache_strategy,
                                                                            is_put=True)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta,
                                                                      temp_cache_strategy=temp_cache_strategy,
                                                                      is_put=True)
        match_finalizers = self._collect_finalizers(cpu_matched_result, ssd_matched_result)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        #if len(cpu_matched_blocks) > len(ssd_matched_blocks):
        #    print(f"[PUT_LOCAL] CPU matched blocks are greater than SSD matched blocks, skipping")
        #    return self._empty_put_return(request_id)


        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )

        if enable_ssd:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)

        if len(fragment12_cpu_blocks) < fragment12_num_blocks or \
            len(fragment2_ssd_blocks) < fragment2_num_blocks:
            print(f"[WARNING] PUT request {request_id} FAILED: CPU={len(fragment12_cpu_blocks)}/{fragment12_num_blocks}, SSD={len(fragment2_ssd_blocks)}/{fragment2_num_blocks}")
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            if enable_ssd:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
            # No transfer will consume the matched prefix; release its ref now.
            for fn in match_finalizers or []:
                fn()
            return self._empty_put_return(request_id)

        cpu_swa_slot = -1
        ssd_swa_slot = -1

        if self.swa_op_constructor.enabled:
            cpu_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                cpu_matched_result.last_node)
            if cpu_swa_slot >= 0 and fragment2_num_blocks > 0:
                ssd_swa_slot = self.ssd_cache_engine._alloc_swa_slot(
                    ssd_matched_result.last_node)
            if (cpu_swa_slot < 0 or
                    (fragment2_num_blocks > 0 and ssd_swa_slot < 0)):
                return self._fail_put_before_insert(
                    request_id=request_id,
                    reason="swa_slot_alloc_failed",
                    cpu_blocks=fragment12_cpu_blocks,
                    cpu_swa_slot=cpu_swa_slot,
                    ssd_blocks=fragment2_ssd_blocks if enable_ssd else None,
                    ssd_swa_slot=ssd_swa_slot,
                    match_finalizers=match_finalizers,
                )

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            dp_client_id = dp_client_id,
        )
        flexkv_logger.info(
            "[FlexKV-SEGV-DEBUG] cache_engine create D2H op (local_put) "
            f"request_id={request_id}, op_id={op_d2h.op_id}, "
            f"graph_id={transfer_graph.graph_id}, dp_client_id={dp_client_id}, "
            f"fragment12_num_blocks={fragment12_num_blocks}, "
            f"fragment2_num_blocks={fragment2_num_blocks}, "
            f"{summarize_id_tensor('gpu_src', fragment12_gpu_blocks)}, "
            f"{summarize_id_tensor('cpu_dst', fragment12_cpu_blocks)}"
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if fragment2_num_blocks > 0:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                flexkv_logger.warning(f"fragment12_cpu_blocks: {len(fragment12_cpu_blocks)}, "
                                      f"fragment2_num_blocks: {fragment2_num_blocks}, "
                                      f"cpu match blocks are bigger than SSD match blocks number. "
                                      f"This should not often happen if CPU cache size is smaller than SSD cache size.")
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        if cpu_swa_slot >= 0:
            empty = np.array([], dtype=np.int64)
            swa_ops = self.swa_op_constructor.build_put_chain(
                transfer_graph,
                gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
                cpu_slot_ids=np.array([cpu_swa_slot], dtype=np.int64),
                ssd_slot_ids=(np.array([ssd_swa_slot], dtype=np.int64)
                              if ssd_swa_slot >= 0 else empty),
                remote_slot_ids=empty,
                dp_client_id=dp_client_id,
                return_op_ids=True,
            )
            assert swa_ops.d2h_id is not None
            if fragment2_num_blocks > 0:
                assert swa_ops.h2disk_id is not None
            finished_ops_ids.append(swa_ops.d2h_id)

        """insert and lock"""
        cpu_node_to_unlock = self.cpu_cache_engine.insert(
            sequence_meta,
            fragment12_cpu_blocks,
            is_ready=False,
            match_result=cpu_matched_result,
        )
        # insert() returns None when nothing was attached (suffix already in the
        # shared tree) — no unready node to flip ready after the transfer.
        if cpu_node_to_unlock is not None:
            op_node_to_ready[op_d2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if len(fragment2_ssd_blocks) > 0:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(
                sequence_meta,
                fragment2_ssd_blocks,
                is_ready=False,
                match_result=ssd_matched_result,
            )
            if ssd_node_to_unlock is not None:
                op_node_to_ready[op_h2disk.op_id] = (DeviceType.SSD, ssd_node_to_unlock, ssd_node_to_unlock.size())
        on_complete: List[Callable[[], None]] = []
        if cpu_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.CPU,
                                     cpu_node_to_unlock, cpu_node_to_unlock.size(), is_put=True))
        if ssd_node_to_unlock is not None:
            on_complete.append(self._defer_node_release(DeviceType.SSD,
                                     ssd_node_to_unlock, ssd_node_to_unlock.size(), is_put=True))
        on_complete.extend(match_finalizers)

        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)
        if cpu_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.d2h_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.CPU, cpu_node_to_unlock, cpu_swa_slot),
            )
        if ssd_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2disk_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.SSD, ssd_node_to_unlock, ssd_swa_slot),
            )
        skipped_gpu_blocks = len(cpu_matched_blocks)
        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            op_callback_dict=op_callback_dict,
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
            skipped_gpu_blocks=skipped_gpu_blocks,
            on_complete=on_complete,
        )

    def _put_impl_radixshmem(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> PutTransferPlan:
        """PUT planner for the radixshmem CPU+SSD tiers.

        Local-only, and not by choice: ``TransferType`` has H2PEERH/H2PEERSSD but
        ``transfer_engine`` routes neither, so there is no way to write into a
        peer's slots. A PUT match is therefore ``with_peer=False`` -- never spliced --
        and the only thing that differs from ``_put_impl_local`` is WHEN the tree
        learns about the slots: radixshmem accepts them only once they hold data,
        so both inserts move into the graph-completion callback.

        Block index:  0        cpu_tot        ssd_tot        block_mask_end
            GPU     : (skipped) |          fragment          |
                                     |  D2H into new slots
            CPU     : (cached) -+                            |
                                              |  H2DISK
            SSD     : (cached) ---------------+              |

        The two tiers have independent matched prefixes, so their new spans start
        at different blocks and each gets its own insert.
        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        assert enable_gpu
        assert self.cache_config.enable_cpu
        assert self.cpu_cache_engine is not None
        self._assert_radixshmem_no_swa("PUT")

        cpu_match, ssd_match = self._match_radixshmem(
            sequence_meta, temp_cache_strategy, is_get=False)

        def _release_match() -> PutTransferPlan:
            # Nothing will consume the matched prefix; drop the query's ref now.
            cpu_match.release()
            ssd_match.release()
            return self._empty_put_return(request_id)

        # How much of the window each tier already holds. ``is_get=False`` means no
        # peer tail, so the whole match is local, and intersecting it with the
        # window is what bounds it -- the same thing ``_put_impl_local`` gets from
        # slicing its matched blocks. A match stopping short of ``block_mask_start``
        # counts for nothing and one running past ``block_mask_end`` is trimmed, so
        # there is no boundary arithmetic to do here.
        num_skipped = len(cpu_match.local_range(block_mask_start, block_mask_end))
        # Nothing to spill without SSD: treat the tier as covering the window already.
        num_ssd_cached = ((block_mask_end - block_mask_start) if not enable_ssd
                          else len(ssd_match.local_range(block_mask_start, block_mask_end)))

        # First window block each tier does not already hold.
        cpu_tot = block_mask_start + num_skipped
        ssd_tot = block_mask_start + num_ssd_cached
        num_cpu_new = block_mask_end - cpu_tot
        num_ssd_new = block_mask_end - ssd_tot
        # Same policy as _put_impl_local: a fully-matched CPU prefix ends the PUT
        # even when SSD is still short of it.
        if num_cpu_new == 0:
            return _release_match()

        cpu_new = self.cpu_cache_engine.take(num_required_blocks=num_cpu_new,
                                             strict=False)
        if num_ssd_new > 0:
            assert self.ssd_cache_engine is not None
            ssd_new = self.ssd_cache_engine.take(num_required_blocks=num_ssd_new,
                                                 strict=False)
        else:
            ssd_new = np.array([], dtype=np.int64)

        if len(cpu_new) < num_cpu_new or len(ssd_new) < num_ssd_new:
            flexkv_logger.warning(
                f"radixshmem PUT {request_id} skipped: CPU "
                f"{len(cpu_new)}/{num_cpu_new}, SSD {len(ssd_new)}/{num_ssd_new}"
            )
            self.cpu_cache_engine.recycle(cpu_new)
            if num_ssd_new > 0:
                self.ssd_cache_engine.recycle(ssd_new)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            return _release_match()

        transfer_graph = TransferOpGraph()
        finished_ops_ids: List[int] = []

        fragment_gpu_blocks = gpu_block_ids[num_skipped:]
        op_d2h = TransferOp(
            graph_id=transfer_graph.graph_id,
            transfer_type=TransferType.D2H,
            src_block_ids=fragment_gpu_blocks,
            dst_block_ids=cpu_new,
            dp_client_id=dp_client_id,
        )
        transfer_graph.add_transfer_op(op_d2h)
        # Task end is D2H alone, as in _put_impl_local: the request is free once
        # its GPU blocks are drained, the SSD spill finishes behind it.
        finished_ops_ids.append(op_d2h.op_id)

        if len(ssd_new) > 0:
            # H2DISK covers absolute blocks [ssd_tot, block_mask_end). From cpu_tot on that is
            # the staging just taken; anything before it is already in CPU cache,
            # so it is read from the matched slots.
            if ssd_tot >= cpu_tot:
                h2disk_src = cpu_new[ssd_tot - cpu_tot:]
            else:
                h2disk_src = np.concatenate(
                    [cpu_match.local_range(ssd_tot, cpu_tot), cpu_new])
            assert len(h2disk_src) == len(ssd_new)
            op_h2disk = TransferOp(
                graph_id=transfer_graph.graph_id,
                transfer_type=TransferType.H2DISK,
                src_block_ids=h2disk_src,
                dst_block_ids=ssd_new,
                dp_client_id=dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)
            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        # Both inserts run at graph completion rather than on their own op's
        # callback: radixshmem admits only blocks that already hold data, and
        # completion is the first point past every op that writes them -- D2H
        # filling the CPU staging, and H2DISK reading it back out.
        on_complete: List[Callable[[], None]] = []

        def _arm(engine, slots: np.ndarray,
                 hold: Callable[[], None], label: str) -> None:
            # ``hold`` is the ref this tier's insert needs: its span starts where
            # that match ran out, and radixshmem rejects a span whose start the
            # local tree no longer reaches. Handing it to the staged insert is
            # what keeps it alive for exactly that long -- both exits drop it.
            staged = StagedRadixInsert(engine=engine,
                                        sequence_meta=sequence_meta,
                                        slots=slots,
                                        path_end=block_mask_end,
                                        label=label,
                                        holds=[hold])
            on_complete.append(staged.publish)
            # on_complete never runs for a cancelled graph, and the slots plus the
            # locked prefix would stay held for the life of the region. Both exits
            # are idempotent, so arming this as well as the publish is fine.
            transfer_graph.add_cancel_cleanup(staged.abort)

        _arm(self.cpu_cache_engine, cpu_new,
             cpu_match.release, f"PUT {request_id} CPU")
        if len(ssd_new) > 0:
            _arm(self.ssd_cache_engine, ssd_new,
                 ssd_match.release, f"PUT {request_id} SSD")
        else:
            # No SSD span to insert, so nothing constrains when this ref goes.
            on_complete.append(ssd_match.release)
            transfer_graph.add_cancel_cleanup(ssd_match.release)

        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            op_callback_dict={},
            num_gpu_blocks_to_transfer=len(fragment_gpu_blocks),
            skipped_gpu_blocks=num_skipped,
            on_complete=on_complete,
        )

    @staticmethod
    def _collect_finalizers(*match_results: Optional[MatchResultAccel]) -> List[Callable]:
        """Gather radixshmem match finalizers; process-internal tiers yield none."""
        return [mr.finalize for mr in match_results
                if mr is not None and mr.finalize is not None]

    def _defer_node_release(self,
                            device_type: DeviceType,
                            node: object,
                            ready_length: int,
                            is_put: bool) -> Callable[[], None]:
        """Lock ``node`` now; return a closure that unlocks + set_ready
        (+ PUT publish) it, to run at graph completion."""
        engine = self.cache_engines[device_type]
        engine.lock_node(node)

        def _release() -> None:
            engine.release_node(node, ready_length)
            if not is_put:
                return
            if device_type == DeviceType.CPU and self.cache_config.enable_p2p_cpu:
                engine.local_index.insert_and_publish(node)
            elif device_type == DeviceType.SSD and self.cache_config.enable_p2p_ssd:
                engine.local_index.insert_and_publish(node)
            elif device_type == DeviceType.REMOTE and self.enable_kv_sharing:
                engine.insert_and_publish(node)
        return _release

    def _defer_recycle(self,
                       device_type: DeviceType,
                       blocks: np.ndarray) -> Optional[Callable[[], None]]:
        """Return a closure recycling ``blocks`` back to ``device_type``'s pool,
        or None when there is nothing to recycle."""
        if len(blocks) == 0:
            return None
        engine = self.cache_engines[device_type]
        return lambda: engine.recycle(blocks)

    @staticmethod
    def _transfer_callback(on_complete: List[Callable[[], None]]) -> None:
        """Run every deferred completion action in order (node release, buffer
        recycle, radixshmem source-ref release)."""
        for action in on_complete:
            action()

    def _op_callback(self, device_type: DeviceType, node_to_ready: RadixNode, ready_length: int) -> None:
        if device_type == DeviceType.CPU:
            assert self.cpu_cache_engine is not None
            self.cpu_cache_engine.set_ready(node_to_ready, True, ready_length)
        elif device_type == DeviceType.SSD:
            assert self.ssd_cache_engine is not None
            self.ssd_cache_engine.set_ready(node_to_ready, True, ready_length)
        elif device_type == DeviceType.REMOTE:
            assert self.remote_cache_engine is not None
            self.remote_cache_engine.set_ready(node_to_ready, True, ready_length)

    @nvtx.annotate("Match Prefix Accel", color="yellow")
    def match_local_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_put: bool = False,
                        gpu_matched_blocks: int = 0) \
                            -> Tuple[MatchResultAccel, MatchResultAccel]:
        #from flexkv.common.debug import flexkv_logger, summarize_id_tensor
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            if not self.cache_config.enable_p2p_cpu:
                cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
            else:
                #flexkv_logger.info(f"[MATCH DEBUG] CPU P2P enabled, calling match_all() instead of match_local()")
                if is_put:
                    cpu_matched_result = self.cpu_cache_engine.match_local(sequence_meta)
                else:
                    cpu_matched_result = self.cpu_cache_engine.match_all(sequence_meta, gpu_matched_blocks)
        if temp_cache_strategy.ignore_ssd:
            return cpu_matched_result, ssd_matched_result
        #TODO: we assume that ssd and gds are not enabled at the same time
        if self.ssd_cache_engine:
            if not self.cache_config.enable_p2p_ssd:
                ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
            else:
                #flexkv_logger.info(f"[MATCH DEBUG] SSD P2P enabled, calling match_all() instead of match_local()")
                if is_put:
                    ssd_matched_result = self.ssd_cache_engine.match_local(sequence_meta)
                else:
                    ssd_matched_result = self.ssd_cache_engine.match_all(sequence_meta, gpu_matched_blocks)

        return cpu_matched_result, ssd_matched_result

    def _is_mooncake_swa_tier(self, device_type: DeviceType) -> bool:
        """True for the key-addressed mooncake-store REMOTE tier: SWA hits are
        keyed by the hit block's tail hash instead of a node-mounted slot."""
        return (self.use_mooncake_store_backend
                and device_type == DeviceType.REMOTE)

    def _select_swa_read_source(
        self,
        block_mask_start: int,
        block_mask_end: int,
        tier_match_results: Dict[DeviceType, object],
        sequence_meta: Optional[SequenceMeta] = None,
    ) -> Tuple[int, SWAReadSource]:
        """Return the largest usable SWA-aware Full-KV end and its exact SWA source."""
        if not self.swa_op_constructor.enabled or not tier_match_results:
            return block_mask_start, SWAReadSource()

        candidates: List[Tuple[int, DeviceType, object]] = []
        for device_type, match_result in tier_match_results.items():
            if match_result is None:
                continue

            swa_hit = int(match_result.swa_hit_blocks)
            if swa_hit <= block_mask_start:
                continue

            if swa_hit > block_mask_end:
                # The radix match covers the complete token sequence, while the
                # request mask may stop earlier. A snapshot for a deeper trailing
                # window cannot serve this request window; try another tier.
                continue

            if not self._is_mooncake_swa_tier(device_type):
                assert match_result.last_swa_node is not None
            candidates.append((swa_hit, device_type, match_result))

        for usable_end, device_type, match_result in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            engine = self.cache_engines.get(device_type)
            if engine is None or not getattr(engine, "swa_enabled", False):
                continue

            if self._is_mooncake_swa_tier(device_type):
                assert sequence_meta is not None, (
                    "mooncake SWA source selection requires sequence_meta "
                    "for the tail hash")
                tail_hash = str(sequence_meta.block_hashes[usable_end - 1])
                return usable_end, SWAReadSource(
                    hit_blocks=usable_end,
                    device_type=device_type,
                    mooncake_tail_hash=tail_hash,
                )

            source_node = match_result.last_swa_node
            source_slot = int(source_node.swa_host_slot)
            assert source_slot >= 0

            return usable_end, SWAReadSource(
                hit_blocks=usable_end,
                host_slot=source_slot,
                node=source_node,
                device_type=device_type,
                engine=engine,
            )

        return block_mask_start, SWAReadSource()

    def _reserve_swa_read_source(
        self,
        graph: TransferOpGraph,
        source: SWAReadSource,
        protected_cpu_node,
        dp_client_id: int,
    ) -> Optional[SWAReadReservation]:
        """Pin a source and build its SWA load chain before committing a Full hit.

        Non-CPU sources need a transient CPU SWA staging slot. Allocation may
        evict through the CPU radix, so protect the CPU Full-KV node referenced by
        this GET. Returning ``None`` means the caller must report no cache hit;
        Full-only restore is invalid for an SWA-aware GET.

        Mooncake-store REMOTE sources are key-addressed: no pin / host slot;
        a placeholder remote slot id and ``mooncake_tail_hashes`` key the
        SWA ``REMOTE2H`` op.
        """
        assert self.cpu_cache_engine is not None
        if not source.found:
            return None

        is_mooncake_source = source.is_mooncake
        # Mooncake-store will skip pin_swa_node for remote source.
        if not is_mooncake_source:
            source.engine._pin_swa_node(source.node)

        staging_slot = -1
        cpu_swa_slots = np.array([source.host_slot], dtype=np.int64)
        ssd_swa_slots = np.array([], dtype=np.int64)
        remote_swa_slots = np.array([], dtype=np.int64)

        if source.device_type != DeviceType.CPU:
            staging_slot = self.cpu_cache_engine._alloc_swa_slot(
                protected_node=protected_cpu_node)
            if staging_slot < 0:
                if not is_mooncake_source:
                    self._swa_release_load_lock(
                        node=source.node, engine=source.engine)
                flexkv_logger.warning(
                    "[FlexKV-SWA] GET staging allocation failed; "
                    f"source={source.device_type}, hit_blocks={source.hit_blocks}"
                )
                return None
            cpu_swa_slots = np.array([staging_slot], dtype=np.int64)
            if is_mooncake_source:
                remote_swa_slots = np.array([0], dtype=np.int64)
            else:
                source_slots = np.array([source.host_slot], dtype=np.int64)
                if source.device_type == DeviceType.SSD:
                    ssd_swa_slots = source_slots
                else:
                    remote_swa_slots = source_slots

        h2d_id = self.swa_op_constructor.build_get_chain(
            graph,
            gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
            cpu_slot_ids=cpu_swa_slots,
            ssd_slot_ids=ssd_swa_slots,
            remote_slot_ids=remote_swa_slots,
            dp_client_id=dp_client_id,
            mooncake_tail_hashes=(
                [source.mooncake_tail_hash] if is_mooncake_source else None),
        )
        if h2d_id is None:
            if is_mooncake_source:
                self._swa_release_load_lock(node=None, staging_slot=staging_slot)
            else:
                self._swa_release_load_lock(
                    node=source.node,
                    staging_slot=staging_slot,
                    engine=source.engine,
                )
            return None

        return SWAReadReservation(
            source=source,
            staging_slot=staging_slot,
            h2d_id=h2d_id,
        )

    def _release_swa_read_reservation(
        self, reservation: Optional[SWAReadReservation]) -> None:
        if reservation is None:
            return
        self._swa_release_load_lock(
            node=reservation.source.node,
            staging_slot=reservation.staging_slot,
            engine=reservation.source.engine,
        )

    # The GPU-side SWA slot is a size-1 placeholder here (window == one page ==
    # one slot on DSv4). It is rebound late from the request's swa_slot_mapping
    # via TransferOpGraph.set_swa_gpu_blocks() in launch, mirroring the Full-KV
    # GPU late-bind.

    _SWA_GPU_PLACEHOLDER = np.array([0], dtype=np.int64)

    def _swa_release_load_lock(self, node, staging_slot: int = -1, engine=None) -> None:
        """SWA H2D completion callback: release the source pin and free any
        transient CPU staging slot.

        For a CPU-sourced load, ``node`` is the matched CPU SWA node and its pin
        is dropped with the plain dec (dec_swa_lock_ref, NOT dec_swa_lock_only):
        the loaded window stays cached for future reuse. For a staged
        (SSD/REMOTE) source, ``node`` is the source-tier node (same pin release)
        and ``staging_slot`` is the transient CPU SWA slot used as the DISK2H/
        REMOTE2H destination — it is unmounted (not a cached entry), so free it
        back to the CPU SWA pool. No-op on parts that are absent."""
        try:
            if node is not None and getattr(node, "swa_lock_ref", 0) > 0:
                node.dec_swa_lock_ref()
                if engine is not None:
                    engine.index.unlock(node)
                elif hasattr(node, "unlock"):
                    node.unlock()
                else:
                    node.lock_cnt -= 1
        except Exception:  # noqa: BLE001 — never let a callback crash the loop
            pass
        try:
            if staging_slot is not None and staging_slot >= 0:
                cpu_engine = self.cpu_cache_engine
                if cpu_engine is not None:
                    cpu_engine._free_swa_slot(int(staging_slot))
        except Exception:  # noqa: BLE001
            pass

    @nvtx.annotate("Match Prefix", color="yellow")
    def match_local(self,
                    sequence_meta: SequenceMeta,
                    temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                    is_put: bool = False) \
                        -> Tuple[MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result

    @nvtx.annotate("Match All Prefix accel", color="yellow")
    def match_all_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_get: bool = True) \
                            -> Tuple[MatchResultAccel, MatchResultAccel, MatchResultAccel]:
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        remote_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine and not temp_cache_strategy.ignore_remote:
            if self.enable_kv_sharing:
                if is_get:
                    remote_matched_result = self.remote_cache_engine.match_all(sequence_meta)
                else:
                    remote_matched_result = self.remote_cache_engine.match_local(sequence_meta)
            else:
                remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    @nvtx.annotate("Match All Prefix", color="yellow")
    def match_all(self,
                  sequence_meta: SequenceMeta,
                  temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY) \
                      -> Tuple[MatchResult, MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        remote_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine and not temp_cache_strategy.ignore_remote:
            remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    def _check_input(self,
                      token_ids: np.ndarray,
                      token_mask: np.ndarray,
                      slot_mapping: np.ndarray) -> None:
        assert token_ids.dtype == np.int64
        # assert token_mask.dtype == np.bool_, f"token_mask.dtype={token_mask.dtype}"
        assert slot_mapping.dtype == np.int64
        assert token_ids.ndim == 1
        assert token_mask.ndim == 1
        assert slot_mapping.ndim == 1
        assert token_ids.size == token_mask.size, f"token_ids.size={token_ids.size}, token_mask.size={token_mask.size}"
        assert slot_mapping.size == token_mask.sum(), \
            f"slot_mapping.size={slot_mapping.size}, token_mask.sum()={token_mask.sum()}"

    @staticmethod
    def slot_mapping_to_block_ids(slot_mapping: np.ndarray, tokens_per_block: int) -> np.ndarray:
        block_ids: np.ndarray = slot_mapping[::tokens_per_block] // tokens_per_block
        return block_ids

    def swa_slot_mapping_to_slot_ids(self, swa_slot_mapping: np.ndarray) -> np.ndarray:
        """Convert an SWA slot_mapping into page-granular SWA pool slot ids."""
        window = self.tokens_per_block
        sm = np.asarray(swa_slot_mapping, dtype=np.int64)
        return sm[::window] // window

    def _get_block_range(self,
                         token_mask: np.ndarray) -> Tuple[int, int]:
        mask_idx = np.where(token_mask)[0]
        if len(mask_idx) == 0:
            return 0, 0
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
