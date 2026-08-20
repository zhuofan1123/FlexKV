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

import os
import subprocess
from typing import Optional, Tuple, List, Dict, Union, Iterable
import time

import numpy as np
import torch

from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig
from flexkv.common.transfer import TransferOpGraph
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.common.debug import flexkv_logger
from flexkv.cache.redis_meta import RedisMeta


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 dp_client_id: int = 0,
                 server_recv_port: str = "",
                 gpu_register_port: str = "",
                 event_collector: Optional[KVEventCollector] = None):
        flexkv_logger.info(f"{model_config = }")
        flexkv_logger.info(f"{cache_config = }")
        self.model_config = model_config
        self.cache_config = cache_config

        if server_recv_port != "":
            self.server_recv_port = server_recv_port
        else:
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if gpu_register_port != "":
            self.gpu_register_port = gpu_register_port
        else:
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

        self.use_radix_shmem = bool(getattr(GLOBAL_CONFIG_FROM_ENV,
                                            "radix_shmem", False))
        if self.use_radix_shmem and cache_config.enable_remote:
            # CacheEngineRadixShmem indexes the CPU/SSD tiers it owns in shm and
            # reaches peers over RDMA; the 3rd-party (PCFS) tier has its own
            # Redis-published index and GET planner. Mixing the two would need a
            # combined match, so keep them mutually exclusive.
            raise ValueError(
                "radix_shmem and enable_remote (3rd-party remote storage) "
                "cannot be enabled at the same time"
            )
        self._shm_radix_server_id = getattr(GLOBAL_CONFIG_FROM_ENV,
                                            "shm_radix_server_id", "default")

        flexkv_logger.info(
            f"[KVManager] IPC ports: server_recv_port={self.server_recv_port}, "
            f"gpu_register_port={self.gpu_register_port}"

        )

        if self.use_radix_shmem:
            flexkv_logger.info(f"[KVManager] radix_shmem is enabled"
                               f"[KVManager] shm_radix_server_id: {self._shm_radix_server_id}")

        # Multi-instance mode also requires server_client_mode
        if self.use_radix_shmem:
            # Force server_client_mode False — KVServer is bypassed entirely.
            self.server_client_mode = False
        else:
            self.server_client_mode = (model_config.dp_size > 1 or
                                       model_config.instance_num > 1 or
                                       GLOBAL_CONFIG_FROM_ENV.server_client_mode)

        flexkv_logger.info(
            f"[KVManager] instance_num={model_config.instance_num}, dp_size={model_config.dp_size}, "

            f"server_client_mode={self.server_client_mode}, "
            f"use_radix_shmem: {self.use_radix_shmem}"

        )

        self.redis_meta_client = None
        self.enable_mps = GLOBAL_CONFIG_FROM_ENV.enable_mps
        # Flat, instance-wide unique DP label (instance_id * dp_size + dp_rank),
        # so it doubles as the radix-shmem path's per-CE id: disjoint graph/op id
        # ranges and TE channel number.
        self.dp_client_id = dp_client_id
        # Owner handle for shm radix regions — only the bootstrap process
        # holds this; others have None.
        self._shm_radix_owners = None
        # TE-process handle — only the bootstrap process holds this.
        self._shm_te_process = None
        # Local KVTaskEngine for the radix-shmem path (per-DP).
        self.kv_task_engine = None
        self.server_handle = None

        if self.use_radix_shmem:
            self._init_radix_shmem_path(event_collector)
        elif self.server_client_mode:
            if dp_client_id == 0:
                self.server_handle = KVServer.create_server(model_config=model_config,
                                                            cache_config=cache_config,
                                                            gpu_register_port=self.gpu_register_port,
                                                            server_recv_port=self.server_recv_port,
                                                            inherit_env=False)

            else:
                self.server_handle = None
            self.dp_client = KVDPClient(
                self.server_recv_port,
                model_config=model_config,
                dp_client_id=dp_client_id,
            )
        else:
            # In non-server_client_mode, create RedisMeta here and pass to KVTaskEngine
            if self.cache_config.enable_kv_sharing:
                flexkv_logger.info(f"[kv manager] initializing RedisMeta and connection to "
                                   f"{self.cache_config.redis_host}:{self.cache_config.redis_port}")
                self.redis_meta_client = RedisMeta(
                    self.cache_config.redis_host,
                    self.cache_config.redis_port,
                    self.cache_config.redis_password,
                    self.cache_config.local_ip,
                    node_ttl_seconds=self.cache_config.node_ttl_seconds,
                )
                self.redis_meta_client.init_meta()
                # update distributed_node_id
                self.cache_config.distributed_node_id = self.redis_meta_client.get_node_id()

            self.server_handle = None
            self.kv_task_engine = KVTaskEngine(
                model_config,
                self.cache_config,
                self.gpu_register_port,
                redis_meta=self.redis_meta_client,
                event_collector=event_collector,
            )

    def _init_radix_shmem_path(self,
                               event_collector: Optional[KVEventCollector]) -> None:
        """Initialize the radix-shmem multi-DP path.

        Bootstrap proc (instance 0, dp 0):
          1. Create radix shm regions (one RadixServer per device type).
          2. Spawn the single TE subprocess (which creates per-CE shm channels).

        Other DP procs:
          1. Poll for the radix shm regions (created by bootstrap).
          2. Continue — TE channel attach blocks on `ShmControlBlock.wait_ready`
             inside `TransferManagerShmChannelHandle`.

        Each CE process gets a disjoint graph_id range so submissions to the
        single shared TE never collide.
        """
        from flexkv.server.shm_radix_bootstrap import (
            create_shm_radix_regions, attach_shm_radix_clients,
        )
        from flexkv.transfer_manager import TransferManagerShmTEProcess

        # dp_client_id is the flat cross-instance label, so id 0 is the single
        # bootstrap proc (same rule the server_client_mode branch uses).
        is_bootstrap = (self.dp_client_id == 0)
        total_clients = self.model_config.total_clients
        server_id = self._shm_radix_server_id
        radix_rank = getattr(GLOBAL_CONFIG_FROM_ENV, "radix_rank", 0)
        radix_world_size = getattr(
            GLOBAL_CONFIG_FROM_ENV, "radix_world_size", 1
        )
        # Two simulated ranks can share one host, so keep their TE channel / shm
        # namespaces apart.
        te_server_id = (
            f"{server_id}_r{radix_rank}"
            if radix_world_size > 1 else server_id
        )

        # Disjoint graph_id and op_id ranges per CE process: 2^32 ids per CE,
        # high bits = dp_client_id. Critical for the multi-DP path where all CE
        # procs feed a single TE that uses op_id as a primary key for internal
        # bookkeeping (op_id_to_op, op_id_to_nvtx_range, etc.).
        from flexkv.common.transfer import TransferOp
        TransferOpGraph.set_graph_id_range(
            self.dp_client_id << 32,
            (self.dp_client_id + 1) << 32,
        )
        TransferOp.set_op_id_range(
            self.dp_client_id << 32,
            (self.dp_client_id + 1) << 32,
        )

        # Radix regions come up FIRST: the cluster rank is an etcd-assigned
        # output of bootstrap, and it is the FlexKV node id everything downstream
        # (Redis registration, then the TE process) is keyed on.
        if is_bootstrap:
            self._shm_radix_owners = create_shm_radix_regions(
                self.model_config, self.cache_config,
                server_id=server_id,
                rank=radix_rank,
                world_size=radix_world_size,
                registry=getattr(GLOBAL_CONFIG_FROM_ENV, "radix_registry", ""),
                cluster_id=getattr(
                    GLOBAL_CONFIG_FROM_ENV, "radix_cluster_id", "flexkv"
                ),
                rpc_address=getattr(
                    GLOBAL_CONFIG_FROM_ENV, "radix_rpc_address", ""
                ),
                rpc_interface=getattr(
                    GLOBAL_CONFIG_FROM_ENV, "radix_rpc_interface", ""
                ),
                rdma_dev=getattr(GLOBAL_CONFIG_FROM_ENV, "radix_rdma_dev", ""),
                gid_idx=getattr(GLOBAL_CONFIG_FROM_ENV, "radix_gid_idx", 3),
                bootstrap_timeout_sec=getattr(
                    GLOBAL_CONFIG_FROM_ENV,
                    "radix_bootstrap_timeout_sec", 120,
                ),
                remote_op_transport=getattr(
                    GLOBAL_CONFIG_FROM_ENV, "radix_remote_op_transport", "dc"
                ),
            )
            cluster_rank = self._shm_radix_owners.cluster_rank
        else:
            # Wait until bootstrap created the shm radix regions before
            # GlobalCacheEngine tries to attach as RadixClient, and read the
            # cluster rank straight off the region — no second rendezvous.
            clients = attach_shm_radix_clients(
                self.cache_config,
                server_id=server_id,
                rank=radix_rank,
                world_size=radix_world_size,
            )
            cluster_rank = (int(next(iter(clients.values())).rank())
                            if clients else radix_rank)

        if self.cache_config.enable_kv_sharing:
            # A distributed radix query names its peer by cluster RANK, and the
            # peer data path (PEERH2H / PEERSSD2H) addresses it by FlexKV node
            # id — so make them the same number. etcd hands out the rank during
            # bootstrap and every process reads it back off the shm region, so
            # the index control plane still needs no Redis at all: Redis is left
            # carrying only the data path's address book (`meta:<id>` =
            # mooncake/zmq addresses + buffer base pointers, `node:<id>` =
            # liveness TTL), published by the transfer worker. Only the bootstrap
            # proc registers, since all DP procs of a rank share one node
            # identity (one mooncake engine in the shared TE process), and it
            # must happen BEFORE the TE process is spawned — that reads
            # cache_config.distributed_node_id.
            node_id = cluster_rank
            flexkv_logger.info(
                f"[kv manager] radix local label r{radix_rank} -> cluster rank "
                f"{cluster_rank}/{radix_world_size} = FlexKV node id {node_id}; "
                f"RedisMeta at "
                f"{self.cache_config.redis_host}:{self.cache_config.redis_port}"
            )
            self.redis_meta_client = RedisMeta(
                self.cache_config.redis_host,
                self.cache_config.redis_port,
                self.cache_config.redis_password,
                self.cache_config.local_ip,
                node_ttl_seconds=self.cache_config.node_ttl_seconds,
            )
            if is_bootstrap:
                if self.redis_meta_client.init_meta(node_id) is None:
                    raise RuntimeError(
                        f"Failed to register radix cluster rank {cluster_rank} "
                        f"as FlexKV node id {node_id}: "
                        f"{self.redis_meta_client.get_init_error()}"
                    )
            else:
                self.redis_meta_client.set_node_id(node_id)
            self.cache_config.distributed_node_id = int(node_id)

        if is_bootstrap:
            # Reserve extra channels beyond the internal DP clients so external
            # processes (e.g. a prefetch controller) can attach to the shared TE
            # using channel_ids in [total_clients, total_clients + num_extra).
            num_extra = getattr(GLOBAL_CONFIG_FROM_ENV, "num_extra_te_channels", 0)
            self._shm_te_process = TransferManagerShmTEProcess(
                self.model_config, self.cache_config,
                gpu_register_port=self.gpu_register_port,
                server_id=te_server_id,
                num_channels=total_clients + num_extra,
                total_clients=total_clients,
            )
            self._shm_te_process.start()

        # GlobalCacheEngine inspects GLOBAL_CONFIG_FROM_ENV.radix_shmem and
        # constructs CacheEngineRadixShmem (RadixClient) per device type.
        # KVTaskEngine wires up the shm-mode TransferManagerHandle.
        self.kv_task_engine = KVTaskEngine(
            self.model_config, self.cache_config,
            self.gpu_register_port,
            redis_meta=self.redis_meta_client,
            event_collector=event_collector,
            shm_te_server_id=te_server_id,
            shm_te_channel_id=self.dp_client_id,
        )

    def start(self) -> None:
        if self.enable_mps:
            # try to start MPS
            subprocess.run(['nvidia-cuda-mps-control', '-d'], check=False)
            flexkv_logger.debug("MPS started")

        if not self.server_client_mode:
            self.kv_task_engine.start()
        else:
            # send the start request to the server
            self.dp_client.start_server_and_register()

    def is_ready(self) -> bool:
        if self.server_client_mode:
            return self.dp_client.is_ready()
        else:
            return self.kv_task_engine.is_ready()

    def shutdown(self) -> None:
        if self.server_client_mode:
            self.dp_client.shutdown()
            # Wait for the server process to exit after sending shutdown request
            if self.server_handle is not None:
                self.server_handle.shutdown()
                self.server_handle = None
        else:
            if self.kv_task_engine is not None:
                self.kv_task_engine.shutdown()

        # Multi-DP radix-shmem teardown — only the bootstrap proc owns these.
        if self._shm_te_process is not None:
            self._shm_te_process.shutdown()
            self._shm_te_process = None
        if self._shm_radix_owners is not None:
            self._shm_radix_owners.shutdown()
            self._shm_radix_owners = None

        if self.enable_mps:
            flexkv_logger.info(
                "MPS is enabled. To stop MPS daemon manually, run: "
                "'echo quit | nvidia-cuda-mps-control'"
            )

    def get_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.get_async(token_ids,
                                               slot_mapping,
                                               token_mask,
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.get_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id

    def get_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  cpu_only: bool = False,
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False,
                  ) -> Tuple[int, np.ndarray]:
        """Match a prefix and build the load graph; return (task_id, mask).

        ``swa_aware=True`` clamps the Full-KV transfer to the reusable SWA window
        (from the same single match); the SWA window is the trailing block of the
        returned mask, which the caller reads directly. ``swa_aware=False``
        (default) is the plain path.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.get_match(token_ids,
                                                     token_mask,
                                                     cpu_only=cpu_only,
                                                     namespace=namespace,
                                                     swa_aware=swa_aware)
        else:
            task_id, mask = self.kv_task_engine.get_match(
                token_ids=token_ids,
                token_mask=token_mask,
                cpu_only=cpu_only,
                namespace=namespace,
                swa_aware=swa_aware,
            )
        return task_id, mask

    def put_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.put_async(token_ids, slot_mapping, token_mask,
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.put_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id

    def put_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.put_match(token_ids, token_mask,
                                                     namespace=namespace)
        else:
            task_id, mask = self.kv_task_engine.put_match(
                token_ids=token_ids,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id, mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       namespace: Optional[List[str]] = None) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.prefetch_async(token_ids, namespace=namespace)
            return task_id, 0
        else:
            task_id, actual_prefetch_tokens = self.kv_task_engine.prefetch_async(
                token_ids,
                namespace=namespace,
            )
            flexkv_logger.info(f"[FlexKV] prefetch: task_id={task_id}, actual_prefetch_tokens={actual_prefetch_tokens}")
        return task_id, actual_prefetch_tokens

    def launch(self,
               task_ids: Union[int, List[int]],
               slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]],
               swa_slot_mappings: Optional[Union[np.ndarray, List[Optional[np.ndarray]], torch.Tensor, List[Optional[torch.Tensor]]]] = None,
               as_batch: bool = False,
               layerwise_transfer: bool = False,
               counter_id: int = 0) -> List[int]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        # SWA GPU slot_mappings (optional): the connector supplies these only when
        # it registered an SWA GPU pool and the request has an SWA reuse window.
        if swa_slot_mappings is not None and not isinstance(swa_slot_mappings, List):
            swa_slot_mappings = [swa_slot_mappings]
        if isinstance(swa_slot_mappings, List):
            swa_slot_mappings = [
                sm.numpy() if isinstance(sm, torch.Tensor) else sm
                for sm in swa_slot_mappings
            ]
        if self.server_client_mode:
            return self.dp_client.launch_tasks(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                swa_slot_mappings=swa_slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id,
            )
        else:
            return self.kv_task_engine.launch_tasks(
                task_ids,
                slot_mappings,
                swa_slot_mappings=swa_slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id,
            )

    def cancel(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            self.dp_client.cancel_tasks(task_ids)
        else:
            self.kv_task_engine.cancel_tasks(task_ids)

    def wait(self,
             task_ids: Union[int, List[int]],
             timeout: float = 20.0,
             completely: bool = False) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.wait(task_ids, timeout, completely)
        else:
            return self.kv_task_engine.wait(task_ids, timeout, completely)

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.try_wait(task_ids)
        else:
            return self.kv_task_engine.try_wait(task_ids)

    # Only for testing
    def _clear_cpu_cache(self) -> None:
        if self.server_client_mode:
            flexkv_logger.error("clear_cache is not supported in server client mode")
            return
        else:
            self.kv_task_engine._clear_cpu_cache()
