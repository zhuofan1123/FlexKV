# radixshmem 跨节点 KV reuse（多机 prefix 复用）

本文讲的是**让不同机器上的 FlexKV 互相复用 KV cache**：它是什么、怎么装、怎么配、怎么验证。
从零开始的一整套步骤都在本文里，不需要先读别的文档。

⚠️ **跨节点的装法和单机不一样**，不是"单机装好了再打开一个开关"：

- shmradix 必须**带 RDMA + etcd 重新编**（单机三个开关都可以关）
- 必须装 **mooncake**，而且要带上你配的那个 metadata backend（单机完全不需要 mooncake）
- 还要额外准备**一个 etcd** 和**一个专用 Redis**

所以请照 §3 从头装一遍，别复用单机的安装结论。

---

## 0. 名词速查

如果你是第一次接触这套东西，先扫一眼这张表，后文不再解释这些词。

| 名词 | 一句话解释 |
|---|---|
| KV cache | 大模型推理时每个 token 算出来的中间结果。同样的前缀（prefix）如果之前算过，把 KV 拿回来就不用再算一遍。 |
| block | KV cache 的最小管理单位，固定装 `tokens_per_block` 个 token（常见 16）。命中/搬运都按 block 计。 |
| radix tree | 一棵"前缀树"：拿一串 token 去查，能查出"最长的、已经算过的那一段前缀"对应哪些 block。 |
| shm radix tree | FlexKV 把这棵树放在 POSIX 共享内存（`/dev/shm`）里，同机所有 DP 进程直接读同一棵树，不用互相发消息。这就是 radixshmem。 |
| DP | data parallel。一个 vllm 实例里 N 个各自持有一份模型权重的 engine 进程，`--data-parallel-size N`。 |
| **节点（node）** | 本文里指"一套 FlexKV"：一棵 shm radix tree + 一个 transfer engine + 若干 DP。**通常一台机器一个节点**，但一台机器上也可以起多个（验证脚本就是这么干的）。 |
| cluster | 参加同一次跨节点复用的所有节点。节点数 = `world_size`。 |
| cluster rank / node id | etcd 在启动时给每个节点分配的编号 `0..world_size-1`。它同时就是数据面寻址用的 FlexKV node id。**它是输出，不是你配进去的值。** |
| RDMA | 网卡直接读写对端内存的技术。这里用来"隔着网络翻对端那棵树"，以及搬 KV 字节。 |
| etcd | 一个小型分布式键值存储。这里只干一件事：让所有节点在启动时互相认识（成员表 + 分配 rank）。 |
| mooncake | 一个点对点数据传输库（Mooncake Transfer Engine）。这里负责把 KV 字节从对端机器真的搬过来。 |
| PEERH2H / PEERSSD2H | FlexKV 内部的传输类型名：从 peer 的 CPU 内存 / SSD 搬到本机。日志里会看到这两个词。 |

---

## 1. 这个功能解决什么问题

单机的 shm radix tree 只在**同一台机器**的 DP 之间共享。于是：

- 节点 A 算过的 prefix，节点 B 不知道，来了同样的 prompt 还得重算一遍；
- 集群越大，重复计算越多，每台机器的 CPU cache 里存的东西高度重复。

跨节点 reuse 把这些树连起来：**任何一个节点都能命中集群里任何一个节点算过的 prefix**，
命中之后 KV 通过网络搬过来直接用。适用场景是"多机同模型服务 + prompt 前缀高度重复"
（system prompt、few-shot、多轮对话、同一批文档反复问）。

---

## 2. 它是怎么工作的

把 `world_size` 配成大于 1 之后，一次查询分两步走：

**控制面（先找到"这段 prefix 在谁手上"）**

```
本地这棵树先走 ──► 本地走完了，prefix 还没匹配完
                    │
                    ├─► 去 cluster 的 router hash table 查：谁在延续这段 prefix
                    │      （这张表本身也在 RDMA 可读的共享内存里）
                    └─► 隔着网络（RDMA read）直接 walk 对端那棵树，接着往下匹配
```

注意这里**没有中心索引服务、没有轮询、没有广播**：查询进程用 RDMA 单边读直接翻对端的
树，对端的 CPU 甚至不参与。远程 walk 的传输方式由 `FLEXKV_RADIX_REMOTE_OP_TRANSPORT`
决定（`dc` = RDMA DC，默认；`zmq` = 退化到消息通道，调试用）。

**数据面（再把 KV 字节搬过来）**

控制面返回的 `remote_node_id`（= 对端的 cluster rank）就是数据面要寻址的 node id。FlexKV
用 mooncake 从对端节点读那些 block：CPU 里的走 `PEERH2H`，SSD 上的走 `PEERSSD2H`。这里
Redis 只当**通讯录**用（`meta:<node_id>` 存对端的 mooncake / zmq 地址 + buffer 基址，
`node:<node_id>` 存存活 TTL），不参与查找。

**每一层（CPU / SSD）各自是一个独立的 cluster**：各有自己的树、自己的 rank 空间、自己的
etcd 子命名空间（`flexkv/server/shm_radix_bootstrap.py:cluster_id_for`）。正常情况下各层
分到的 rank 是一致的；万一不一致 FlexKV 会直接报错退出，因为一个 node id 必须能代表所有层。

---

## 3. 安装

一共五样东西：

| # | 装什么 | 跨节点的特殊要求 |
|---|---|---|
| §3.1 | shmradix | **必须** `-DENABLE_RDMA=ON -DENABLE_ETCD=ON` |
| §3.2 | FlexKV | 本体照常装；跨节点的数据面配置只能走 JSON 文件（§4.2） |
| §3.3 | mooncake | **必须装**，且要带上配置里写的 metadata backend（redis 或 http） |
| §3.4 | etcd | 全集群共用一个 |
| §3.5 | Redis | 这个 cluster **独占**一个 |

### 3.1 shmradix：必须带 RDMA + etcd

源码：<https://gitlab-master.nvidia.com/zhuofanl/radixshmem>，本文对应 **`dev` 分支**
（写作时 `26e3885 registry: lift the etcd plane out of the indexer`）。

```bash
git clone -b dev ssh://git@gitlab-master.nvidia.com:12051/haoxu/radixshmem.git
cd radixshmem
# 系统依赖：libxxhash-dev、liburing-dev、cmake、pybind11
#   跨节点还要：libibverbs-dev（RDMA）、Go toolchain（etcd client 是 Go 写的）
mkdir build && cd build
cmake .. -DENABLE_RDMA=ON -DENABLE_ETCD=ON && make -j
# Python binding（产物：python/shmradix/_core.cpython-3xx-x86_64-linux-gnu.so）
cd ../python && pip install -e . --no-build-isolation
```

三个构建开关（默认全 ON）的含义：

| 开关 | 依赖 | 关掉的后果 |
|---|---|---|
| `ENABLE_RDMA` | libibverbs | **没有跨节点**：region 只能是单机的，`is_distributed()` 恒假 |
| `ENABLE_ETCD` | Go toolchain | **没有 cluster bootstrap**，跨节点这条路整条不存在 |
| `ENABLE_MOONCAKE` | Mooncake 头 + 库（`-DMOONCAKE_ROOT=`） | 关掉 shmradix 自带的数据面。FlexKV 用自己的 PEERH2H / PEERSSD2H，**不需要它** |

也就是说：只跑单机时三个都可以关；跑跨节点必须 `-DENABLE_RDMA=ON -DENABLE_ETCD=ON`，
`ENABLE_MOONCAKE` 无所谓。

装错了不会静默降级：FlexKV 在 bootstrap 之后会检查 `is_distributed()`，为假就直接抛
`the installed shmradix extension was built without RDMA`。

### 3.2 FlexKV

```bash
cd FlexKV
pip install -e . --no-build-isolation
```

依赖：Cython（构建时）、torch、numpy、xxhash、liburing、expiring_dict、zmq、redis 等，详见
`requirements.txt`。运行时把两个库都放进 path：

```bash
export PYTHONPATH=/work/FlexKV:/work/radixshmem/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

容器里（例如 `vllm/vllm-openai:vX.X.X` 镜像不含 FlexKV 和它的 native 依赖）常见做法是把
FlexKV 和 radixshmem 的源码 bind-mount 进容器，再跑一次性 setup 脚本装依赖。注意
**§3.3 那步替换 `engine.so` 改的是容器 rootfs，容器重建后要重做**，最好一起写进 setup 脚本。

### 3.3 mooncake：必须装，且要带上配置里写的 metadata backend

单机模式完全不需要 mooncake；跨节点的**数据面**靠它搬 KV 字节。装是一行：

```bash
pip install mooncake-transfer-engine
```

但**光装上不一定够用**。mooncake 自己需要一个 metadata server 来交换
segment descriptor（谁的哪块内存、rkey 多少）。下文 §4.2 的配置模板写的是
`metadata_backend: "redis"`，但 **PyPI 上的 `mooncake-transfer-engine` wheel 是
`USE_REDIS=OFF` 编的**，只有 `P2PHANDSHAKE` 和 `http` 两个 plugin。配了 redis 会在
`transfer_metadata_plugin.cpp` 里报 `Unable to find metadata storage plugin redis` 然后
glog CHECK abort —— 表现是 FlexKV 的 transfer worker 进程 `exitcode=-6`。

先确认手里这份带不带：

```bash
SP=$(python -c 'import mooncake,os;print(os.path.dirname(mooncake.__file__))')
strings "$SP/engine.so" | grep -c RedisStoragePlugin   # >0 = 带 redis plugin；官方 wheel 是 0
```

**方案 A（不重编，最省事）**：把 mooncake 的 metadata 后端换成 http —— wheel 是
`USE_HTTP=ON` 编的，服务端也自带：

```bash
python -m mooncake.http_metadata_server --port 8080 &   # 集群里起一个，所有节点都指它
```

然后每个节点的 mooncake config 写 `"metadata_backend": "http"` +
`"metadata_server": "http://<那台机器>:8080/metadata"`。

**注意这只换 mooncake 自己的 metadata 通道**，跟 FlexKV 的 `meta:<node_id>` /
`node:<node_id>` 通讯录是两回事 —— §3.5 那个 Redis 仍然必需。两者可以是两个不同的实例
（`scripts/multi-nodes/start_multi_node_serving.sh` 就是 6379 给 FlexKV、6380 给 mooncake）。

**方案 B（源码重编，本文实测的路子）**：只重编 transfer engine 的 pybind 模块
（`engine` target），把产物换进已装好的 wheel 目录：

```bash
# 1. 版本跟已装的 wheel 对齐（这里 wheel 0.3.11.post1 <-> tag v0.3.11）
pip show mooncake-transfer-engine | grep -i version
git clone https://github.com/kvcache-ai/Mooncake.git /tmp/Mooncake
cd /tmp/Mooncake && git checkout v0.3.11 && git submodule update --init --recursive

# 2. 系统依赖。关键是 libhiredis-dev —— redis plugin 靠它，engine.so 运行时动态链
#    libhiredis.so.1，缺了 import 就报 undefined symbol
apt-get install -y build-essential cmake ninja-build pkg-config patchelf \
    libibverbs-dev libgoogle-glog-dev libjsoncpp-dev libcurl4-openssl-dev \
    libhiredis-dev libnuma-dev libpython3-dev libssl-dev libunwind-dev libasio-dev

# 3. yalantinglibs 是 CMake 的 REQUIRED 依赖，从 submodule 装
cmake -S extern/yalantinglibs -B /tmp/ylt-build \
      -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build /tmp/ylt-build -j && cmake --install /tmp/ylt-build
#    （懒人版：bash dependencies.sh，把 2+3 一起做掉，耗时更长）

# 4. 只编 engine target；store / etcd / cuda 这几块 FlexKV 不用，全关掉省时间
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DUSE_REDIS=ON -DUSE_HTTP=ON \
      -DUSE_ETCD=OFF -DWITH_STORE=OFF -DUSE_CUDA=OFF
cmake --build build --target engine -j

# 5. 替换 wheel 里的 engine.so（先备份官方版）
SP=$(python -c 'import mooncake,os;print(os.path.dirname(mooncake.__file__))')
cp -n "$SP/engine.so" "$SP/engine.so.wheel-bak"
cp build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so "$SP/engine.so"
```

只换这一个文件就够：`engine.so` 把 `transfer_engine` 静态链进去了，`INSTALL_RPATH` 是
`$ORIGIN`，所以它继续用 wheel 目录里自带的 `libasio.so`，其余依赖都是系统库 —— 不需要
重打 wheel。

验证：

```bash
SP=$(python -c 'import mooncake,os;print(os.path.dirname(mooncake.__file__))')
strings "$SP/engine.so" | grep -c RedisStoragePlugin   # 现在应该 >0
ldd "$SP/engine.so" | grep hiredis                     # 能解析到 libhiredis.so.1
python -c "import mooncake.engine; print('ok')"
```

坑：

- `pip install -U mooncake-transfer-engine` 会把 `engine.so` 覆盖回官方版，升级后要重新
  拷一次（`engine.so.wheel-bak` 是官方那份，需要回滚时用它）
- 容器里改的是 rootfs，容器**重建**后要重做（把这几行塞进容器的 setup 脚本里）
- 大版本要跟 wheel 对齐：`mooncake` 包里的 `.py` 和 `engine.so` 的 pybind 接口是耦合的，
  跨版本混用会 import 失败或行为不一致
- Redis 带密码时：mooncake config 里填 `metadata_server_auth`，或者用
  `MC_REDIS_PASSWORD` / `MC_REDIS_USERNAME` / `MC_REDIS_DB_INDEX` 这几个 env

### 3.3 etcd：一个就够

所有节点连同一个 etcd 即可（不需要每节点一个）。它只在启动阶段用一次：每个节点把自己写进
`<cluster_id>/.../peers`，leader 等到 `world_size` 个成员齐了，按 key 排序分配 rank。

单节点 etcd（测试足够）：

```bash
etcd --name t --data-dir /tmp/flexkv_etcd.etcd \
     --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://10.0.0.1:2379 \
     --listen-peer-urls http://0.0.0.0:2380 \
     --initial-advertise-peer-urls http://10.0.0.1:2380 \
     --initial-cluster t=http://10.0.0.1:2380 &
```

`FLEXKV_RADIX_REGISTRY` 写 `etcd://10.0.0.1:2379` 或 `http://10.0.0.1:2379` 都认（scheme
会被剥掉）。

### 3.4 Redis：必须是这个 cluster 独占的

数据面的通讯录，装个普通 `redis-server` 就行，但**不能跟别的 FlexKV 部署共用**，原因见
§4.4 的第一条注意事项。

---

## 4. 配置与启动

### 4.1 环境变量

分成两类 —— **"全 cluster 必须一致"** 和 **"每节点必须不同"**。弄反了不会报错，只会表现成
"各自成了一个 cluster"（互相看不见，peer 命中恒为 0）或"卡在 bootstrap 直到超时"。

全 cluster 一致（写错一个字就找不到彼此）：

```bash
export FLEXKV_RADIX_SHMEM=1
export FLEXKV_RADIX_WORLD_SIZE=4                     # 实际节点数，必须精确
export FLEXKV_RADIX_REGISTRY=etcd://10.0.0.1:2379    # 同一个 etcd
export FLEXKV_RADIX_CLUSTER_ID=flexkv                # etcd 命名空间
export FLEXKV_SHM_RADIX_ID=prod_a                    # 见下面 ⚠️
```

> ⚠️ `FLEXKV_SHM_RADIX_ID` 在**单机**模式下只是本机 shm / TE channel 的名字（同机多实例
> 时用来区分），但跨节点模式下它同时参与 etcd 命名空间
> （`<cluster_id>/<shm_radix_id>/<tier>`，见
> `flexkv/server/shm_radix_bootstrap.py:cluster_id_for`），所以**必须全 cluster 一致**。
> 本机 shm 名不会因此撞车：`world_size > 1` 时 region 名会自动加 `_r<rank>` 后缀。

每节点不同：

```bash
export FLEXKV_RADIX_RANK=2                  # 本机身份标签，0..N-1 唯一即可（cluster rank 由 etcd 分配）
export FLEXKV_RADIX_RPC_ADDRESS=10.0.0.12   # 本机 IP，写进 etcd 给对端拨回
# 或 export FLEXKV_RADIX_RPC_INTERFACE=eth0 # 给网卡名让它自己取（两个都设时用 interface）
export FLEXKV_RADIX_RDMA_DEV=mlx5_0         # 本机 RDMA 设备；空 = 第一个可用设备
export FLEXKV_DP_SIZE=8                     # 本节点的 --data-parallel-size（各节点可以不同）
export FLEXKV_CONFIG_PATH=/etc/flexkv/node2.json
export MOONCAKE_CONFIG_PATH=/etc/flexkv/mooncake_node2.json
```

其余可选项：

```bash
export FLEXKV_RADIX_GID_IDX=3                 # RDMA GID index
export FLEXKV_RADIX_REMOTE_OP_TRANSPORT=dc    # 远程 walk 的通道：dc（默认）或 zmq
export FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC=120 # 等全员到齐的上限，秒
```

`FLEXKV_SERVER_RECV_PORT` 只在**一台机器上跑多个节点**时才需要区分（它是本机 ipc 路径）。

关于 `FLEXKV_RADIX_RANK`：它只是**本机的身份标签**，用来给本机 shm region 和 etcd key 起
名。真正的 cluster rank 是 etcd 分配的（按 `/peers` key 排序），FlexKV 从
`RadixServer.rank()` 读回来当 node id 用。所以 rank 标签只要节点间互不相同就行，不必等于
最终的 node id。

### 4.2 数据面必须走 JSON 配置文件

`enable_p2p_cpu` / `redis_*` / `local_zmq_*` / `local_ip` **没有对应的环境变量**
（`load_user_config_from_env()` 不读它们），只有 `FLEXKV_CONFIG_PATH` 指向的 JSON/YAML 会被
`update_default_config_from_user_config` 拷进 `CacheConfig`。每节点一份：

```json
{"cpu_cache_gb": 200,
 "ssd_cache_gb": 0,
 "enable_p2p_cpu": true,
 "enable_p2p_ssd": false,
 "redis_host": "10.0.0.1",
 "redis_port": 6379,
 "redis_password": "",
 "local_ip": "10.0.0.12",
 "local_zmq_ip": "10.0.0.12",
 "local_zmq_port": 5454}
```

- `enable_p2p_cpu` / `enable_p2p_ssd` 决定哪一层允许从 peer 取；它们会自动打开
  `enable_kv_sharing`，Mooncake + Redis 随之生效
- `local_ip` / `local_zmq_ip` 填**本机**可路由 IP（不是 127.0.0.1，否则对端拨不回来）
- `local_zmq_port` 的下一个端口也要空着（PEERSSD2H 用 `port+1`）
- 各节点的 `cpu_cache_gb` 可以不同（容量不必对齐），但**block 的字节布局必须一样**：
  同一个模型、同样的 `tokens_per_block`、同样的 KV dtype。最省心的做法是所有节点配成一样

对应的 mooncake 配置（每节点一份，`engine_ip` 填本机 IP、`device_name` 填本机网卡）：

```json
{"engine_ip": "10.0.0.12", "engine_port": 12345,
 "metadata_backend": "redis", "metadata_server": "redis://10.0.0.1:6380",
 "metadata_server_auth": "", "protocol": "rdma", "device_name": "mlx5_0"}
```

（`metadata_backend` 选 `redis` 还是 `http`，取决于 §3.3 你走的是哪条路。）

### 4.3 启动

环境变量按 §4.1 配好之后，每个节点起自己的 `vllm serve`：

```bash
vllm serve <model_path> \
    --tensor-parallel-size 1 --data-parallel-size 8 \
    --port 31002 \
    --max-num-seqs 256 --max-num-batched-tokens 8192 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.28 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
```

| 参数 | 必填 | 解释 |
|---|---|---|
| `--kv-transfer-config '{...}'` | ✅ | 启用 FlexKV connector，没有它 FlexKV 整条路都不生效 |
| `--data-parallel-size N` | ✅ | 必须跟本节点的 `FLEXKV_DP_SIZE` 一致 |
| `--no-enable-prefix-caching` | 强烈建议 | 关掉 vllm 自带的 prefix cache，让 FlexKV 路径独占；否则两层 cache 混在一起，命中率没法评估 |
| `--gpu-memory-utilization` | 看卡 | 给 KV cache 留余量 |

**所有节点必须一起拉起。** `world_size > 1` 的 bootstrap 是 collective 的：每个节点都阻塞
在 `RadixServer.bootstrap()` 里等全员到齐（上限 `FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC`，
默认 120 s），所以先起的那个节点在最后一个节点起来之前不会 ready。超时就整体失败退出。

集群级前提清单：

- 所有节点的 shmradix 都带 `-DENABLE_RDMA=ON -DENABLE_ETCD=ON`（§3.1）
- 节点之间 RDMA 互通，TCP 也要能互连：bootstrap / XRC 的监听端口是 OS 临时分配的，FlexKV
  没暴露成固定值，**所以别在节点之间做端口白名单**
- 一个所有节点都连得上的 etcd（§3.4）
- 一个这个 cluster 独占的 Redis（§3.5）
- 每个节点的 mooncake 能用上配置里写的 metadata backend（§3.3）

`scripts/multi-nodes/start_multi_node_serving.sh` 可以当启动模板读（它本身是老的
Redis-索引路径，但配置文件的形状可以照抄）。

### 4.4 注意事项与限制

- **这个 Redis 必须由这一个 cluster 独占。** 控制面本身不需要 Redis（prefix 靠 RDMA 走
  router hash table 发现），Redis 只剩数据面通讯录：`meta:<node_id>`（mooncake / zmq 地址 +
  buffer 基址）和 `node:<node_id>`（存活 TTL）。代价是 cluster rank 直接当 node_id 用，而
  别的 FlexKV 部署的 node_id 是 `INCR global:node_id` 从 1 开始分配的，撞上 rank 1,2,3 是
  常态。**撞了不会报错**：按 rank 注册是无条件覆盖 `node:<rank>`（没有占用校验），而
  `meta:` 里存的是对端 CPU buffer 的基址，于是 peer read 会读到另一个部署的内存。共用一个
  Redis 之前必须自己确认没有别的 FlexKV 在写。
- **不会静默降级。** bootstrap 返回失败直接抛；`bootstrap()` 成功但 `is_distributed()` 为假
  （= 构建时没开 RDMA）也直接抛，不会悄悄退回 local-only。
- **不能和 `enable_remote`（PCFS 这一层第三方存储）同时开**，`KVManager` 会直接报错：那一层
  有自己的 Redis 索引和 GET 规划，两套 match 合不到一起。
- **SWA（滑窗）模型不支持**：radixshmem 后端的 GET/PUT 规划只处理连续 prefix，
  `_get_impl_radixshmem` / `_put_impl_radixshmem` 遇到 SWA 配置直接报错，不会悄悄按
  full-attention 搬一遍。
- **各层 rank 必须一致**：CPU / SSD 是各自独立 bootstrap 的 cluster，正常会分到同一个
  rank；不一致时 FlexKV 报错退出（一个 node id 必须能代表所有层）。

### 4.5 命中之后的行为

- GET 是本地和 peer 的组合匹配
- GET 不做 promotion：从 peer（或本地 SSD）搬过来的 block 只是这次请求的 staging，H2D 读完
  就把 slot 还给 mempool，不会插进本地 CPU 索引。复用并不因此丢掉，请求跑完时本来就会 PUT
  一次，本地副本由那次 PUT 写入。
- tail 只来自一个 peer。prefix 的延续散在两个 peer 上时只吃到第一跳那段，剩下的算 miss ——
  少赚一次，不影响正确性。

---

## 5. 验证

建议分两步：**先在一台机器上用 `scripts/run_shmradix_repro.sh` 把整条链路跑通**（不需要
第二台机器，RDMA 走 loopback），确认代码 + 依赖 + 配置都对；再按 §5.2 把同一套配置铺到真正
的多台机器上。

### 5.1 单机预演：`scripts/run_shmradix_repro.sh` 在做什么

一句话：**在一台多卡机器上"假装"成两个 FlexKV 节点，让节点 0 先算一批 prompt，再把一模一样
的 prompt 只发给节点 1，检查节点 1 有没有真的把节点 0 算好的 KV 从网络上拿过来复用、而且
拿到的字节是对的。**

脚本自己会把下面这一整套拉起来（包含一个私有 redis），跑完自动清干净：

```
   节点 0（radix rank 0）  :31000       节点 1（radix rank 1）  :31001
   vllm serve --data-parallel-size 4    vllm serve --data-parallel-size 4
   CUDA_VISIBLE_DEVICES=0,1,2,3         CUDA_VISIBLE_DEVICES=4,5,6,7
    └ 一棵 shm radix tree + 一个 TE      └ 一棵 shm radix tree + 一个 TE
                       \               /
                        etcd 成员表（world_size=2）  ← 谁在这个 cluster 里
                        RDMA 远程 walk 对端的树      ← 控制面：prefix 在谁手上
                        mooncake + Redis             ← 数据面：把 KV 真搬过来
```

两个"节点"就是同一台机器上的两组互不相干的进程：各占一半 GPU，各有自己的 shm radix tree、
自己的 transfer engine、自己的 API 端口，彼此**只能**通过 etcd + RDMA + mooncake 通信。跟真
的两台机器唯一的区别是网络走 loopback。

**workload 为什么能证明"跨节点"**：

| 阶段 | 请求打到哪 | 作用 |
|---|---|---|
| phase 1 | 只发节点 0 | 把这批 prompt 的 KV 灌进节点 0 的 CPU cache |
| phase 2 | 只发节点 1（同一批 prompt） | 节点 1 自己从没见过这些 token，vllm 自带 prefix cache 也关了 —— 所以命中只可能来自"远程 walk 到节点 0 的树 + 把 KV 搬过来" |
| phase 3 | 再发节点 0（同一批 prompt） | 节点 0 从**自己本地** cache 复用同一段 prefix，当 phase 2 的对照组 |

- phase 1 之后脚本会先 drain（给每个 engine 补一条一次性请求）再 sleep 20 s：FlexKV 的 PUT
  是异步的，而且一个 block 只有在字节落地之后才 insert 进 radix 索引，节点 1 问得太早必然
  miss。
- phase 2 与 phase 3 同模型、同 prompt、同一段被复用的 block、`temperature=0`，唯一的区别是
  KV 从哪来，所以**输出必须逐字相同**；不同就说明过网的字节不是节点 0 存的那份。逐条比对
  结果写在 `$LOG_DIR/output_check.json`。

**前置条件**（缺哪样 preflight 会直接报出来）：

- 一个能 `import flexkv / shmradix / mooncake.engine` 的 venv（`VENV=`，默认
  `/root/vllm-env`），且 vllm 里注册了 `FlexKVConnectorV1`
- GPU 数 ≥ 2 × DP（默认 DP=4 → 要 8 卡；卡少就 `DP=2` / `DP=1`）
- 一个带 ACTIVE 口的 RDMA 设备（控制面的硬要求，不是性能选项）
- 一个连得上的 etcd（`ETCD=`，默认 `http://127.0.0.1:2379`；起法见 §3.4）
- PATH 上有 `redis-server`（脚本起一个私有实例，退出时杀掉）
- mooncake 数据面默认 `PROTOCOL=tcp`（同机 loopback 够用）；`PROTOCOL=rdma` 时才需要
  device_name。两种 protocol 都要求 mooncake 带 redis metadata plugin（脚本写的是
  `metadata_backend: redis`，装法见 §3.3）

**跑法**：

```bash
bash scripts/run_shmradix_repro.sh                      # 默认 2 节点 × DP4 = 8 卡
DP=2 bash scripts/run_shmradix_repro.sh                 # 2 节点 × DP2 = 4 卡
PROTOCOL=rdma HIT_MIN=100 bash scripts/run_shmradix_repro.sh
OUTPUT_CHECK=warn bash scripts/run_shmradix_repro.sh    # 两半 GPU 型号不同时（kernel 不一致，
                                                        # 正确的字节也可能解码出不同的词）
```

常用 env：`MODEL`（默认 `Qwen/Qwen3-0.6B`）、`PORT_BASE`（默认 31000）、`CPU_CACHE_GB`、
`NUM_PROMPTS`、`HIT_MIN`（phase 2 跨节点命中率下限，默认 50 %）、`LOG_DIR`、
`READY_TIMEOUT`、`OUTPUT_CHECK`（`strict`/`warn`/`off`）。

**通过标准**：脚本最后打一串 gate，全过才 `RESULT: PASS`。每条在查什么：

| gate | 查的是 |
|---|---|
| all N engines on radixshmem | 每个 DP engine 都走了 shmradix 路径，没有谁悄悄退回老路 |
| single shm radix region owner（每节点） | 一个节点只有一个 DP 建 region，其余 attach —— 两个 owner = `dp_client_id` 分配塌了 |
| N distinct FlexKV node ids | 两个节点确实 join 了**同一个** cluster 并拿到不同的 rank（各自 standalone 的话两边都是 0） |
| both nodes registered all DP GPUs | 每节点的所有 DP GPU 都接上了那个节点唯一的 TE |
| disjoint per-engine graph-id ranges | 各 engine 的 graph id 不串（id 是 `dp_client_id<<32`） |
| **peer prefix found（`peer_hit>0`）** | **本测试的核心：查询真的在对端的树上找到了 prefix** |
| remote tree walk issued RDMA reads | 那次查询真的发了 RDMA 读，不是本地侥幸命中 |
| PEERH2H transfers / moved non-zero bytes | KV 字节真的过了数据面（0 字节 = mooncake 空转） |
| **phase-2 peer KV == phase-3 local KV** | **搬过来的是"对的"字节：两次输出逐字相同** |
| no scheduler-loop errors / no tracebacks | 没崩 |
| node1 phase-2 GETs served from the peer ≥ `HIT_MIN`% | 跨节点复用率：节点 1 在 phase 2 发的每条 peer GET 里，命中对端的比例 |

日志留在 `$LOG_DIR`（默认 `/tmp/flexkv-xnode-<pid>`）：`n0.log` / `n1.log` 是两个节点的 vllm
全量日志，`all.log` 是合并版（gate 就是 grep 它），`output_check.json` 是 phase2/phase3 的
逐条输出对比。退出时脚本会杀掉本次运行的所有进程、清 `/dev/shm`、删自己在 etcd 里的 key、
杀私有 redis、停 MPS daemon。

### 5.2 在真正的多台机器上验证

按 §4 配好并启动之后，起服务时带上 `FLEXKV_TRACE_RADIX_PEER=1`（peer query 追踪，**必须在
启动前设好**；诊断用，线上关掉）。然后先只往节点 A 发一批 prompt，等它的异步 PUT 落地（几秒
到十几秒），再把同一批 prompt 发给节点 B，在**节点 B** 的日志里看：

```bash
grep "= FlexKV node id" vllm.log          # cluster rank / node id，各节点必须不同
# [kv manager] radix local label r2 -> cluster rank 2/4 = FlexKV node id 2; ...

grep "RADIX PEER QUERY" vllm.log | tail -3
# [RADIX PEER QUERY] shm=... local_only=False blocks=25 local_hit=10
#                    peer_rank=0 peer_hit=15 total_hit=25 ... rdma_reads=2 ...

grep "PEERH2H transfer request" vllm.log | tail -3
# [FlexKV-IO] ... bytes=42991616 PEERH2H transfer request: 0 finished ... 0.04 GB ...
```

- `peer_hit > 0` = 分布式 match 真的走到了 peer（`local_hit` 是拼接点，即本地那段有多长）
- `rdma_reads > 0` = 远程 walk 真的发了 RDMA 读
- `PEERH2H ... bytes=` 非 0 = KV 字节真的搬过来了（0 说明 mooncake 空转，查数据面配置）
- 如果 `peer_hit` 恒为 0 但 bootstrap 成功：多半是等得不够久（PUT 还没落地/发布），或两个
  节点其实不在同一个 etcd 命名空间（对一下 `FLEXKV_RADIX_CLUSTER_ID` +
  `FLEXKV_SHM_RADIX_ID`）

### 5.3 相关测试

| 文件 | 覆盖什么 |
|---|---|
| `tests/test_e2e_radix_peer_data.py` | per-(layer, block) 的**字节级**比对：peer 搬过来的 KV 与源节点的原始 KV 完全一致 |
| `tests/test_radix_shmem_engine.py` | 2-rank 的 engine 级单测（match / GET 规划 / rank 语义） |

---

## 6. 排错速查

| 现象 | 原因 |
|---|---|
| 所有节点都卡在启动，最后 bootstrap 超时 | bootstrap 是 collective 的：只起了一部分节点 / etcd 不通 / `world_size` 写得比实际节点数大 |
| `the installed shmradix extension was built without RDMA` | shmradix 编的时候没开 `ENABLE_RDMA`（§3.1） |
| bootstrap 直接失败，日志提示 registry | `FLEXKV_RADIX_REGISTRY` 没设或 etcd 连不上；或 shmradix 没开 `ENABLE_ETCD` |
| transfer worker `exitcode=-6` + `Unable to find metadata storage plugin redis` | mooncake 不是 `-DUSE_REDIS=ON` 编的（§3.3） |
| bootstrap 成功但 `peer_hit` 恒为 0 | 等得不够久（异步 PUT 没落地），或两节点不在同一 etcd 命名空间（`FLEXKV_RADIX_CLUSTER_ID` + `FLEXKV_SHM_RADIX_ID` 必须一致） |
| `peer_hit > 0` 但 `PEERH2H ... bytes=0` | 数据面没通：mooncake config 的 IP / device_name / metadata backend，或 Redis 通讯录 |
| 搬过来的 KV 解码出乱码 | block 字节布局不一致（模型 / `tokens_per_block` / KV dtype 各节点必须一样），或 Redis 被别的部署共用（§4.4 第一条） |
| `Still waiting for GPU registrations` / `Error in scheduler loop: KeyError` | `FLEXKV_DP_SIZE` 没跟 `--data-parallel-size` 对齐（vllm 在非 MoE DP 子进程里把 `data_parallel_size` 改成 1，宽度只能靠这个 env 传，见 `flexkv/integration/config.py:_resolve_vllm_dp_size`） |
| `no RDMA device with an ACTIVE port` | 机器上没有可用 RDMA 口，这条路走不了（脚本里 `RDMA_DEV=` 可以强制指定） |
| 报"不同 tier 分到了不同 cluster rank" | etcd 里有陈旧的 peer key，或不同层的 `world_size` 不一致；清掉 `<cluster_id>` 下的 key 重启全部节点 |

---

## 7. 相关文件

| 文件 | 作用 |
|---|---|
| `flexkv/server/shm_radix_bootstrap.py` | cluster bootstrap：shm region 命名、etcd 命名空间、rank 读回、各层一致性校验 |
| `flexkv/cache/radix_shmem_engine.py` | radixshmem 后端的 match / GET-PUT 规划（含 peer 拼接） |
| `scripts/run_shmradix_repro.sh` | 单机模拟 2 个 FlexKV 节点、端到端验证跨节点 reuse 的一键脚本（§5.1） |
| `scripts/multi-nodes/start_multi_node_serving.sh` | 多节点启动脚本模板（老的 Redis-索引路径，配置形状可照抄） |
| `tests/test_e2e_radix_peer_data.py` | peer KV 的字节级正确性测试 |
| `docs/radixshmem_integration.md` | 另一篇文档：**单机** radixshmem 集成的架构 / 调优 / 已知问题。跨节点不依赖它，想了解 shm radix tree 内部机制时再看 |
