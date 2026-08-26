# radixshmem 跨节点 KV reuse（多机 prefix 复用）

本文讲的是**让不同机器上的 FlexKV 互相复用 KV cache**：它是什么、怎么装、怎么配、怎么验证。
从零开始的一整套步骤都在本文里。

⚠️ shmradix 要**带 RDMA + etcd 重新编译**，还要额外装
mooncake、Redis。请照 §2 从头装一遍，别复用单机的安装结论。

---

## 0. 名词速查

| 名词                     | 一句话解释                                                                                                   |
|------------------------|---------------------------------------------------------------------------------------------------------|
| shm radix tree         | FlexKV 把这棵树放在 POSIX 共享内存（`/dev/shm`）里，同机所有 DP 进程直接读同一棵树，不用互相发消息。这就是 radixshmem。                         |
| node                   | 本文里指"一套 FlexKV"：一棵 shm radix tree + 一个 transfer engine + 若干 DP。**通常一台机器一个节点**，但一台机器上也可以起多个（验证脚本就是这么干的）。 |
| cluster                | 参加同一次跨节点复用的所有节点。节点数 = `world_size`。                                                                     |
| cluster rank / node id | etcd 在启动时给每个节点分配的编号 `0..world_size-1`。它同时就是数据面寻址用的 FlexKV node id。                                      |
| etcd                   | 一个小型分布式键值存储。这里只干一件事：让所有节点在启动时互相认识（成员表 + 分配 rank）。                                                       |
| mooncake               | 一个点对点数据传输库（Mooncake Transfer Engine）。这里负责把 KV 字节从对端机器真的搬过来。                                             |
| PEERH2H / PEERSSD2H    | FlexKV 内部的传输类型名：从 peer 的 CPU 内存 / SSD 搬到本机。日志里会看到这两个词。                                                  |

---

## 1. 它是怎么工作的

把 `world_size` 配成大于 1 之后，一次查询分两步走：

**控制面（先找到"这段 prefix 在谁手上"）**

```
本地这棵树先走 ──► 本地走完了，prefix 还没匹配完
                    │
                    ├─► 去 cluster 的 router hash table 查：谁在延续这段 prefix
                    │      （这张表本身也在 RDMA 可读的共享内存里）
                    └─► 隔着网络（RDMA read）直接 walk 对端那棵树，接着往下匹配
```

这里**没有中心索引服务、没有轮询、没有广播**：查询进程用 RDMA 单边读直接翻对端的树，对端的
CPU 甚至不参与。远程 walk 的传输方式由 `FLEXKV_RADIX_REMOTE_OP_TRANSPORT` 决定
（`dc` = RDMA DC，默认；`zmq` = 退化到消息通道，调试用）。

**数据面（再把 KV 字节搬过来）**

控制面返回的 `remote_node_id`（= 对端的 cluster rank）就是数据面要寻址的 node id。FlexKV
用 mooncake 从对端节点读那些 block：CPU 里的走 `PEERH2H`，SSD 上的走 `PEERSSD2H`。mooncake
通过 Redis 中的 `meta:<node_id>` 查询对端的 mooncake / zmq 地址 + buffer 基址，
Redis 的 `node:<node_id>` 存存活 TTL。

每一层（CPU / SSD）各有自己的树，各自在 etcd 的**独立命名空间**（`radix/<cluster_id>_cpu/`、
`radix/<cluster_id>_ssd/`）里报到、各自领 rank；同一个节点的各层用的是**同一个节点身份**
（§3.1），因此各层拿到的 rank 也一致。

---

## 2. 安装

一共五样东西：

### 2.1 shmradix：必须带 RDMA + etcd

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

三个构建开关（默认全 ON，单机时都可以关）的含义：

| 开关                | 依赖                                 | 关掉的后果                                                       |
|-------------------|------------------------------------|-------------------------------------------------------------|
| `ENABLE_RDMA`     | libibverbs                         | **没有跨节点**：region 只能是单机的，`is_distributed()` 恒假               |
| `ENABLE_ETCD`     | Go toolchain                       | **没有 cluster bootstrap**，跨节点这条路整条不存在                        |
| `ENABLE_MOONCAKE` | Mooncake 头 + 库（`-DMOONCAKE_ROOT=`） | 关掉 shmradix 自带的数据面。FlexKV 用自己的 PEERH2H / PEERSSD2H，**不需要它** |

### 2.2 FlexKV

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

### 2.3 mooncake：必须装

单机模式完全不需要 mooncake；跨节点的**数据面**靠它搬 KV 字节。装是一行：

```bash
pip install mooncake-transfer-engine
```

但**光装上不一定够用**。mooncake 自己需要一个 metadata server 来交换 segment descriptor
（谁的哪块内存、rkey 多少），而 **PyPI 上的 wheel 是 `USE_REDIS=OFF` 编的**，只有
`P2PHANDSHAKE` 和 `http` 两个 plugin。§3.2 的模板写的是 `metadata_backend: "redis"`，配了它
而 plugin 不在，就会 `Unable to find metadata storage plugin redis` 然后 abort —— 表现是
FlexKV 的 transfer worker 进程 `exitcode=-6`。

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

**注意这只换 mooncake 自己的 metadata 通道**，跟 FlexKV 的 `meta:<node_id>` 通讯录是两回事
—— §2.5 那个 Redis 仍然必需。两者可以是两个不同的实例
（`scripts/multi-nodes/start_multi_node_serving.sh` 就是 6379 给 FlexKV、6380 给 mooncake）。

**方案 B（源码重编，本文实测的路子）**：只重编 transfer engine 的 pybind 模块
（`engine` target），把产物换进已装好的 wheel 目录：

```bash
# 1. 版本跟已装的 wheel 对齐（这里 wheel 0.3.11.post1 <-> tag v0.3.11）
pip show mooncake-transfer-engine | grep -i version
git clone https://github.com/kvcache-ai/Mooncake.git /tmp/Mooncake
cd /tmp/Mooncake && git checkout v0.3.11 && git submodule update --init --recursive

# 2. 系统依赖。关键是 libhiredis-dev —— redis plugin 靠它，缺了 import 就报 undefined symbol
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

只换这一个文件就够，不需要重打 wheel。验证：

```bash
strings "$SP/engine.so" | grep -c RedisStoragePlugin   # 现在应该 >0
ldd "$SP/engine.so" | grep hiredis                     # 能解析到 libhiredis.so.1
python -c "import mooncake.engine; print('ok')"
```

### 2.4 etcd

所有节点连同一个 etcd 即可（不需要每节点一个）。它只在启动阶段用一次。

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

### 2.5 Redis：必须是这个 cluster 独占的

数据面的通讯录，装个普通 `redis-server` 就行，但**不能跟别的 FlexKV 部署共用**，原因见
§3.4 第一条。

---

## 3. 配置与启动

### 3.1 环境变量

分成两类 —— **"全 cluster 必须一致"** 和 **"每节点必须不同"**。

全 cluster 一致：

```bash
export FLEXKV_RADIX_SHMEM=1
export FLEXKV_RADIX_WORLD_SIZE=4                     # 实际节点数，必须精确
export FLEXKV_RADIX_REGISTRY=etcd://10.0.0.1:2379    # 同一个 etcd
export SHMRADIX_CLUSTER_ID=flexkv                    # etcd 命名空间的前缀，不设时是 default；FlexKV 按层
                                                     # 拼成 <前缀>_cpu / <前缀>_ssd，各层互不干扰；
                                                     # 同一个 etcd 上跑多套集群必须各起一个名字；同一个 etcd 下仅一个集群时，可以不设
```

每节点不同：

```bash
export FLEXKV_RADIX_RPC_ADDRESS=10.0.0.12   # 本机 IP，写进 etcd 给对端拨回；也决定节点身份
# 或 export FLEXKV_RADIX_RPC_INTERFACE=eth0 # 给网卡名让它自己取（两个都设时用 interface）
export FLEXKV_RADIX_RDMA_DEV=mlx5_0         # 本机 RDMA 设备；空 = 第一个可用设备
export FLEXKV_DP_SIZE=8                     # 本节点的 --data-parallel-size（各节点可以不同）
export FLEXKV_CONFIG_PATH=/etc/flexkv/node2.json
export MOONCAKE_CONFIG_PATH=/etc/flexkv/mooncake_node2.json
```

其余可选项：

```bash
export FLEXKV_RADIX_GID_IDX=3                 # RDMA GID index， 默认是3
export FLEXKV_RADIX_REMOTE_OP_TRANSPORT=dc    # 远程 walk 的通道：dc（默认）或 zmq
export FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC=120 # 等全员到齐的上限，默认是120秒
```

**节点身份**是 etcd 里 `/peers` 下的 key，默认由 bind IP 派生成 `node<bind-ip>`，bind IP 就是
上面 `FLEXKV_RADIX_RPC_ADDRESS` / `_INTERFACE` 解析出来的那个。**它必须每节点唯一**：两个节点
算出同一个身份就等于只有一个 peer 报到，bootstrap 会一直等到超时。所以

- 一台机器一个节点时，各机 IP 天然不同，不用额外配什么 —— 但**别填 `0.0.0.0`**，那样所有
  节点会算出同一个身份；
- 一台机器上跑多个节点时，给每个节点一个不同的具体地址（`127.0.0.1` / `127.0.0.2` / …，
  端口由内核分配不会撞），或者用 `SHMRADIX_NODE_NAME` 直接指定身份（验证脚本走的是后者）。

一台机器上跑多个flexkv实例时还需要为每个实例设置单独的ID：

```bash
export FLEXKV_SHM_RADIX_ID=node2               # shm region 名 + TE channel 名，默认 flexkv
```

### 3.2 JSON 配置文件

`enable_p2p_cpu` / `redis_*` / `local_zmq_*` / `local_ip` **没有对应的环境变量**，只有
`FLEXKV_CONFIG_PATH` 指向的 JSON/YAML 会被读进来。每节点一份：

```json
{
  "cpu_cache_gb": 200,
  "ssd_cache_gb": 2000,
  "ssd_cache_dir": "/mnt/nvme/flexkv_ssd",
  "enable_p2p_cpu": true,
  "enable_p2p_ssd": true,
  "redis_host": "10.0.0.1",
  "redis_port": 6379,
  "redis_password": "",
  "local_ip": "10.0.0.12",
  "local_zmq_ip": "10.0.0.12",
  "local_zmq_port": 5454
}
```

- `enable_p2p_cpu` / `enable_p2p_ssd` 决定哪一层允许从 peer 取；它们会自动打开
  `enable_kv_sharing`，Mooncake + Redis 随之生效
- `ssd_cache_gb` 必须**严格大于** `cpu_cache_gb`（否则 `CacheConfig` 直接报错）；不想开 SSD
  层就填 0 并把 `enable_p2p_ssd` 设 false
- `local_ip` / `local_zmq_ip` 填**本机**可路由 IP（不是 127.0.0.1，否则对端拨不回来）；
  `local_zmq_port` 的**下一个端口也要空着**，SSD 层的 peer 读用它
- 各节点的 `cpu_cache_gb` 可以不同（容量不必对齐），但**block 的字节布局必须一样**：
  同一个模型、同样的 `tokens_per_block`、同样的 KV dtype。最省心的做法是所有节点配成一样

对应的 mooncake 配置（每节点一份，`engine_ip` 填本机 IP、`device_name` 填本机网卡；
`metadata_backend` 选 `redis` 还是 `http` 取决于 §2.3 走的哪条路）：

```json
{
  "engine_ip": "10.0.0.12",
  "engine_port": 12345,
  "metadata_backend": "redis",
  "metadata_server": "redis://10.0.0.1:6380",
  "metadata_server_auth": "",
  "protocol": "rdma",
  "device_name": "mlx5_0"
}
```

### 3.3 启动

§2 的五样东西就位、环境变量按 §3.1 配好之后，每个节点起自己的 `vllm serve`：

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

| 参数                             | 必填  | 解释                                          |
|--------------------------------|-----|---------------------------------------------|
| `--kv-transfer-config '{...}'` | ✅   | 启用 FlexKV connector，没有它 FlexKV 整条路都不生效      |
| `--data-parallel-size N`       | 看需要 | 必须跟本节点的 `FLEXKV_DP_SIZE` 一致                 |
| `--no-enable-prefix-caching`   | 看需要 | true时关掉 vllm 自带的 prefix cache，让 FlexKV 路径独占 |
| `--gpu-memory-utilization`     | 看需要 | 给 KV cache 留余量                              |

两条网络前提：节点之间 RDMA 要互通，TCP 也要能互连 —— bootstrap 的监听端口是 OS 临时分配
的，FlexKV 没暴露成固定值，**所以别在节点之间做端口白名单**。

**所有节点必须一起拉起。** `world_size > 1` 的 bootstrap 是 collective 的：每个节点都阻塞
在里面等全员到齐（上限 `FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC`，默认 120 s），所以先起的那个
节点在最后一个节点起来之前不会 ready。超时就整体失败退出。

### 3.4 注意事项与限制

- ** FlexKV拉起的Redis 必须由这一个 cluster 独占。因为 cluster rank 直接当 node_id 用，而别的
  FlexKV 部署的 node_id 是从 1 开始递增分配的，撞上 rank 1,2,3 是常态。**撞了不会报错**：
  注册是无条件覆盖，而 `meta:` 里存的是对端 CPU buffer 的基址，于是 peer read 会读到另一个
  部署的内存。
- GET 是本地和 peer 的组合匹配，如果 peer 可以延长同一前缀，会从一个 peer 上读取尾部的部分。取KV的顺序是本地 CPU → peer
  CPU →
  本地 SSD → peer SSD。
- **各层的 rank 必须一致**：正常都会分到同一个 rank，不一致时 FlexKV 报错退出（一个 node id
  必须能代表所有层）。
- **不能和 `enable_remote`（PCFS 那一层第三方存储）同时开**，`KVManager` 会直接报错：那一层
  有自己的 Redis 索引和 GET 规划，两套 match 合不到一起。
- **SWA（滑窗）模型不支持**：radixshmem 后端的 GET/PUT 规划只处理连续 prefix，遇到 SWA 配置
  直接报错，不会悄悄按 full-attention 搬一遍。

---

## 4. 验证

建议分两步：先在一台机器上用 `scripts/run_shmradix_repro.sh` 把整条链路跑通，确认代码 + 依赖 + 配置都对；再按 §4.2
把同一套配置铺到真正
的多台机器上。

### 4.1 单机预演：`scripts/run_shmradix_repro.sh` 在做什么

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

两组进程各占一半 GPU、各有自己的树和 TE，彼此**只能**通过 etcd + RDMA + mooncake 通信 ——
跟真的两台机器唯一的区别是网络走 loopback。

| 阶段      | 请求打到哪              | 作用                                                                                     |
|---------|--------------------|----------------------------------------------------------------------------------------|
| phase 1 | 只发节点 0，8 条 prompt      | 把这 8 条的 KV 灌进节点 0（CPU 1 GB + SSD 8 GB 两层都写）                                            |
| flood   | 只发节点 0，另外 40 条         | 这 40 条跟上面 8 条没有公共 prefix，只用来占满节点 0 的小 CPU 层，把 phase 1 的 prefix 挤得只剩 SSD 层上那份            |
| phase 2 | 只发节点 1，phase 1 那 8 条   | 节点 1 自己从没见过这些 token，vllm 自带 prefix cache 也关了 —— 所以命中只可能来自"远程 walk 到节点 0 的树 + 把 KV 搬过来" |
| phase 3 | 再发节点 0，还是那 8 条         | 节点 0 复用同一段 prefix，当 phase 2 的对照组                                                       |

*这里40条prompt足够让1G CPU Cache发生驱逐*

**前置条件**（缺哪样 preflight 会直接报出来）：

- 一个能 `import flexkv / shmradix / mooncake.engine` 的 venv（`VENV=`，默认
  `/root/vllm-env`），且 vllm 里注册了 `FlexKVConnectorV1`
- GPU 数 ≥ 2 × DP（默认 DP=4 → 要 8 卡；卡少就 `DP=2` / `DP=1`）
- 一个带 ACTIVE 口的 RDMA 设备：控制面的硬要求，默认也是数据面（`PROTOCOL=rdma`）走的路
- 一个连得上的 etcd（`ETCD=`，默认 `http://127.0.0.1:2379`；起法见 §2.4）
- PATH 上有 `redis-server`（脚本起一个私有实例，退出时杀掉）
- mooncake 带 redis metadata plugin（脚本写的是 `metadata_backend: redis`，装法见 §2.3）
- SSD 层的磁盘空间：脚本给两个节点各分一个 8 GB 的 ssd 目录（同一个目录会让它们互相写对方的
  block 文件），退出时删掉

**跑法**：

```bash
bash scripts/run_shmradix_repro.sh                      # 默认 2 节点 × DP4 = 8 卡
DP=2 bash scripts/run_shmradix_repro.sh                 # 2 节点 × DP2 = 4 卡
SSD_CACHE_GB=0 bash scripts/run_shmradix_repro.sh       # 只测 CPU 层，phase 2 走 PEERH2H
PROTOCOL=tcp bash scripts/run_shmradix_repro.sh         # 数据面退回 tcp（RDMA 口不可用时）
OUTPUT_CHECK=warn bash scripts/run_shmradix_repro.sh    # 两半 GPU 型号不同时（kernel 不一致，
                                                        # 正确的字节也可能解码出不同的词）
```

其他 env：`MODEL`（默认 `Qwen/Qwen3-0.6B`）、`PORT_BASE`（默认 31000）、`CPU_CACHE_GB`（默认
1）、`SSD_CACHE_GB`（默认 8，填 0 = 只测 CPU 层）、`FLOOD_PROMPTS`（默认 40）、`NUM_PROMPTS`、
`HIT_MIN`（phase 2 跨节点命中率下限，默认 50 %）、`PROTOCOL`（默认 `rdma`）、`RDMA_DEV`、
`LOG_DIR`、`READY_TIMEOUT`、`OUTPUT_CHECK`（`strict`/`warn`/`off`）。

**通过标准**：输出 `RESULT: PASS`。
日志留在 `$LOG_DIR`（默认 `/tmp/flexkv-xnode-<pid>`）：`n0.log` / `n1.log` 是两个节点的 vllm
全量日志，`all.log` 是合并版，`output_check.json` 是逐条输出对比。退出
时脚本会清掉本次运行的所有进程、`/dev/shm`、自己在 etcd 里的 key、私有 redis 和 MPS daemon。

### 4.2 在真正的多台机器上验证

按 §3 配好并启动之后，起服务时带上 `FLEXKV_TRACE_RADIX_PEER=1`（peer query 追踪，**必须在
启动前设好**；诊断用，线上关掉）。然后先只往节点 A 发一批 prompt，等它的异步 PUT 落地（几秒
到十几秒），再把同一批 prompt 发给节点 B，在**节点 B** 的日志里看：

```bash
grep "= FlexKV node id" vllm.log          # cluster rank / node id，各节点必须不同
# [kv manager] radix node node2 = FlexKV node id 2, registered at 10.0.0.9:6379

grep "RADIX PEER QUERY" vllm.log | tail -3
# [RADIX PEER QUERY] shm=... local_only=False blocks=25 local_hit=10
#                    peer_rank=0 peer_hit=15 total_hit=25 ... rdma_reads=2 ...

grep -E "PEER(H2H|SSD2H) transfer request" vllm.log | tail -3
# [FlexKV-IO] ... bytes=42991616 PEERH2H transfer request: 0 finished ... 0.04 GB ...
```

- `peer_hit > 0` = 分布式 match 真的走到了 peer（`local_hit` 是拼接点，即本地那段有多长）
- `rdma_reads > 0` = 远程 walk 真的发了 RDMA 读
- `bytes=` 非 0 = KV 字节真的搬过来了；从对端 CPU 层来的是 `PEERH2H`，SSD 层是 `PEERSSD2H`

### 4.3 相关测试

| 文件                                  | 覆盖什么                                                       |
|-------------------------------------|------------------------------------------------------------|
| `tests/test_e2e_radix_peer_data.py` | per-(layer, block) 的**字节级**比对：peer 搬过来的 KV 与源节点的原始 KV 完全一致 |
| `tests/test_radix_shmem_engine.py`  | 2-rank 的 engine 级单测（match / GET 规划 / rank 语义）              |

---

## 5. 排错速查

| 现象                                                                             | 原因                                                                                                                                                                                                   |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 所有节点都卡在启动，最后 bootstrap 超时                                                      | bootstrap 是 collective 的：只起了一部分节点 / etcd 不通 / `world_size` 写得比实际节点数大 / 多个节点在 etcd 里是同一个身份，因此只看到一个 peer（§3.1）                                                                                         |
| `shm_open attach failed` 或 `Timed out attaching to shm radix region`           | 同机多节点没给不同的 `FLEXKV_SHM_RADIX_ID` 或身份相同，两边算出同一个 region 名互相抢（§3.1）；或 owner 那边 bootstrap 还没成功（跨节点的 region 是 bootstrap 里才创建的）；或 attach 方与 owner 的 `FLEXKV_RADIX_RPC_ADDRESS` / `_INTERFACE` 不一致，算出的名字对不上 |
| `cannot derive this node's radixshmem identity`                                | `world_size > 1`，但 `SHMRADIX_NODE_NAME` 和 `FLEXKV_RADIX_RPC_ADDRESS` / `_INTERFACE` 都没配（或网卡没有 IPv4），身份无从算起                                                                                           |
| `the installed shmradix extension was built without RDMA`                      | shmradix 编的时候没开 `ENABLE_RDMA`（§2.1）                                                                                                                                                                  |
| bootstrap 直接失败，日志提示 registry                                                   | `FLEXKV_RADIX_REGISTRY` 没设或 etcd 连不上；或 shmradix 没开 `ENABLE_ETCD`                                                                                                                                     |
| transfer worker `exitcode=-6` + `Unable to find metadata storage plugin redis` | mooncake 不是 `-DUSE_REDIS=ON` 编的（§2.3）                                                                                                                                                                |
| bootstrap 成功但 `peer_hit` 恒为 0                                                  | 等得不够久（异步 PUT 还没落地），或两节点不在同一 etcd 命名空间（`SHMRADIX_CLUSTER_ID` 这个前缀必须全 cluster 一致）                                                                                                                      |
| `peer_hit > 0` 但传输日志里 `bytes=0`                                                | 数据面没通：mooncake config 的 IP / device_name / metadata backend，或 Redis 通讯录                                                                                                                              |
| 某条请求发出去再也不返回（客户端超时，服务端不报错）                                                     | 某个 peer 传输 op 失败了，而 FlexKV 没有 graph 级别的 abort/timeout，那条请求就一直挂着。日志里找 zmq notify 超时和 mooncake 的握手重试；网卡不干净时可以先用 `PROTOCOL=tcp` 排除数据面                                                                   |
| 搬过来的 KV 解码出乱码                                                                  | block 字节布局不一致（模型 / `tokens_per_block` / KV dtype 各节点必须一样），或 Redis 被别的部署共用（§3.4 第一条）                                                                                                                  |
| 日志里一条 `PEERSSD2H` 都没有                                                          | JSON 里没开 `enable_p2p_ssd`；或 peer 的 CPU 层还留着同一段前缀 —— 取数顺序里 peer CPU 在 peer SSD 前面，轮不到 SSD（§3.4 第二条）                                                                                                   |
| 报"不同 tier 分到了不同 cluster rank"                                                  | etcd 里有陈旧的 peer key；清掉 `radix/<cluster_id>_<层>/` 下的 key 重启全部节点                                                                                                                                       |
| `Still waiting for GPU registrations` / `Error in scheduler loop: KeyError`    | `FLEXKV_DP_SIZE` 没跟 `--data-parallel-size` 对齐（vllm 在 DP 子进程里把 `data_parallel_size` 改成 1，宽度只能靠这个 env 传）                                                                                               |
| `no RDMA device with an ACTIVE port`                                           | 机器上没有可用 RDMA 口，这条路走不了（脚本里 `RDMA_DEV=` 可以强制指定）                                                                                                                                                        |
