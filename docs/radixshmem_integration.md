# FlexKV × radixshmem 集成（多 DP 路径）

将 FlexKV 的多 DP 路径从 "N 个 CE 客户端 → zmq → 单 KVServer 进程" 改造为 "N 个 CE 在自己地址空间里 attach 同一段共享 shm radix tree、并行查询，1 个共享 TE 通过 N 条 ShmChannel 接收 transfer graph"。结果是端到端 mean 时延快 ~2.4×、QPS 高 ~65%、命中率不变，并且消除 baseline 路径下高 QPS 时观察到的多种 hang/race。

## 目录

1. [radixshmem 是什么 / 解决了什么](#1-radixshmem-是什么--解决了什么)
2. [架构](#2-架构)
3. [编译与依赖](#3-编译与依赖)
4. [启用方法 / 配置项](#4-启用方法--配置项)
5. [对比 baseline 的方法](#5-对比-baseline-的方法)
6. [高压力测试结果](#6-高压力测试结果)
7. [正确性验证](#7-正确性验证)
8. [配套脚本清单](#8-配套脚本清单)
9. [已知坑](#9-已知坑)

---

## 1. radixshmem 是什么 / 解决了什么

### 1.1 原 multi-DP 路径的问题

FlexKV 原本的 multi-DP 路径走 `server_client_mode=True`：N 个推理引擎 scheduler 进程里各跑一个 `KVDPClient`，所有请求经 zmq PUSH/PULL 汇集到唯一的 `KVServer` 进程，由 `KVServer.run()` 单线程串行 dispatch 到 `KVTaskEngine` → `GlobalCacheEngine` → 单棵 RadixTree。

这个架构在 multi-DP 场景下有两个独立瓶颈：

- **N→1 query 漏斗**：所有 DP 的 `get_match` / `put_match` 都被串行化到一个 zmq 事件循环。`get_match` 单次 ~440 µs（含 zmq 往返 + 排队），高 QPS 下时延爆涨。
- **跨 DP 共享 cache 是 KVServer 进程的副产物**：cache 共享是好事（任意 DP put 进的都能被任意 DP get 出来），但代价是必须用单个 server 进程承接所有访问。

### 1.2 radixshmem 做了什么

[radixshmem](../../../radixshmem-zf/radixshmem)（同事写的库）把一棵 **radix tree** 整体放在 POSIX 共享内存里（`shm_open` + `mmap`），用 process-shared `pthread_rwlock` 协调，自带 buddy allocator + slot mempool。多个进程 attach 到同一段 shm 之后，可以**并行**做 prefix match / insert，**不需要任何中心进程**。

FlexKV 端做了三件事：

1. **新增 `CacheEngineRadixShmem` 后端**（`flexkv/cache/radix_shmem_engine.py`），跟原有 `CacheEngineAccel` 接口对齐，让 `GlobalCacheEngine` 透明 dispatch。
2. **去掉 KVServer**，多 DP 模式下每个 DP scheduler 进程在自己地址空间里持有 `KVTaskEngine`，所有 DP 通过 shm 共享同一棵 radix tree。
3. **新增 `ShmChannel` (SPSC ring + futex wake)**（`flexkv/transfer/shm_channel.py`）替换 zmq，让 N 个 CE 进程跟唯一的 TE 进程通讯。每 CE 一条独立 channel。

**两条独立维度的价值**：

| 维度 | baseline | shmradix |
|---|---|---|
| **Cache 跨 DP 共享** | ✅ 通过单 KVServer 共享 | ✅ 通过共享 shm tree |
| **Query 路径无 N→1 漏斗** | ❌ 所有查询过单 server | ✅ 每 DP 在自己进程并行 |

vllm 自带 `--enable-prefix-caching` 是 per-DP GPU-local 的，**两条维度都没有**（路由命中只有 1/DP 概率），所以在 multi-DP 高 QPS 下比 baseline FlexKV 还慢一个量级（见 §6 三路对比）。

---

## 2. 架构

```
DP scheduler 进程 0   ...   DP scheduler 进程 N-1
  ┌──────────────────┐         ┌──────────────────┐
  │ KVDPClient (本地)│         │ KVDPClient (本地)│
  │ KVTaskEngine     │   ...   │ KVTaskEngine     │
  │ GlobalCacheEngine│         │ GlobalCacheEngine│
  │  (radixshmem CE) │         │  (radixshmem CE) │
  └────────┬─────────┘         └────────┬─────────┘
           │                            │
        ┌──┴────────────────────────────┴──┐
        ▼                                  ▼
  /dev/shm/<id>_cpu (radixshmem)   ── 所有 CE 直接读写
  /dev/shm/<id>_ssd  (radixshmem)
  /dev/shm/<id>_remote (radixshmem)
        │                                  │
        ▼ ShmChannel × N (SPSC ring + futex)
  ┌──────────────────────────────────────────┐
  │  TransferEngine 进程（单实例）           │
  │  - StorageEngine (CPU buf / SSD / GPU)   │
  │  - 多通道 selector loop                  │
  └──────────────────────────────────────────┘
```

总进程数：`N (DP scheduler) + 1 (TE) + 1 (radix server，bootstrap 时由 DP-0 KVManager 启动 TreeServer，其余 DP attach 为 TreeClient)`。**KVServer 进程被完全去掉**。

跟原架构的对比：

| 组件 | baseline | shmradix |
|---|---|---|
| `KVServer` 子进程 | ✅ 单进程 / zmq 串行 | ❌ 不需要 |
| `KVTaskEngine` | 在 KVServer 进程里 | 每个 DP 进程里一份 |
| `GlobalCacheEngine` | 在 KVServer 进程里 | 每个 DP 进程里一份 |
| RadixTree（实际数据） | 在 KVServer 的 Python heap 里 | 在 POSIX shm，所有 DP attach |
| CE↔TE 通信 | mp.Pipe (single CE) 或 zmq+KVServer (multi DP) | **N 条 ShmChannel**（SPSC ring + futex） |
| dispatch | `KVServer.run()` 单线程 | `_TEShmDispatcher` 用 selector 在 N 个 channel 上 multiplex |

---

## 3. 编译与依赖

### 3.1 radixshmem 库

源码：[`radixshmem-zf/radixshmem`](../../../radixshmem-zf/radixshmem)（同事的仓库）。

预编译产物（容器测试场景）：
- `radixshmem/python/shmradix/_core.cpython-312-x86_64-linux-gnu.so`（Python 3.12，已 ship 在 bind-mount 路径下，不需要重新编译）

如果需要从源码编译：

```bash
cd radixshmem-zf/radixshmem
# 系统依赖：libxxhash-dev、liburing-dev（或用 mooncake_transfer_engine.libs 里的版本）
# 主仓库构建
mkdir build && cd build && cmake .. && make -j
# Python binding
cd ../python && pip install -e .
```

### 3.2 FlexKV

```bash
cd FlexKV
pip install -e . --no-build-isolation
```

依赖：Cython（构建时）、torch、numpy、xxhash、liburing、expiring_dict、zmq、redis 等。详见 `requirements.txt`。

### 3.3 容器化部署（vllm/vllm-openai 镜像）

vllm 官方镜像 `vllm/vllm-openai:vX.X.X` 不含 FlexKV 和它的 native 依赖。常用做法是 bind-mount 源码进容器，并跑一次性 setup 脚本：

```bash
# 创建容器（注意 --shm-size，见 §4.4 / §6.1 容量算法）
docker run -d --name dp-shm-test \
    --network host --shm-size=400g --gpus all \
    -v /path/to/FlexKV:/work/FlexKV \
    -v /path/to/radixshmem:/work/radixshmem \
    -v /path/to/data:/work/data \
    --entrypoint sleep \
    vllm/vllm-openai:vX.X.X \
    infinity

# 一次性 setup：符号链接 liburing、装 expiring_dict、验证 import
# 内容参考 benchmarks/two_phase/setup_container.sh
docker exec dp-shm-test bash /path/to/setup_container.sh
```

**setup_container.sh 做的事**：
1. 从 `mooncake_transfer_engine.libs/liburing-*.so.2` 软链接到 `/usr/lib/x86_64-linux-gnu/liburing.so.2`（FlexKV c_ext 在这找）
2. `pip install expiring_dict`
3. `python3 -c "import flexkv; import shmradix"` 验证

> 容器**每次重启后都要再跑一次 setup**（写在容器 rootfs，不在 bind mount）。

### 3.4 运行时 Python path

```bash
export PYTHONPATH=/work/FlexKV:/work/radixshmem/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

---

## 4. 启用方法 / 配置项

### 4.1 触发 shmradix 路径

```bash
export FLEXKV_RADIX_SHMEM=1
export FLEXKV_SHM_RADIX_ID=<unique-id>     # 同机多 vllm 实例时区分用
export FLEXKV_DP_SIZE=8                    # 必须跟 vllm --data-parallel-size 一致
export FLEXKV_CPU_CACHE_GB=200             # CPU cache 大小，见 §6.1
```

`FLEXKV_RADIX_SHMEM=1` 时 `KVManager` 走 `use_radix_shmem=True` 分支，每个 DP 进程内部构造 `KVTaskEngine`，绕过 `KVServer.create_server()`。Bootstrap DP（`instance_id=0 && dp_client_id=0`）顺带创建 shm radix regions（CPU/SSD/REMOTE 三段，按需）；其它 DP attach。

### 4.2 vllm 启动参数

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
| `--data-parallel-size N` | ✅ | 跟 `FLEXKV_DP_SIZE` 对齐 |
| `--no-enable-prefix-caching` | 强烈建议 | 关掉 vllm 自带 prefix cache，让 FlexKV 路径独占；否则两层 cache 混在一起难评估 |
| `--kv-transfer-config '{...}'` | ✅ | 启用 FlexKV connector |
| `--gpu-memory-utilization` | 看卡 | 留 KV cache 余量，见 §6.1 |

### 4.3 验证生效

启动几个 request 后，看 vllm log：

```bash
grep "FlexKV Hit Ratio" /path/to/vllm.log | tail -3
# 应输出类似：
# [FlexKV] Metric of Recent 100 Requests: ... FlexKV Hit Ratio: 87.45%, Get/Put Token Ratio: 105.32%.

grep "H2D transfer request" /path/to/vllm.log | tail -3
# 应输出类似：
# [FLEXKV] INFO ... H2D transfer request: 60 finished transfer data size: 0.013 GB ... transfer bandwidth: 6.58 GB/s
```

- `Hit Ratio` < 30 %（在你预期命中场景下）= cache 容量不够 / PUT 在 race
- `Get/Put Token Ratio` < 50 % = PUT 大量失败
- `H2D transfer request: ...` 出现 = KV 真实从 CPU 搬到 GPU（不是仅 metadata 命中）

### 4.4 可选诊断 env（**线上关掉**）

```bash
export FLEXKV_TIME_QUERY=50      # 每 50 次 get_match/put_match 聚合一行时延 log
export FLEXKV_NUM_LOG_INTERVAL_REQUESTS=100   # 命中率日志频率
```

---

## 5. 对比 baseline 的方法

### 5.1 切换 baseline / shmradix / vllm-only 三种模式

| 模式 | 关键开关 |
|---|---|
| **baseline**（FlexKV + zmq + 单 KVServer） | `FLEXKV_RADIX_SHMEM=0` + `--kv-transfer-config FlexKVConnectorV1` + `--no-enable-prefix-caching` |
| **shmradix**（FlexKV + 共享 shm tree） | `FLEXKV_RADIX_SHMEM=1` + 同上 |
| **vllm-only**（无 FlexKV，仅 vllm 原生 prefix cache） | 去掉 `--kv-transfer-config` + 加 `--enable-prefix-caching`（不是 `--no-enable-prefix-caching`） |

### 5.2 测试 workload — 两阶段 prefix-reuse

为暴露多 DP 高 QPS + 高 prefix 复用场景下三种路径的差异，我们用一个固定的两阶段 workload：

- **Phase 1（warmup）**：固定生成 8192 个 prompt（每个 128 token，token id 用固定 seed 随机生成），低并发 c=32 推进去。phase 1 结束后所有 prefix 都落到 CPU cache 里。
- **Phase 1 ↔ Phase 2 之间 sleep 15s** 让 FlexKV 异步 PUT 落库。
- **Phase 2（measure）**：同样 8192 个 prompt，每个在 phase 1 原 prompt 后接 10 个新 token（128 → 138 tokens，理论命中率 = 128/138 = 92.7%）。两组配置：
  - **R1**：open-loop QPS 目标 2048，`max_tokens=8`（含少量 decode）
  - **R2**：open-loop QPS 目标 4096，`max_tokens=1`（极限压力，最小化 decode 占比）

每条 phase 2 请求理论上能命中 phase 1 同 index 那条的全部 8 个 16-token block。

### 5.3 一键脚本

```bash
# 1. FlexKV baseline
WHICH=baseline REGEN_DATASET=1 CPU_CACHE_GB=200 \
PHASE2_QPS=4096 PHASE2_MAX_TOKENS=1 \
OUT_DIR=/tmp/run1 \
bash benchmarks/two_phase/run_two_phase_bench.sh

# 2. 重启容器清 /dev/shm（PID 1 不 reap，POSIX shm 不会自动释放）
docker stop <container> && docker start <container>
bash benchmarks/two_phase/setup_container.sh

# 3. FlexKV shmradix（同一个 OUT_DIR）
WHICH=shmradix REGEN_DATASET=0 CPU_CACHE_GB=200 \
PHASE2_QPS=4096 PHASE2_MAX_TOKENS=1 \
OUT_DIR=/tmp/run1 \
bash benchmarks/two_phase/run_two_phase_bench.sh

# 4. 重启容器 + setup
docker stop <container> && docker start <container>
bash benchmarks/two_phase/setup_container.sh

# 5. vllm-only（独立 OUT_DIR，一次跑 phase1+R1+R2）
OUT_DIR=/tmp/run1_vllm_only \
bash benchmarks/two_phase/run_vllm_only_bench.sh

# 6. 三路分析
python3 benchmarks/two_phase/analyze_three_way.py \
    --baseline-result   /tmp/run1/phase2-baseline.json \
    --baseline-log      /tmp/run1/vllm-baseline.log \
    --baseline-marker   /tmp/run1/phase2-baseline.start_marker \
    --shmradix-result   /tmp/run1/phase2-shmradix.json \
    --shmradix-log      /tmp/run1/vllm-shmradix.log \
    --shmradix-marker   /tmp/run1/phase2-shmradix.start_marker \
    --vllm-only-result  /tmp/run1_vllm_only/phase2-vllm_only-r2.json \
    --vllm-only-log     /tmp/run1_vllm_only/vllm-vllm_only.log \
    --vllm-only-marker  /tmp/run1_vllm_only/phase2-vllm_only-r2.start_marker \
    --label R2
```

整套耗时 35-45 分钟（3 次 vllm 启动 + 3 次 phase1+phase2 + 3 次容器重启）。

---

## 6. 高压力测试结果

测试环境：H20 单机 8 × GPU，2 TB RAM，容器 `--shm-size=400g`。Qwen3-8B（36 layers / 8 KV heads / head_dim=128 / bf16）。

### 6.1 容量账（先把数算清楚）

```
per_token_KV = num_kv_heads × head_dim × 2(K+V) × dtype_bytes × num_layers
             = 8 × 128 × 2 × 2 × 36 = 144 KB

per_block_KV = per_token_KV × block_size (16) = 2.25 MB
```

| 项 | 大小 | 备注 |
|---|---:|---|
| **CPU cache 配置** (`FLEXKV_CPU_CACHE_GB`) | 200 GB | |
| Phase 1 工作集 | 8192 × 128 × 144 KB | **147 GB**（75 % 占用，无 eviction） |
| Phase 2 新增 token 量 | < 2 GB | 多数是 partial block，PUT 量小 |
| GPU 每卡物理容量 | ~143 GB | H20 |
| vllm 占用 (`--gpu-memory-utilization 0.28`) | ~40 GB / 卡 | |
| 模型权重 (Qwen3-8B bf16) | ~16 GB / 卡 | |
| **每卡 GPU KV 空间** | ~24 GB | = 40 − 16 |
| 每卡 GPU KV 容量 | ~170 K tokens / ~10 600 blocks | 24 GB / 144 KB |
| 8 DP 合计 GPU KV | ~192 GB | 每 DP 独立、不共享 |

**关键观察**：phase 1 之后单卡 GPU 上有约 131K tokens 的 prefix KV（占 77 %）。vllm 自带 prefix cache **有充分容量本钱**，但 vllm v1 默认 DP 路由是**纯负载均衡**（不是 prefix-aware，见 `vllm/v1/engine/core_client.py:1337` 的 `DPLBAsyncMPClient.get_core_engine_for_request`），所以 phase 2 的请求落到原 DP 的概率 ≈ 1/8 = 12.5 %。这就是 vllm-only 路径性能塌的根因。

### 6.2 R2 高压极限（QPS=4096，max_tokens=1）

> **测量版本**：以下 baseline / shmradix 数据为删除 `flexkv/transfer/worker.py` 里 D2H/H2D worker 多余的 `torch.cuda.synchronize()` 之后重测（参见 §9 第 10 项）。这一行 sync 把本该流水线化的 transfer op 串行成 device-wide drain，对 baseline mean/p50/p95/p99 ~10% degrade，对 shmradix 尾延迟 ~12% degrade。`vllm-only` 不走 FlexKV transfer 路径，数据未变。

| Metric | baseline | shmradix | vllm-only | shmradix vs baseline |
|---|---:|---:|---:|---:|
| wall (s) | 4.23 | **2.56** | 13.22 | **−39.5 %** |
| **observed QPS** | 1938 | **3206** | **620** | **+65.4 %** |
| lat mean (ms) | 1728 | **709** | 5994 | **−59.0 %** |
| lat p50 (ms) | 1837 | **702** | 5894 | −61.8 % |
| lat p95 (ms) | 2406 | **951** | 10856 | −60.5 % |
| lat p99 (ms) | 2612 | **1116** | 11036 | −57.3 % |
| **FlexKV hit %** | **90.68** | **90.68** | n/a | 完全相同 |
| qtime get mean (µs) | 1859 | **151** | n/a | −91.9 % |
| qtime put mean (µs) | 932 | **68** | n/a | −92.7 % |

> 三个 backend 都是 8186 ok / 6 empty / 0 err。vllm-only 比 baseline 慢 3.5 ×，比 shmradix 慢 8.5 ×。
>
> qtime（match/put_match 时延）来自 FlexKV query 路径，与被删除的 transfer 端 sync 无关，沿用原 R2 数。

### 6.3 R1 标准压力（QPS=2048，max_tokens=8）

> R1 未重测；以下为含 sync 的旧数。删除 sync 对 R1 的影响方向与 R2 一致（baseline 全线 ~10%，shmradix 尾延迟 ~12%），相对关系（shmradix vs baseline）变化幅度有限。

| Metric | baseline | shmradix | vllm-only | shmradix vs baseline |
|---|---:|---:|---:|---:|
| wall (s) | 4.83 | 4.21 | 14.26 | −12.8 % |
| observed QPS | 1696 | 1944 | **574** | +14.6 % |
| lat mean (ms) | 933 | **165** | 7487 | **−82.3 %** |
| lat p50 (ms) | 978 | **124** | 7374 | −87.3 % |
| lat p95 (ms) | 1493 | **460** | 10866 | −69.2 % |
| FlexKV hit % | 90.68 | 90.68 | n/a | 完全相同 |
| qtime get mean (µs) | 1457 | 176 | n/a | −87.9 % |

### 6.4 三路对比小结

| 维度 | baseline | shmradix | vllm-only |
|---|---|---|---|
| Cache 储存位置 | CPU pinned + GPU | CPU pinned + GPU | **GPU only** |
| Cache 跨 DP 共享？ | ✅ 是（单 KVServer） | ✅ 是（POSIX shm tree） | ❌ **否** |
| Cache 容量上限 | 200 GB（CPU） | 200 GB（CPU） | ~24 GB / DP |
| Query 路径 | zmq → 单 server 串行 | per-DP 本地直查 shm | vllm 进程内 |
| Multi-DP 实际命中率 | 90.68 % | **90.68 %** | **~12.5 %（1/DP 上限）** |
| QPS 上限（R2） | ~1940 | **~3210** | ~620 |

### 6.5 关键结论

1. **shmradix 和 baseline 命中率完全相同（90.68 %）** —— 二者之间的差距全部来自 query 路径
2. **baseline QPS 上限 ≈ 1940**（R2 target 4096 跑 1938；R1 target 2048 跑 1696）—— 单 KVServer 串行处理的硬上限
3. **shmradix QPS 上限 ≈ 3210**（R2），是 baseline 的 1.65 ×；mean 时延是 baseline 的 0.41 ×（2.44 × 更快）
4. **vllm-only QPS 上限 ≈ 620**（R1/R2 都是），是 baseline 的 0.32 ×、shmradix 的 0.19 × —— 不是因为 vllm 慢，是 multi-DP 路由让 GPU prefix cache 命中率塌
5. **baseline 的 `get_match` 时延随负载剧增**（R1: 1457 µs → R2: 1859 µs），shmradix **完全不随负载变化**（R1: 176 µs → R2: 151 µs）—— scaling 本质区别
6. **FlexKV 在 multi-DP serving 下的两条独立价值**：(a) cache 跨 DP 共享带来命中率优势；(b) query 路径去 N→1 漏斗带来时延/QPS 优势。shmradix 在 baseline (a) 的基础上再加 (b)

---

## 7. 正确性验证

用真实 ShareGPT 多轮对话数据集（50 conversations × 3 turns = 150 个 request 每后端）。

### 7.1 输出可用性

verifier 跑五项垃圾检查：replacement char、控制字符、低熵、private-use unicode、空串。

| 后端 | turn | ok | empty | err | **garbled** | 平均长度 (chars) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 / 2 / 3 | 50 / 50 / 50 | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** | 305 / 314 / 312 |
| **shmradix** | 1 / 2 / 3 | **50 / 50 / 50** | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** | 303 / 315 / 313 |

两边都 150/150 ok，**0 garbled / 0 empty / 0 error**。抽样输出都是切题英文，turn 2/3 显式延续 turn 1 上下文（"continue explaining"、"asking about the get(i) method I provided earlier"），KV cache 命中后模型行为完全正常。

### 7.2 KV cache 真实复用（H2D 证据）

从 FlexKV worker 日志抓 `H2D transfer request: N finished transfer data size: X GB transfer bandwidth: Y GB/s`，每行 = 一次真实 CPU→GPU 内存拷贝。

| 后端 | H2D 事件数 | 总传输 (GB) | 平均带宽 (GB/s) |
|---|---:|---:|---:|
| baseline | 81 | **6.510** | 26.74 |
| shmradix | 92 | **6.510** | 24.90 |

两边**总搬运字节完全相等**（同 workload 同 cache → 同物理 IO），证明 FlexKV 命中**不是仅 metadata 命中**，是真把 KV bytes 从 CPU pinned buffer cudaMemcpy 回 GPU 槽。带宽 25-27 GB/s 是 H20 PCIe 正常范围。

### 7.3 跨后端输出一致性

| 维度 | 值 |
|---|---|
| 比较的 (conv, turn) 单元格 | 150 |
| 文本完全相同 | 107 (71.3 %) |
| 不同但都合理 | 43 (28.7 %) |
| 乱码 | **0** |

差异都是 CUDA kernel 非确定性 + DP-server 自然负载均衡导致的换词（同 backend 重跑两次也有 ~10-15 % 漂移）。所有差异样本都是**前缀几乎一致，后段分支选词不同**。

---

## 8. 配套脚本清单

测试基础设施全部在 `benchmarks/two_phase/`：

| 文件 | 作用 |
|---|---|
| `setup_container.sh` | 容器重建/重启后一次性 setup（liburing 软链接、装 expiring_dict、import 验证） |
| `gen_dataset.py` | 生成 phase1/phase2 token JSON（参数全 CLI 化：n-convs、seq-len、extension-len、seed 等） |
| `run_phase1_warmup.py` | phase 1 closed-loop ThreadPool driver |
| `run_phase2_bench.py` | phase 2 driver，支持 `--mode qps`（open-loop asyncio）或 `--mode concurrency`（closed-loop） |
| `correctness_multiturn.py` | ShareGPT 多轮 driver（正确性验证用） |
| `verify_correctness.py` | 五项垃圾检查 + H2D 字节统计 + 跨后端 diff |
| `run_two_phase_bench.sh` | FlexKV 路径一键 orchestrator（`WHICH=baseline/shmradix/both`） |
| `run_vllm_only_bench.sh` | vllm-only 路径一键 orchestrator（关 FlexKV、开 vllm 原生 prefix cache） |
| `analyze_phase2.py` | baseline vs shmradix 两路对比分析 |
| `analyze_three_way.py` | baseline / shmradix / vllm-only 三路对比分析 |
| `README.md` | 工具自身文档（参数表 + 容量陷阱 + 推荐 workflow） |

---

## 9. 已知坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 1. 容器 PID 1 是 `sleep infinity` 不 reap zombie | 两个 vllm session 间 `/dev/shm` 卡满（~200 GB 残留） | `docker stop && start` 容器，然后再跑 `setup_container.sh` |
| 2. `pkill -f "vllm serve"` 杀不全 | 留下 `VLLM::EngineCore_*` 子进程 | 同时 `pkill -f "VLLM::"` |
| 3. vllm v1 默认路由不是 prefix-aware | r2 落 r1 当时 DP 的概率仅 1/N | 这正是 shmradix 的价值；不要试图修 vllm 端 |
| 4. vllm 官方镜像不含 FlexKV | 重建容器后 import 失败 | 跑 `setup_container.sh`，或在 Dockerfile 里加 install 步骤 |
| 5. `docker stop` 报 "did not receive an exit event" | 看着像没停 | 等 30-90 秒，容器自然变 Exited(137) |
| 6. vllm 不把所有 env 传给 engine 子进程 | 自定义 `FLEXKV_*` env 在 engine 进程里 `os.getenv` 拿不到 | env 的 default 行为只继承一部分；FlexKV 已用 `FlexKVConfig.from_env` 序列化进 config；若新加 env 需要 lazy resolve |
| 7. `--shm-size` 一旦设定，container restart 不能改 | 修不了 | `docker rm` 重建 |
| 8. `FLEXKV_SHM_RADIX_ID` 重复 | 多个 vllm 实例 attach 同一段 shm tree，状态串了 | 每个实例用唯一 ID |
| 9. CPU cache 跨 DP 共享、但 GPU KV cache 是 per-DP | 容易混淆容量估算 | 见 §6.1，CPU 是 200 GB 共池，GPU 是 8 × 24 GB 独立 |
| 10. `flexkv/transfer/worker.py` 里曾在 `_transfer_impl` 后 `torch.cuda.synchronize()` | 把 D2H/H2D op 串行成 device-wide drain，R2 mean / p95 / p99 各 ~10-12% degrade | device-wide sync 已移除；`GPUCPUTransferWorker._transfer_impl` 现把 `sync=True` 下推给 C++ `transfer_kv_blocks`，收尾只做 stream-scoped 的 `cudaStreamSynchronize(stream)`（`csrc/transfer.cu:300`），不再 drain 整个 device。下游 worker 队列自带 stream-aware 排序，多余的 device sync 没有保护任何 invariant。NIXL worker 的 sync（`worker.py:2411`，由 PR #142 引入）属于另一条路径，与本项无关 |

---

## 附录：本次工作中修复的几处 race

为达到上面的 0-hang 表现，本轮调试一并修复了 4 类 race（每条都在 `flexkv/...` 里有对应改动）：

| race | 修复 |
|---|---|
| `batch merged graph_id` 跨 DP 撞 ID（`merge_to_batch_graph` 用 batch_id 覆盖全局 disjoint 分配） | 删除 `set_graph_id(batch_id)`，让 merged graph 用 `TransferOpGraph()` 的 disjoint range（`flexkv/common/transfer.py:399`） |
| `ShmChannel.result_send` 满时抛异常 → TE result 线程死 → 丢 completion | 改成永久 spin + 周期错误日志（`flexkv/transfer/shm_channel.py:344`），ring 容量 256 → 1024 |
| `CacheEngineRadixShmem.match(lock=True)` 的 ref 每次泄漏一次 → 长跑后节点钉死无法 evict | 引入 `MatchResultAccel.pre_locked_node` 字段，4 个 `_impl` 方法所有 success/early-return 路径显式释放（`flexkv/cache/cache_engine.py`） |
| `KVTaskManager._pending_release_tasks` 在多 DP 共用 KVTaskEngine 下被任意 DP 抢先清理 → `NOTFOUND` race | 上游 PR #164（commit `0d6b1ed`）通过 `shed_heavy_resources()` + owner-observed release 修复 |
| `submit_send` 只 bump channel wake、TE idle 时等的是 ctrl wake → 多等 100ms futex timeout | `submit()` / `submit_batch()` 后显式 `_ctrl.notify()`（`flexkv/transfer/shm_channel_handle.py`） |

这些都已 commit 在 `main` / 当前分支。

---

## 联系

- 主仓库：[`FlexKV`](https://github.com/taco-project/FlexKV)
- radixshmem 依赖：（参见各项目内部联系方式）
- 完整调试历史：本仓库 git log + 本目录下其它 dp_shmradix_*.md 文档
