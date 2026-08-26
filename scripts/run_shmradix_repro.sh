#!/usr/bin/env bash
# =============================================================================
# FlexKV × radixshmem: CROSS-NODE KV reuse test (doc radixshmem_cross_node.md §4.1),
# driven through vLLM.
#
# Simulates TWO FlexKV nodes on ONE host, each running 4 DP engines:
#
#     node 0 (radix rank 0)  :31000         node 1 (radix rank 1)  :31001
#     vllm serve --data-parallel-size 4     vllm serve --data-parallel-size 4
#     CUDA_VISIBLE_DEVICES=0,1,2,3          CUDA_VISIBLE_DEVICES=4,5,6,7
#      ├─ DP0  ├─ DP1  ├─ DP2  ├─ DP3        ├─ DP0  ├─ DP1  ├─ DP2  ├─ DP3
#        \___ one shm radix tree ___/          \___ one shm radix tree ___/
#        \___ one transfer engine ___/         \___ one transfer engine ___/
#                        \                    /
#                         etcd rendezvous (world_size=2)
#                         RDMA remote tree walk   <- control plane
#                         mooncake + Redis        <- data plane
#
# WHY FLEXKV_DP_SIZE IS REQUIRED, NOT COSMETIC:
#   FlexKV keeps the DP engines of one node apart by
#   `dp_client_id = instance_id*dp_size + dp_rank`, and sizes the node's single
#   shared transfer engine from `total_clients = instance_num*dp_size`. For
#   non-MoE models vLLM runs each DP rank as a fully independent engine and
#   RESETS that child's data_parallel_size to 1 and data_parallel_rank to 0,
#   keeping the real rank only in data_parallel_index (vllm/v1/engine/core.py,
#   "Non-MoE DP ranks are completely independent, so treat like DP=1").
#   FlexKV recovers the rank from data_parallel_index, but the WIDTH survives
#   nowhere in parallel_config -- hence this env var (see
#   flexkv/integration/config.py:_resolve_vllm_dp_size).
#   Without it every DP engine computes dp_client_id=0 and collides on:
#   shm-radix bootstrap ownership, transfer-engine channel 0, the gpu_register
#   endpoint, and overlapping graph/op id ranges -- surfacing as
#   "Error in scheduler loop: KeyError" plus "Still waiting for GPU
#   registrations". `total_gpus` also stays 1, which turns off the
#   CUDA_VISIBLE_DEVICES clearing the TE subprocess needs to open every DP
#   rank's IPC handles (flexkv/transfer_manager.py:1209).
#
# WHY THE WORKLOAD PROVES *CROSS-NODE* REUSE:
#   Phase 1 sends every prompt only to node 0's port. Phase 2 replays the SAME
#   prompts only to node 1's port. vLLM prefix caching is off, and
#   node 1's own CPU tier never saw these tokens, so any phase-2 hit must have
#   been found by a remote radix tree walk into node 0 and fetched over the peer
#   data path. A local-only regression scores ~0% in phase 2.
#
# WHY IT ALSO CHECKS THE KV, NOT JUST THE PLUMBING:
#   Moving bytes is not the same as moving the RIGHT bytes. Phase 2 answers the
#   prompts on node 1 with the prefix pulled off node 0; phase 3 answers the SAME
#   prompts on node 0, which holds that prefix itself. Same model, same prompt,
#   same reused blocks, temperature=0 -- the only difference is where the KV came
#   from, so the two must produce identical completions. If they differ, the KV
#   that crossed the network is not the KV node 0 stored. Completions land in
#   $LOG_DIR/output_check.json.
#   Scope: end-to-end evidence. For a per-(layer, block) byte comparison of the
#   peer path, see tests/test_e2e_radix_peer_data.py.
#
# Config via env (defaults shown):
#   VENV=/root/vllm-env  MODEL=Qwen/Qwen3-0.6B  PORT_BASE=31000  DP=4
#   CPU_CACHE_GB=1  SSD_CACHE_GB=8  FLOOD_PROMPTS=40  NUM_PROMPTS=8  HIT_MIN=50
#   SHM_ID=xnode_<pid>  LOG_DIR=/tmp/flexkv-xnode-<pid>  READY_TIMEOUT=900
#   RDMA_DEV=<first ACTIVE>  ETCD_PORT=<free>  REDIS_PORT=<free>
#   PROTOCOL=rdma            # mooncake data-plane protocol: rdma | tcp
#   OUTPUT_CHECK=strict      # peer-KV vs local-KV check: strict | warn | off
#   RHT_SLOTS=4              # RHT slots per bucket: 1 | 2 | 4 | 8
#
# THE DEFAULT RUN COVERS BOTH TIERS, AND AIMS PHASE 2 AT THE PEER'S SSD:
#   Each tier rendezvouses in its own etcd namespace, so a node running CPU + SSD
#   registers both without one overwriting the other -- covered only when both are
#   on. `_shm_get_spans` ranks a peer's CPU tier ahead of its SSD tier, so a
#   surviving peer CPU copy would serve every block and PEERSSD2H would never be
#   asked; hence the small CPU pool and the flood phase that evicts it (below).
#   SSD_CACHE_GB=0 falls back to a CPU-only run over PEERH2H.
#
# Usage:  bash scripts/run_shmradix_repro.sh
#         DP=2 bash scripts/run_shmradix_repro.sh   # 2 nodes x 2 DP = 4 GPUs
#         HIT_MIN=100 bash scripts/run_shmradix_repro.sh
#         OUTPUT_CHECK=warn bash scripts/run_shmradix_repro.sh
#         SSD_CACHE_GB=0 bash scripts/run_shmradix_repro.sh  # CPU tier only
#         PROTOCOL=tcp bash scripts/run_shmradix_repro.sh    # no RDMA NIC
#
# Runs from anywhere -- it drives vLLM over HTTP and needs no repo-relative path.
# Nothing here depends on the other scripts in this directory.
# =============================================================================
set -uo pipefail

VENV="${VENV:-/root/vllm-env}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT_BASE="${PORT_BASE:-31000}"
DP="${DP:-4}"                       # engines per node
NODES=2                             # radix world_size; the point of the test
SSD_CACHE_GB="${SSD_CACHE_GB:-8}"
if (( SSD_CACHE_GB > 0 )); then
  # A peer SSD read only serves blocks the peer's CPU tier no longer holds, so the
  # CPU pool is kept small enough for FLOOD_PROMPTS to evict.
  CPU_CACHE_GB="${CPU_CACHE_GB:-1}"
  FLOOD_PROMPTS="${FLOOD_PROMPTS:-40}"
else
  CPU_CACHE_GB="${CPU_CACHE_GB:-8}"
  FLOOD_PROMPTS=0
fi
SHM_ID="${SHM_ID:-xnode_$$}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
LOG_DIR="${LOG_DIR:-/tmp/flexkv-xnode-$$}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
HIT_MIN="${HIT_MIN:-50}"
PROTOCOL="${PROTOCOL:-rdma}"
# How to treat a phase-2 (peer KV) completion that differs from its phase-3 (local
# KV) counterpart: strict fails the run, warn reports it and keeps going -- for
# hosts whose two node halves are not kernel-identical (mixed GPU models), where
# even correct bytes can decode differently -- and off skips the comparison.
OUTPUT_CHECK="${OUTPUT_CHECK:-strict}"
# RHT slots per bucket. At shmradix's default of 1 a bucket write is a blind
# overwrite, so a reader that holds part of the prefix republishes itself over the
# peer's entry and no peer stays routable. This run survives 1 only because
# phase 2's reader holds none of it -- production does not, so mirror production.
RHT_SLOTS="${RHT_SLOTS:-4}"
# Hit-ratio logs are emitted every N requests per engine. Each engine sees
# NUM_PROMPTS/DP requests per phase, so this makes the last window per engine
# contain phase-2 requests only -- otherwise it straddles both phases and
# reports roughly half the real reuse rate.
LOG_INTERVAL=$(( NUM_PROMPTS / DP )); (( LOG_INTERVAL > 0 )) || LOG_INTERVAL=1

TOTAL=$(( NODES * DP ))   # engines overall; API ports are one per NODE
PIDS=(); PGIDS=()
ETCD_PID=""; REDIS_PID=""
# CacheConfig rejects an SSD tier that is not strictly larger than the CPU one
# (flexkv/common/config.py:876), and the eviction workload needs the headroom too.
(( SSD_CACHE_GB == 0 || SSD_CACHE_GB > CPU_CACHE_GB )) || \
  { echo "[xnode] ERROR: SSD_CACHE_GB=$SSD_CACHE_GB must exceed CPU_CACHE_GB=$CPU_CACHE_GB"; exit 2; }
# shmradix takes anything else back to 1 with only a WARN on rank 0's stderr, so
# catch it here rather than let the run quietly lose cross-node reuse.
case "$RHT_SLOTS" in
  1|2|4|8) ;;
  *) echo "[xnode] ERROR: RHT_SLOTS=$RHT_SLOTS must be 1, 2, 4 or 8"; exit 2 ;;
esac
CONF_DIR="$LOG_DIR/conf"

log() { echo "[xnode] $*"; }
die() { log "ERROR: $*"; exit 2; }

free_port_block() { python - "$1" <<'PY'
import socket, sys
n = int(sys.argv[1])
for _ in range(64):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); base = s.getsockname()[1]; s.close()
    held = []
    try:
        for off in range(n):
            k = socket.socket(); k.bind(("127.0.0.1", base + off)); held.append(k)
        print(base); break
    except OSError:
        continue
    finally:
        for k in held: k.close()
else:
    raise SystemExit("no free port block")
PY
}

# --- kill engine/TE procs orphaned by THIS run (the pgid kill in cleanup() only
#     reaches processes that stayed in the launched groups; /proc scan => no pgrep
#     self-match) ---
#     Identity comes from the environ, not the cmdline: every engine, worker and
#     TE child inherits FLEXKV_SHM_RADIX_ID from the launch, and no other job has
#     this run's value. Matching on the cmdline instead would reap unrelated work
#     from the same venv -- the interpreter path "$VENV/bin/python" itself contains
#     "vllm", so a plain *vllm* pattern hits every pytest run on the box.
#     The value is per-node ("${SHM_ID}_r<node>"), so match on the prefix.
reap_run_procs() {
  local p exe
  for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null) || continue
    [[ "$exe" == "$VENV/"* ]] || continue
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null \
      | grep -qF "FLEXKV_SHM_RADIX_ID=$SHM_ID" || continue
    kill -9 "$p" 2>/dev/null
  done
}

cleanup() {
  local rc=$? g i p
  trap - EXIT INT TERM          # disarm to avoid re-entry
  echo; log "=== graceful cleanup ==="
  # 1) the vLLM process groups: SIGTERM, wait, then SIGKILL
  for g in "${PGIDS[@]:-}"; do [[ -n "$g" ]] && kill -s TERM "-$g" 2>/dev/null; done
  for _ in $(seq 1 15); do
    local alive=0
    for g in "${PGIDS[@]:-}"; do [[ -n "$g" ]] && kill -0 "-$g" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break; sleep 1
  done
  for g in "${PGIDS[@]:-}"; do [[ -n "$g" ]] && kill -s KILL "-$g" 2>/dev/null; done
  # 2) free the API ports
  for ((i=0; i<NODES; i++)); do fuser -k "$(( PORT_BASE + i ))/tcp" 2>/dev/null; done
  # 3) reap engine / TE procs of THIS run that outlived their process group
  reap_run_procs
  # 4) stop the NVIDIA MPS daemon. Graceful quit first so the control daemon
  #    reaps its mps-server children; then a comm-based kill for any straggler --
  #    MPS procs have an EMPTY /proc/pid/cmdline so `pgrep -f` misses them and
  #    their comm is truncated to "nvidia-cuda-mps". An mps-server whose parent
  #    already died lingers as a harmless zombie when the container's PID 1 does
  #    not reap; it holds no GPU and clears on container restart.
  echo quit | nvidia-cuda-mps-control 2>/dev/null
  pkill -9 nvidia-cuda-mps 2>/dev/null
  # 5) the POSIX shm radix / TE regions for this run, plus the ssd tier's block
  #    files -- those are GBs; the logs next to them are kept
  rm -f /dev/shm/*flexkv* /dev/shm/*shmradix* /dev/shm/*"${SHM_ID}"* 2>/dev/null
  rm -f /tmp/flexkv_"${SHM_ID}"* 2>/dev/null
  rm -rf "$LOG_DIR"/ssd_r* 2>/dev/null
  # 6) the etcd + redis this run started, last so the engines are gone before
  #    their rendezvous and address book disappear. Both are private to the run,
  #    so their whole state goes with them -- no per-key deletion needed.
  # `wait` reaps them here, so bash does not print its own "Killed" job notice.
  for p in "$ETCD_PID" "$REDIS_PID"; do
    [[ -n "$p" ]] && { kill -9 "$p" 2>/dev/null; wait "$p" 2>/dev/null; }
  done
  rm -rf "$LOG_DIR/etcd.data" 2>/dev/null
  log "GPU mem after cleanup: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  log "logs kept in $LOG_DIR"
  log "cleanup done (exit $rc)"
  exit "$rc"
}
trap cleanup EXIT INT TERM

# ---------------- preflight ----------------
mkdir -p "$CONF_DIR" || die "cannot create $CONF_DIR"

[[ -x "$VENV/bin/python" ]] || die "venv not found at $VENV (set VENV=...)"
set +u; source "$VENV/bin/activate"; set -u
export LD_LIBRARY_PATH="$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

VLLM_VER=$(python -c "import vllm;print(vllm.__version__)" 2>/dev/null || echo "?")
NGPU=$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
log "vLLM=$VLLM_VER torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null) GPUs=$NGPU"

python -c "import torch; from flexkv import c_ext; import shmradix, mooncake.engine" \
  || die "flexkv / shmradix / mooncake import failed"
python -c "from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory as F;import sys;sys.exit(0 if 'FlexKVConnectorV1' in (getattr(F,'_registry',{}) or {}) else 1)" \
  && log "FlexKVConnectorV1 registered" \
  || log "WARN: FlexKVConnectorV1 NOT registered in vLLM $VLLM_VER (needs >=0.17.2 built-in, or a connector patch)"
case "$VLLM_VER" in 0.2[2-9].*|0.[3-9][0-9].*|1.*) log "WARN: vLLM $VLLM_VER is newer than the tested 0.17-0.21 range";; esac

(( NGPU >= TOTAL )) || die "need $TOTAL GPUs for ${NODES} nodes x ${DP} DP, host has $NGPU (lower DP=)"

# The radix control plane walks the peer tree over RDMA, so an ACTIVE port is a
# hard requirement -- not a performance choice.
if [[ -z "${RDMA_DEV:-}" ]]; then
  for d in /sys/class/infiniband/*; do
    [[ -e "$d" ]] || break
    for p in "$d"/ports/*; do
      grep -q ACTIVE "$p/state" 2>/dev/null && { RDMA_DEV=$(basename "$d"); break 2; }
    done
  done
fi
[[ -n "${RDMA_DEV:-}" ]] || die "no RDMA device with an ACTIVE port (cross-node needs one; set RDMA_DEV= to force)"
log "RDMA device: $RDMA_DEV  gid_idx=${FLEXKV_RADIX_GID_IDX:-3}"

# ---------------- private etcd (cluster rendezvous) ----------------
# Started here rather than shared, so a leftover peer entry from another run can
# never join this cluster: the state dies with the instance.
command -v etcd >/dev/null 2>&1 || die "etcd not on PATH (world_size>1 bootstrap rendezvouses through it)"
ETCD_PORT="${ETCD_PORT:-$(free_port_block 2)}" || die "no etcd port block"
ETCD="http://127.0.0.1:$ETCD_PORT"
ETCD_PEER="http://127.0.0.1:$(( ETCD_PORT + 1 ))"
rm -rf "$LOG_DIR/etcd.data"
etcd --name xnode --data-dir "$LOG_DIR/etcd.data" \
  --listen-client-urls "$ETCD" --advertise-client-urls "$ETCD" \
  --listen-peer-urls "$ETCD_PEER" --initial-advertise-peer-urls "$ETCD_PEER" \
  --initial-cluster "xnode=$ETCD_PEER" >"$LOG_DIR/etcd.log" 2>&1 &
ETCD_PID=$!
for _ in $(seq 1 30); do
  curl -s --max-time 2 "$ETCD/version" >/dev/null 2>&1 && break; sleep 1
done
curl -s --max-time 5 "$ETCD/version" >/dev/null 2>&1 \
  || { tail -20 "$LOG_DIR/etcd.log"; die "etcd did not come up on $ETCD"; }
log "etcd ready on $ETCD (pid $ETCD_PID): $(curl -s --max-time 5 "$ETCD/version")"

# ---------------- private redis (address book + mooncake metadata) ----------------
# Carries both FlexKV's meta:/node: keys and mooncake's metadata channel; private
# for the same reason as etcd -- a stale peer address there points the data plane
# at a dead buffer.
command -v redis-server >/dev/null 2>&1 && command -v redis-cli >/dev/null 2>&1 \
  || die "redis-server / redis-cli not on PATH (peer data-plane address book)"
REDIS_PORT="${REDIS_PORT:-$(free_port_block 1)}" || die "no free redis port"
redis-server --port "$REDIS_PORT" --save '' --appendonly no --daemonize no \
  --bind 127.0.0.1 >"$LOG_DIR/redis.log" 2>&1 &
REDIS_PID=$!
for _ in $(seq 1 30); do
  redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG && break; sleep 1
done
redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG \
  || { tail -20 "$LOG_DIR/redis.log"; die "redis did not come up on $REDIS_PORT"; }
log "redis ready on 127.0.0.1:$REDIS_PORT (pid $REDIS_PID)"

# ---------------- per-node config files ----------------
# enable_p2p_cpu / redis / zmq have NO env vars: load_user_config_from_env()
# (flexkv/common/config.py:934) does not read them. The JSON path does --
# load_user_config_from_file + update_default_config_from_user_config copies
# them into CacheConfig (config.py:1201-1215). That is the only way to open the
# peer data plane through vLLM.
ZMQ_BASE=$(free_port_block $(( NODES * 4 ))) || die "no zmq port block"
for ((R=0; R<NODES; R++)); do
  MC_PORT=$(free_port_block 1)
  # metadata_backend=redis needs a mooncake built with -DUSE_REDIS=ON; a build
  # without it aborts at "Unable to find metadata storage plugin redis".
  cat > "$CONF_DIR/mooncake_r$R.json" <<EOF
{"engine_ip":"127.0.0.1","engine_port":$MC_PORT,"metadata_backend":"redis",
 "metadata_server":"redis://127.0.0.1:$REDIS_PORT","metadata_server_auth":"",
 "protocol":"$PROTOCOL","device_name":"$([[ "$PROTOCOL" == rdma ]] && echo "$RDMA_DEV")"}
EOF
  # local_zmq_port needs its successor free too (PEERSSD2H uses port+1), so the
  # two nodes get non-adjacent bases.
  # Each node needs its OWN ssd dir: the two share a host, and one dir would have
  # them writing each other's block files.
  SSD_DIR="$LOG_DIR/ssd_r$R"; rm -rf "$SSD_DIR"; mkdir -p "$SSD_DIR"
  cat > "$CONF_DIR/flexkv_r$R.json" <<EOF
{"cpu_cache_gb": $CPU_CACHE_GB,
 "ssd_cache_gb": $SSD_CACHE_GB,
 "ssd_cache_dir": "$SSD_DIR",
 "enable_p2p_cpu": true,
 "enable_p2p_ssd": $([[ "$SSD_CACHE_GB" -gt 0 ]] && echo true || echo false),
 "redis_host": "127.0.0.1",
 "redis_port": $REDIS_PORT,
 "local_ip": "127.0.0.1",
 "local_zmq_ip": "127.0.0.1",
 "local_zmq_port": $(( ZMQ_BASE + R * 4 ))}
EOF
  log "node $R: mooncake engine_port=$MC_PORT zmq_base=$(( ZMQ_BASE + R * 4 ))$( (( SSD_CACHE_GB > 0 )) && echo " ssd=$SSD_DIR" )"
done

# pre-clean stale state
for ((i=0; i<NODES; i++)); do fuser -k "$(( PORT_BASE + i ))/tcp" 2>/dev/null; done
rm -f /dev/shm/*flexkv* /dev/shm/*shmradix* 2>/dev/null
echo quit | nvidia-cuda-mps-control 2>/dev/null   # release GPU held by a stale daemon
pkill -9 nvidia-cuda-mps 2>/dev/null

# ---------------- launch 2 nodes x DP engines ----------------
# Shared by all engines: SHMRADIX_CLUSTER_ID, the base of the etcd namespaces they
# meet in -- FlexKV appends the tier (radix/<id>_cpu/, radix/<id>_ssd/) so a node's
# tiers cannot overwrite each other's peer entry -- and SHMRADIX_RHT_SLOTS, which
# rank 0's value settles cluster-wide anyway.
# Distinct per node: FLEXKV_SHM_RADIX_ID (names its shm regions),
# SHMRADIX_NODE_NAME (its etcd identity, which would otherwise come from the bind
# IP both nodes share), the TE ipc endpoint (and the gpu_register port derived
# from it), the config files, the GPU set, the API port. Distinct per engine
# within a node: dp_rank (from vLLM's data_parallel_index) and dp_client_id.
# MC_FORCE_TCP is what actually makes PROTOCOL=tcp true: mooncake ignores the
# config's "protocol" and installs its rdma transport whenever it discovers an HCA.
if [[ "$PROTOCOL" == tcp ]]; then export MC_FORCE_TCP=1; fi
export FLEXKV_RADIX_SHMEM=1 \
       FLEXKV_RADIX_WORLD_SIZE="$NODES" \
       FLEXKV_RADIX_REGISTRY="$ETCD" \
       SHMRADIX_CLUSTER_ID="$SHM_ID" \
       SHMRADIX_RHT_SLOTS="$RHT_SLOTS" \
       FLEXKV_RADIX_RDMA_DEV="$RDMA_DEV" \
       FLEXKV_TRACE_RADIX_PEER=1 \
       FLEXKV_DP_SIZE="$DP" \
       FLEXKV_ENABLE_MPS=0 \
       FLEXKV_NUM_LOG_INTERVAL_REQUESTS="$LOG_INTERVAL"

log "launching $NODES nodes x $DP DP = $TOTAL engines, API ports $PORT_BASE-$(( PORT_BASE + NODES - 1 ))"
for ((R=0; R<NODES; R++)); do
  # The node's GPUs, in ascending order and IDENTICAL for every process of the
  # node: the TE subprocess clears CUDA_VISIBLE_DEVICES to open all DP ranks'
  # IPC handles, so a per-engine or rotated list would renumber devices between
  # the registering worker and the TE.
  gpus=$(seq -s, $(( R * DP )) $(( R * DP + DP - 1 )))
  lf="$LOG_DIR/n${R}.log"; : > "$lf"
  CUDA_VISIBLE_DEVICES="$gpus" \
  FLEXKV_SHM_RADIX_ID="${SHM_ID}_r${R}" \
  SHMRADIX_NODE_NAME="${SHM_ID}_r${R}" \
  FLEXKV_RADIX_RPC_ADDRESS=127.0.0.1 \
  FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_${SHM_ID}_r${R}" \
  FLEXKV_CONFIG_PATH="$CONF_DIR/flexkv_r$R.json" \
  MOONCAKE_CONFIG_PATH="$CONF_DIR/mooncake_r$R.json" \
  setsid vllm serve "$MODEL" \
      --tensor-parallel-size 1 --data-parallel-size "$DP" --enforce-eager \
      --port "$(( PORT_BASE + R ))" --max-num-seqs 32 \
      --max-num-batched-tokens 4096 \
      --max-model-len 4096 --gpu-memory-utilization 0.25 \
      --no-enable-prefix-caching --trust-remote-code \
      --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' \
      >> "$lf" 2>&1 &
  PIDS+=($!); PGIDS+=("$(ps -o pgid= -p $! 2>/dev/null | tr -d ' ')")
  log "  node $R: GPUs $gpus -> port $(( PORT_BASE + R )), log $lf"
done

# ---------------- wait for readiness ----------------
# RadixServer.bootstrap() is COLLECTIVE: node 0's and node 1's instance-0
# engines each block inside it until all world_size=2 members arrive. So both
# nodes must be up before either finishes init, and a single node dying during
# startup hangs the other until FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC (default 120)
# expires. The wait below therefore polls every engine and fails fast if any
# process exits.
log "waiting for $NODES nodes to become healthy (timeout ${READY_TIMEOUT}s)..."
ready=0
for ((t=0; t<READY_TIMEOUT; t+=5)); do
  n=0
  for ((i=0; i<NODES; i++)); do
    curl -sf -o /dev/null --max-time 3 "http://localhost:$(( PORT_BASE + i ))/health" 2>/dev/null && n=$(( n + 1 ))
  done
  [[ "$n" == "$NODES" ]] && { ready=1; break; }
  for p in "${PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null || { log "ERROR: a node exited during startup ($n/$NODES healthy)"; grep -m5 -iE "error|Traceback|abort" "$LOG_DIR"/*.log | head -20; exit 1; }
  done
  (( t % 60 == 0 )) && log "  ... $n/$NODES healthy (${t}s)"
  sleep 5
done
[[ "$ready" == 1 ]] || { log "ERROR: only $n/$NODES healthy after ${READY_TIMEOUT}s"; grep -m5 -iE "error|Traceback" "$LOG_DIR"/*.log | head -20; exit 1; }
log "both nodes READY ($TOTAL engines)"

# ---------------- drive the cross-node workload ----------------
log "phase 1 -> node 0 only (populate); phase 2 -> node 1 only (must read across)"
MODEL="$MODEL" PORT_BASE="$PORT_BASE" NUM_PROMPTS="$NUM_PROMPTS" DP="$DP" \
FLOOD_PROMPTS="$FLOOD_PROMPTS" \
LOG_DIR="$LOG_DIR" python - <<'PY'
import os, json, time, urllib.request
MODEL = os.environ["MODEL"]; BASE = int(os.environ["PORT_BASE"])
N = int(os.environ["NUM_PROMPTS"]); DP = int(os.environ["DP"])
FLOOD = int(os.environ["FLOOD_PROMPTS"])
LOG_DIR = os.environ["LOG_DIR"]
# One API port per node. Requests carry "X-data-parallel-rank", which vLLM's DP
# client honours verbatim (v1/engine/core_client.py: request.data_parallel_rank
# short-circuits the load balancer), so prompt i always lands on engine i % DP.
# Without pinning, the balancer's spread decides which engine holds which prefix
# and the per-engine bookkeeping below is unreproducible.
node_ports = {0: BASE + 0, 1: BASE + 1}
filler = "The quick brown fox jumps over the lazy dog near the riverbank. " * 40
prompts = [f"Article number {i}. {filler} Please summarize article {i}." for i in range(N)]

def send(port, p, dp_rank):
    """Returns (status, completion_text). temperature=0 so that two runs of the
    same prompt on the same model can only differ if the KV differs."""
    d = json.dumps({"model": MODEL, "prompt": p, "max_tokens": 8,
                    "temperature": 0}).encode()
    r = urllib.request.Request(f"http://localhost:{port}/v1/completions", data=d,
                              headers={"Content-Type": "application/json",
                                       "X-data-parallel-rank": str(dp_rank)})
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())["choices"][0]["text"]
    except Exception as e:
        return f"ERR:{type(e).__name__}", ""

def phase(name, node):
    """Returns [(status, text), ...] -- the status stays attached so that a request
    which never came back is reported as a transport failure, not a KV difference."""
    port = node_ports[node]
    res = [send(port, p, i % DP) for i, p in enumerate(prompts)]
    ok = sum(1 for c, _ in res if c == 200)
    print(f"  {name}: node {node} (:{port}), {len(prompts)} reqs, ok={ok}", flush=True)
    if ok != len(prompts):
        print(f"    non-200: {[c for c, _ in res if c != 200]}", flush=True)
    return res

def drain(node):
    """Force every engine on `node` to finalize its last request's PUT.

    The connector only reaps a request's save callbacks while it is processing a
    LATER request, so the final request each engine served stays unpublished --
    its bytes land in the CPU tier but insert() (which is what writes the radix
    index and flushes it to the cluster router table) never runs. One short
    throwaway prompt pinned to each engine in turn pushes those callbacks
    through; pinning is what makes the coverage complete rather than likely.
    """
    port = node_ports[node]
    for k in range(DP):
        send(port, f"drain {k}: say ok.", k)
    print(f"  drain: node {node} (:{port}), {DP} reqs (one per engine)",
          flush=True)

def flood(node):
    """Push FLOOD fresh prompts through `node` to evict the phase-1 prefixes from
    its SMALL cpu tier while its larger ssd tier keeps them.

    Only then can phase 2's peer tail come over PEERSSD2H: `_shm_get_spans` ranks
    a peer's cpu tier ahead of its ssd tier, so a surviving peer cpu copy would
    serve every block and the ssd path would never be asked. Runs after phase 1's
    PUTs have landed, or the eviction order is a race between the two workloads.
    """
    port = node_ports[node]
    res = [send(port, f"Flood {i}. {filler} Please summarize flood {i}.", i % DP)
           for i in range(FLOOD)]
    ok = sum(1 for c, _ in res if c == 200)
    print(f"  flood: node {node} (:{port}), {FLOOD} reqs, ok={ok}", flush=True)

fresh = phase("phase1-populate", 0)
# The radix index publishes a block only after its bytes have landed (insert()
# runs post-transfer and flushes), so node 1 cannot see the prefix until node 0's
# async PUT completes.
drain(0)
print("  sleeping 20s for node 0's async PUT to land + publish...", flush=True)
time.sleep(20)
if FLOOD:
    flood(0)
    drain(0)
    print("  sleeping 20s for the flood's PUT + cpu eviction to settle...",
          flush=True)
    time.sleep(20)
peer = phase("phase2-crossnode", 1)
# The same prompts once more, but on node 0, which holds the prefix in its own
# tiers. Everything is shared with phase 2 except the leg that crosses nodes.
# Runs AFTER phase 2 so it cannot perturb it.
local = phase("phase3-localreuse", 0)

# Is the KV that crossed the network the KV node 0 stored? Both phases reused the
# same blocks for the same prompts at temperature=0, so they can only disagree if
# the bytes disagree. Mismatching texts are kept so a failure says WHAT diverged.
paired = [(i, p[1], l[1]) for i, (p, l) in enumerate(zip(peer, local))
          if p[0] == 200 and l[0] == 200]
same = sum(1 for _, a, b in paired if a == b)
mismatches = [{"index": i, "peer": a, "local": b} for i, a, b in paired if a != b]
errors = {name: [c for c, _ in res if c != 200] for name, res in
          (("phase1", fresh), ("phase2", peer), ("phase3", local))}
report = {"total": len(prompts), "compared": len(paired), "peer_equals_local": same,
          "errors": errors, "mismatches": mismatches}
with open(os.path.join(LOG_DIR, "output_check.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(f"  output check: phase-2 peer KV and phase-3 local KV agree on "
      f"{same}/{len(paired)} prompts", flush=True)
time.sleep(8)
PY

# ---------------- gates ----------------
ALL="$LOG_DIR/all.log"; cat "$LOG_DIR"/n[0-9]*.log > "$ALL"

fails=()
check() {  # check <description> <rc> [detail]
  if [[ "$2" == 0 ]]; then log "  ok   : $1${3:+  ($3)}"
  else log "  FAIL : $1${3:+  ($3)}"; fails+=("$1"); fi
}

echo; log "=== gates ==="

# 1. Every engine took the radixshmem path.
eng=$(grep -c "use_radix_shmem: True" "$ALL" || true)
[[ "$eng" == "$TOTAL" ]]; check "all $TOTAL engines on radixshmem" $? "n=$eng"

# 2. Per node and TIER exactly ONE region owner; the other DP engines attach as
#    clients. Two owners in a node means the dp_client_id partitioning collapsed.
#    The name is spelled out (tier + node identity), so a rename fails the gate.
TIERS=(cpu); (( SSD_CACHE_GB > 0 )) && TIERS+=(ssd)
for ((R=0; R<NODES; R++)); do
  for T in "${TIERS[@]}"; do
    o=$(grep -c "creating shm radix region /shmradix_${SHM_ID}_r${R}_${T}_${SHM_ID}_r${R} " "$ALL" || true)
    [[ "$o" == 1 ]]; check "node $R: single $T shm radix region owner" $? "n=$o"
  done
done

# 3. The two nodes joined ONE cluster with DISTINCT etcd-assigned ranks. This is
#    the cross-node control plane actually rendezvousing; a standalone fallback
#    would give both node id 0.
nids=$(grep -oE "= FlexKV node id [0-9]+" "$ALL" | grep -oE "[0-9]+$" | sort -un | tr '\n' ',' )
distinct=$(grep -oE "= FlexKV node id [0-9]+" "$ALL" | grep -oE "[0-9]+$" | sort -un | wc -l)
[[ "$distinct" == "$NODES" ]]; check "$NODES distinct FlexKV node ids" $? "ids=${nids%,}"

# 4. All DP GPUs of each node reached that node's single shared TE.
reg=$(grep -c "All $DP GPUs registered successfully" "$ALL" || true)
[[ "$reg" == "$NODES" ]]; check "both nodes registered all $DP GPUs" $? "n=$reg"

# 5. Disjoint per-engine graph-id ranges are live (ids are dp_client_id<<32), so
#    engines 1..3 emit ids above 2^32. Without it the shared TE mixes up graphs.
#    With DP=1 every engine legitimately owns the low range, so skip it there.
if (( DP > 1 )); then
  hi=$(grep -oE "graph_id=[0-9]+" "$ALL" | cut -d= -f2 | awk '$1>=4294967296' | wc -l)
  [[ "$hi" -gt 0 ]]; check "disjoint per-engine graph-id ranges" $? "ids>=2^32: $hi"
fi

# 6. THE POINT OF THE TEST: a query found a prefix on the PEER's tree.
peerq=$(grep -oE "\[RADIX PEER QUERY\].*peer_hit=[0-9]+" "$ALL" | grep -oE "peer_hit=[0-9]+$" \
        | cut -d= -f2 | awk '$1>0' | wc -l)
[[ "$peerq" -gt 0 ]]; check "peer prefix found (RADIX PEER QUERY peer_hit>0)" $? "n=$peerq"
rdma=$(grep -oE "rdma_reads=[0-9]+" "$ALL" | cut -d= -f2 | awk '$1>0' | wc -l)
[[ "$rdma" -gt 0 ]]; check "remote tree walk issued RDMA reads" $? "n=$rdma"

# 7. ...and bytes really crossed on the peer data path. The flood aims phase 2's
#    tail at the peer's ssd tier, so with the ssd tier on THAT is the gated path and
#    PEERH2H (phase 3 reading node 1's copy back) is only reported.
peer_path_gate() {  # peer_path_gate <TransferType> [info]
  local n b
  n=$(grep -c "$1 transfer request" "$ALL" || true)
  b=$(grep "$1 transfer request" "$ALL" | grep -oE "bytes=[0-9]+" \
      | cut -d= -f2 | awk '{s+=$1} END{print s+0}')
  if [[ -n "${2:-}" ]]; then log "  info : $1 transfers n=$n total=$b B"; return; fi
  [[ "$n" -gt 0 ]]; check "$1 transfers present" $? "n=$n"
  [[ "$b" -gt 0 ]]; check "$1 moved non-zero bytes" $? "total=$b B"
}
if (( SSD_CACHE_GB > 0 )); then
  peer_path_gate PEERSSD2H
  peer_path_gate PEERH2H info
else
  peer_path_gate PEERH2H
fi

# 8. ...and they were the RIGHT bytes: phase 2 (node 1, prefix pulled off the peer)
#    must decode to exactly what phase 3 (node 0, which holds that prefix itself)
#    decodes. Same model, same prompt, same reused blocks, greedy decoding -- the
#    ONLY difference is where the KV came from, so any divergence is the peer path's
#    fault. Correct plumbing cannot satisfy this by accident, which is what makes it
#    the hard gate.
CHK="$LOG_DIR/output_check.json"
GATE_KV="phase-2 peer KV decodes the same as phase-3 local KV"
if [[ "$OUTPUT_CHECK" == off ]]; then
  log "  skip : $GATE_KV  (OUTPUT_CHECK=off)"
else
  read -r agree tot errs < <(python - "$CHK" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    errs = sum(len(v) for v in d.get("errors", {}).values())
    print(int(d["peer_equals_local"]), int(d["compared"]), errs)
except Exception:
    print(-1, 0, -1)  # missing/unparseable => treated as a failure below
PY
)
  # A request that never returned leaves nothing to compare, so it gets its own
  # gate: the KV gate must not absorb (or hide behind) a transport failure.
  [[ "$errs" == 0 ]]; check "every workload request returned 200" $? "non-200=$errs"
  if [[ "$tot" -gt 0 && "$agree" == "$tot" ]]; then
    check "$GATE_KV" 0 "$agree/$tot identical"
  elif [[ "$OUTPUT_CHECK" == warn ]]; then
    log "  WARN : $GATE_KV  ($agree/$tot identical; see $CHK)"
  else
    check "$GATE_KV" 1 "$agree/$tot identical; see $CHK"
  fi
fi

# 9. Nothing crashed.
sched=$(grep -c "Error in scheduler loop" "$ALL" || true)
[[ "$sched" == 0 ]]; check "no scheduler-loop errors" $? "n=$sched"
tb=$(grep -c "^Traceback" "$ALL" || true)
[[ "$tb" == 0 ]]; check "no tracebacks" $? "n=$tb"

# 10. Phase-2 reuse rate on node 1. Node 1 issues exactly NUM_PROMPTS peer-enabled
#    GET queries PER TIER (local_only=False) in phase 2 and never stored those
#    tokens locally, so the share of them that found a peer tail IS the cross-node
#    reuse rate. Counting the queries themselves -- instead of FlexKV's per-engine
#    hit-ratio log -- keeps the measurement independent of how vLLM's DP load
#    balancer happened to spread the requests over engines.
#    The flood hands the serving tier to ssd when it is on, which is why the gated
#    tier moves with it: node 0's cpu copy is gone by design, so its rate is ~0.
GATED_TIER=cpu; (( SSD_CACHE_GB > 0 )) && GATED_TIER=ssd
echo; log "=== node 1 phase-2 cross-node reuse, $GATED_TIER tier (min ${HIT_MIN}%) ==="
N1="$LOG_DIR/n1.log"
for T in "${TIERS[@]}"; do
  # The trace names the region it queried, so the tier is read off the shm name.
  q="\[RADIX PEER QUERY\] shm=/shmradix_${SHM_ID}_r1_${T}_${SHM_ID}_r1 local_only=False"
  gets=$(grep -cE "$q" "$N1" 2>/dev/null || true)
  hitq=$(grep -E "$q" "$N1" 2>/dev/null \
         | grep -oE "peer_hit=[0-9]+" | cut -d= -f2 | awk '$1>0' | wc -l)
  # The parens are load-bearing: bare `g>0 ?` in printf's arg list parses as a
  # redirection into a file named 0.
  rate=$(awk -v h="$hitq" -v g="$gets" 'BEGIN{printf "%.2f", (g>0 ? 100*h/g : 0)}')
  if [[ "$T" != "$GATED_TIER" ]]; then
    log "  info : node1 $T-tier peer GETs $hitq/$gets = ${rate}%"
    continue
  fi
  [[ "$gets" == "$NUM_PROMPTS" ]]
  check "node1 issued one $T peer GET query per prompt" $? "queries=$gets want=$NUM_PROMPTS"
  if (( gets > 0 )); then
    awk -v r="$rate" -v m="$HIT_MIN" 'BEGIN{exit !(r+0 >= m+0)}'
    check "node1 phase-2 GETs served from the peer's $T tier" $? \
          "$hitq/$gets = ${rate}% >= ${HIT_MIN}%"
  else
    check "node1 phase-2 GETs served from the peer's $T tier" 1 \
          "no peer GET queries traced"
  fi
done
# Informational: FlexKV's own per-engine window (proc tag is "EngineCore_DP<i>"
# under DP>1, plain "EngineCore" when DP=1).
for ln in $(grep -oE "\(EngineCore[^ ]* pid=[0-9]+\).*FlexKV Hit Ratio: [0-9.]+%" "$N1" 2>/dev/null \
            | sed -E 's/\(([^ ]+) pid=[0-9]+\).*Hit Ratio: ([0-9.]+)%/\1=\2%/'); do
  log "  info : $ln"
done

echo
if [[ ${#fails[@]} == 0 ]]; then
  log "RESULT: PASS (all gates) -- cross-node reuse verified"; exit 0
else
  log "RESULT: FAIL (${#fails[@]} gate(s)): ${fails[*]}"
  log "inspect $LOG_DIR/n<node>.log"; exit 1
fi
# cleanup() runs automatically here via the EXIT trap
