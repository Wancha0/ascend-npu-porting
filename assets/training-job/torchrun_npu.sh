#!/usr/bin/env bash
# Generic foreground TorchRun launcher for an already initialized Ascend environment.
set -euo pipefail

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_var() {
  local name=$1
  [[ -n ${!name:-} ]] || die "required environment variable is empty: $name"
}

require_uint() {
  local name=$1 value=${!1:-}
  [[ $value =~ ^[0-9]+$ ]] || die "$name must be an unsigned integer"
}

require_var PROJECT_ROOT
require_var RUN_ROOT
require_var RUN_ID
require_var NNODES
require_var NPROC_PER_NODE
require_var NODE_RANK
require_var MASTER_ADDR
require_var MASTER_PORT
require_uint NNODES
require_uint NPROC_PER_NODE
require_uint NODE_RANK
require_uint MASTER_PORT

((NNODES >= 1)) || die "NNODES must be at least 1"
((NPROC_PER_NODE >= 1)) || die "NPROC_PER_NODE must be at least 1"
((NODE_RANK < NNODES)) || die "NODE_RANK must be smaller than NNODES"
((MASTER_PORT >= 1024 && MASTER_PORT <= 65535)) || die "MASTER_PORT must be 1024..65535"
[[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "RUN_ID contains unsafe characters"
[[ $PROJECT_ROOT == /* ]] || die "PROJECT_ROOT must be absolute"
[[ $RUN_ROOT == /* ]] || die "RUN_ROOT must be absolute"
[[ -d $PROJECT_ROOT ]] || die "PROJECT_ROOT is not a directory: $PROJECT_ROOT"
[[ $# -ge 1 ]] || die "usage: torchrun_npu.sh TRAIN_ENTRYPOINT [TRAIN_ARGS...]"
command -v torchrun >/dev/null 2>&1 || die "torchrun is not available on PATH"

TRAIN_ENTRYPOINT=$1
shift
if [[ $TRAIN_ENTRYPOINT != /* ]]; then
  TRAIN_ENTRYPOINT=$PROJECT_ROOT/$TRAIN_ENTRYPOINT
fi
[[ -f $TRAIN_ENTRYPOINT ]] || die "training entrypoint is not a file: $TRAIN_ENTRYPOINT"

if ((NNODES > 1)); then
  case $MASTER_ADDR in
    127.*|localhost|::1) die "multi-node MASTER_ADDR cannot be loopback" ;;
  esac
fi

if [[ -n ${VISIBLE_DEVICES:-} ]]; then
  IFS=',' read -r -a visible_device_list <<< "$VISIBLE_DEVICES"
  ((${#visible_device_list[@]} >= NPROC_PER_NODE)) || \
    die "VISIBLE_DEVICES exposes fewer devices than NPROC_PER_NODE"
  export ASCEND_VISIBLE_DEVICES=$VISIBLE_DEVICES
fi

NODE_RUN_DIR=$RUN_ROOT/$RUN_ID/node-$NODE_RANK
mkdir -p "$RUN_ROOT/$RUN_ID"
mkdir "$NODE_RUN_DIR" || die "node run directory already exists: $NODE_RUN_DIR"
LOG_FILE=$NODE_RUN_DIR/torchrun.log

cd "$PROJECT_ROOT"
{
  printf 'run_id=%s node_rank=%s nnodes=%s nproc_per_node=%s host=%s\n' \
    "$RUN_ID" "$NODE_RANK" "$NNODES" "$NPROC_PER_NODE" "$(hostname)"
  printf 'master=%s:%s project_root=%s entrypoint=%s\n' \
    "$MASTER_ADDR" "$MASTER_PORT" "$PROJECT_ROOT" "$TRAIN_ENTRYPOINT"
  printf 'ASCEND_TORCHRUN_NODE_START\n'
} | tee "$LOG_FILE"

set +e
torchrun \
  --nnodes "$NNODES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "$TRAIN_ENTRYPOINT" "$@" 2>&1 | tee -a "$LOG_FILE"
run_status=${PIPESTATUS[0]}
set -e

if ((run_status == 0)); then
  printf 'ASCEND_TORCHRUN_NODE_PASS\n' | tee -a "$LOG_FILE"
else
  printf 'ASCEND_TORCHRUN_NODE_FAIL exit_code=%s\n' "$run_status" | tee -a "$LOG_FILE" >&2
fi
exit "$run_status"
