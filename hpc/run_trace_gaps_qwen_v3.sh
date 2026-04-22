#!/bin/bash
# Fill the final 3 Qwen Phase A cells — v3_dynamic × 3 datasets.
#
# v3 stalled on a bad node during the previous gap run (log stayed 0 bytes).
# This script extends wait_for_server to 30 min so a slow-cold Lustre node
# still has time to complete imports + model load before giving up.
#
# Usage (inside a live sintr, env activated):
#   conda activate sglang-dev
#   source hpc/unload_prepare.sh
#   bash hpc/run_trace_gaps_qwen_v3.sh

set -uo pipefail

REPO_ROOT="/rds/user/ou222/hpc-work/diss"
cd "$REPO_ROOT"
mkdir -p logs results/traces

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PORT=30000
MODEL="Qwen/Qwen3-8B"
DRAFT="AngelSlim/Qwen3-8B_eagle3"

# Extended to 30 min (was 15) — a cold Lustre node can take >15 min to load
# a model. Gives the server ample time before the script gives up.
SERVER_READY_TIMEOUT=1800

wait_for_server() {
    python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=$SERVER_READY_TIMEOUT)
print('  Server ready.')
"
}
kill_server() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    sleep 3
}

SERVER_LOG="logs/phase_a_gaps3_qwen_v3_dynamic_${TIMESTAMP}.log"
echo ""
echo "################################################################"
echo "#  QWEN v3_dynamic REPLAY  (3 cells, 30-min server timeout)"
echo "#  Model: $MODEL"
echo "#  Draft: $DRAFT"
echo "#  start=(3,1,4)  max=(7,4,8)  policy=v3"
echo "#  Timestamp: $TIMESTAMP"
echo "################################################################"

# Kill anything stale
pkill -9 -u ou222 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 2

( cd /tmp && python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --speculative-num-steps-max 7 --speculative-eagle-topk-max 4 --speculative-num-draft-tokens-max 8 \
    --enable-dynamic-speculative-decoding --dynamic-spec-policy v3 \
    --dynamic-spec-full-logging \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
    > "$REPO_ROOT/$SERVER_LOG" 2>&1 ) &
SERVER_PID=$!
echo "    Server PID: $SERVER_PID   log: $SERVER_LOG"
echo "    Waiting up to $((SERVER_READY_TIMEOUT / 60)) min for server ready..."

if ! wait_for_server; then
    echo ""
    echo "    [FAIL] server never ready in $((SERVER_READY_TIMEOUT / 60)) min."
    echo "    Check $SERVER_LOG size: $(stat -c %s "$SERVER_LOG" 2>/dev/null || echo 0) bytes"
    echo "    Likely cause: Lustre I/O stall on this node. scancel + retry on another."
    kill_server "$SERVER_PID"
    exit 1
fi

for bench in mtbench:20 math500:20 livecodebench:20; do
    dset="${bench%:*}"
    n="${bench#*:}"
    out="results/traces/v3_dynamic_qwen_${dset}_${TIMESTAMP}.json"
    echo ""
    echo "    --- bench: $dset  (n=$n)  → $out"
    python3 test_signal_collection.py \
        --host localhost --port "$PORT" \
        --dataset "$dset" --num-samples "$n" \
        --output "$out" \
        || echo "    [WARN] $dset failed — continuing"
done

kill_server "$SERVER_PID"

echo ""
echo "================================================================"
echo "  v3_dynamic QWEN REPLAY COMPLETE  ($(date +%H:%M:%S))"
echo "  Files: results/traces/v3_dynamic_qwen_*_${TIMESTAMP}.json"
echo "  Total Qwen cells after this: $(ls results/traces/*_qwen_*.json 2>/dev/null | awk -F_ '{print $(NF-3)"_"$(NF-2)"_"$(NF-1)}' | sort -u | wc -l)/18"
echo "================================================================"
