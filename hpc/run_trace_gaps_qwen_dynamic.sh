#!/bin/bash
# Fill the final 6 Qwen Phase A cells — both dynamic policies × 3 datasets.
#
# Remaining after the 2 earlier Qwen runs:
#   * v3_dynamic × {mtbench, math500, livecodebench}
#   * v6_dynamic × {mtbench, math500, livecodebench}
#                                                      --
#                                                      6 cells
#
# ~2 server launches × (60s cold + 3 benches × ~6 min + kill) ≈ 40 min.
# Fits in a 1h INTR with margin if the node isn't sick.
#
# Usage (inside a live sintr, env activated):
#   conda activate sglang-dev
#   source hpc/unload_prepare.sh
#   bash hpc/run_trace_gaps_qwen_dynamic.sh

set -uo pipefail

REPO_ROOT="/rds/user/ou222/hpc-work/diss"
cd "$REPO_ROOT"
mkdir -p logs results/traces

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PORT=30000
MODEL="Qwen/Qwen3-8B"
DRAFT="AngelSlim/Qwen3-8B_eagle3"
TAG="qwen"

wait_for_server() {
    python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=900)
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

run_one() {
    local NAME="$1"; local NS="$2"; local TK="$3"; local NDT="$4"
    local NS_MAX="$5"; local TK_MAX="$6"; local NDT_MAX="$7"; local POLICY="$8"
    shift 8
    local DSETS=("$@")

    local SERVER_LOG="logs/phase_a_gaps2_${TAG}_${NAME}_${TIMESTAMP}.log"
    echo ""
    echo "================================================================"
    echo ">>> [GAP2 $TAG / $NAME] dynamic policy=$POLICY"
    echo "    start=($NS,$TK,$NDT)  max=($NS_MAX,$TK_MAX,$NDT_MAX)"
    echo "    Datasets: ${DSETS[*]}"
    echo "================================================================"

    pkill -9 -u ou222 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 2

    ( cd /tmp && python3 -m sglang.launch_server \
        --model "$MODEL" \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$DRAFT" \
        --speculative-num-steps "$NS" --speculative-eagle-topk "$TK" --speculative-num-draft-tokens "$NDT" \
        --speculative-num-steps-max "$NS_MAX" --speculative-eagle-topk-max "$TK_MAX" --speculative-num-draft-tokens-max "$NDT_MAX" \
        --enable-dynamic-speculative-decoding --dynamic-spec-policy "$POLICY" \
        --dynamic-spec-full-logging \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
        > "$REPO_ROOT/$SERVER_LOG" 2>&1 ) &
    local SERVER_PID=$!
    echo "    Server PID: $SERVER_PID   log: $SERVER_LOG"

    if ! wait_for_server; then
        echo "    [SKIP] server never ready — see $SERVER_LOG"
        kill_server "$SERVER_PID"
        return 0
    fi

    for bench in "${DSETS[@]}"; do
        local dset="${bench%:*}"
        local n="${bench#*:}"
        local out="results/traces/${NAME}_${TAG}_${dset}_${TIMESTAMP}.json"
        echo ""
        echo "    --- bench: $dset  (n=$n)  → $out"
        python3 test_signal_collection.py \
            --host localhost --port "$PORT" \
            --dataset "$dset" --num-samples "$n" \
            --output "$out" \
            || echo "    [WARN] $dset failed for $NAME — continuing"
    done

    kill_server "$SERVER_PID"
}

echo ""
echo "################################################################"
echo "#  QWEN PHASE A — FINAL DYNAMIC GAP FILLER (6 cells)"
echo "#  Model: $MODEL"
echo "#  Draft: $DRAFT"
echo "#  Timestamp: $TIMESTAMP"
echo "################################################################"

# v3: smooth linear interp, DOG-driven, start=(3,1,4) → max=(7,4,8)
run_one "v3_dynamic"  3 1 4    7 4 8 v3     mtbench:20 math500:20 livecodebench:20

# v6: 3-zone, t=0.7*top1+0.3*target_top1, start=(3,1,4) → max=(10,5,16)
run_one "v6_dynamic"  3 1 4    10 5 16 v6   mtbench:20 math500:20 livecodebench:20

echo ""
echo "================================================================"
echo "  QWEN DYNAMIC GAP FILLER COMPLETE  ($(date +%H:%M:%S))"
echo "  Check coverage:"
echo "    ls results/traces/*_${TAG}_*.json | wc -l    # expect 18"
echo "================================================================"
