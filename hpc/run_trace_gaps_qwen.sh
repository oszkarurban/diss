#!/bin/bash
# Fill the 13 missing Qwen Phase A cells from the 2026-04-18 13:32 run.
#
# Completed on disk already:
#   * static_3_1_4 × {mtbench, math500, livecodebench}   (3 cells done)
#   * static_7_1_8 × {mtbench, math500}                  (2 cells done)
#
# Missing (this script runs):
#   * static_7_1_8   × {livecodebench}                   (1 cell)
#   * static_7_4_8   × {mtbench, math500, livecodebench} (3 cells)
#   * static_6_10_60 × {mtbench, math500, livecodebench} (3 cells)
#   * v3_dynamic     × {mtbench, math500, livecodebench} (3 cells)
#   * v6_dynamic     × {mtbench, math500, livecodebench} (3 cells)
#                                                         ---------
#                                                         13 cells
#
# Qwen is slower per-cell than Llama (~5-7 min per 20q bench because Qwen
# generates ~5× longer outputs). Expected wall time:
#   5 server launches × (60s cold + ~15-20 min benches + 5s kill) ≈ 90 min.
#
# Recommended allocation: 2h (drop --qos=INTR) OR split in two 1h chunks.
# The script is resumable — if a second run writes new timestamps,
# analysis/extract_signal_traces.py picks the latest per (config, model,
# dataset) cell so duplicates are harmless.
#
# Usage (inside a live sintr, env activated):
#   conda activate sglang-dev
#   source hpc/unload_prepare.sh
#   bash hpc/run_trace_gaps_qwen.sh

set -uo pipefail

REPO_ROOT="/rds/user/ou222/hpc-work/diss"
cd "$REPO_ROOT"
mkdir -p logs results/traces

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PORT=30000
MODEL="Qwen/Qwen3-8B"
DRAFT="AngelSlim/Qwen3-8B_eagle3"
TAG="qwen"

# ---------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------- #
# run_one NAME NS TK NDT [NS_MAX TK_MAX NDT_MAX POLICY]   DATASETS...
# If NS_MAX+POLICY present → dynamic spec. Otherwise static.
# ---------------------------------------------------------------------- #
run_one() {
    local NAME="$1"; local NS="$2"; local TK="$3"; local NDT="$4"
    shift 4
    local NS_MAX="" TK_MAX="" NDT_MAX="" POLICY=""
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        NS_MAX="$1"; TK_MAX="$2"; NDT_MAX="$3"; POLICY="$4"
        shift 4
    fi
    local DSETS=("$@")

    local IS_DYN=0
    [[ -n "$POLICY" ]] && IS_DYN=1

    local SERVER_LOG="logs/phase_a_gaps_${TAG}_${NAME}_${TIMESTAMP}.log"
    echo ""
    echo "================================================================"
    if [[ $IS_DYN -eq 1 ]]; then
        echo ">>> [GAP $TAG / $NAME] dynamic policy=$POLICY"
        echo "    start=($NS,$TK,$NDT)  max=($NS_MAX,$TK_MAX,$NDT_MAX)"
    else
        echo ">>> [GAP $TAG / $NAME] static  ($NS,$TK,$NDT)"
    fi
    echo "    Datasets: ${DSETS[*]}"
    echo "================================================================"

    pkill -9 -u ou222 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 2

    local CMD=(
        python3 -m sglang.launch_server
        --model "$MODEL"
        --speculative-algorithm EAGLE3
        --speculative-draft-model-path "$DRAFT"
        --speculative-num-steps "$NS" --speculative-eagle-topk "$TK" --speculative-num-draft-tokens "$NDT"
        --dynamic-spec-full-logging
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16
    )
    if [[ $IS_DYN -eq 1 ]]; then
        CMD+=(
            --speculative-num-steps-max "$NS_MAX"
            --speculative-eagle-topk-max "$TK_MAX"
            --speculative-num-draft-tokens-max "$NDT_MAX"
            --enable-dynamic-speculative-decoding
            --dynamic-spec-policy "$POLICY"
        )
    fi

    ( cd /tmp && "${CMD[@]}" > "$REPO_ROOT/$SERVER_LOG" 2>&1 ) &
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
echo "#  QWEN PHASE A — GAP FILLER (13 cells)"
echo "#  Model: $MODEL"
echo "#  Draft: $DRAFT"
echo "#  Timestamp: $TIMESTAMP"
echo "################################################################"

# Gap 1: static_7_1_8 → only livecodebench (mtbench + math500 already done)
run_one "static_7_1_8"   7 1 8    livecodebench:20

# Gap 2: static_7_4_8 → all 3
run_one "static_7_4_8"   7 4 8    mtbench:20 math500:20 livecodebench:20

# Gap 3: static_6_10_60 → all 3
run_one "static_6_10_60" 6 10 60  mtbench:20 math500:20 livecodebench:20

# Gap 4: v3_dynamic → all 3
run_one "v3_dynamic"     3 1 4    7 4 8 v3     mtbench:20 math500:20 livecodebench:20

# Gap 5: v6_dynamic → all 3
run_one "v6_dynamic"     3 1 4    10 5 16 v6   mtbench:20 math500:20 livecodebench:20

echo ""
echo "================================================================"
echo "  QWEN GAP FILLER COMPLETE  ($(date +%H:%M:%S))"
echo "  Check coverage:"
echo "    ls results/traces/*_${TAG}_*.json | wc -l   # expect 18"
echo "================================================================"
