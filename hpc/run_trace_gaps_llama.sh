#!/bin/bash
# Fill the 6 missing Llama Phase A cells from the 2026-04-18 12:34 run.
#
# Missing cells (what this script runs):
#   * static_3_1_4 × {math500, livecodebench}   (crashed pre-answer_keys fix)
#   * v3_dynamic   × {livecodebench}            (sintr expired mid-bench)
#   * v6_dynamic   × {mtbench, math500, livecodebench}  (never started)
#
# Runs 3 server launches (one per config, 3 datasets only on v6_dynamic),
# ~20 min total on a warm node. Re-uses test_signal_collection.py with the
# fix to extract_signal_logs(answer_keys) already in place.
#
# Usage (inside a live 1h sintr, env activated):
#   conda activate sglang-dev
#   source hpc/unload_prepare.sh
#   bash hpc/run_trace_gaps_llama.sh

set -uo pipefail

REPO_ROOT="/rds/user/ou222/hpc-work/diss"
cd "$REPO_ROOT"
mkdir -p logs results/traces

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PORT=30000
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
TAG="llama"

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
# Launch one server, run a list of datasets, kill.
# Args: NAME NS TK NDT [NS_MAX TK_MAX NDT_MAX POLICY]   DATASETS...
# Last arguments after the config block are dataset:N strings.
# ---------------------------------------------------------------------- #
run_one() {
    local NAME="$1"; local NS="$2"; local TK="$3"; local NDT="$4"
    shift 4
    # If next 4 args are integers + v3/v6, treat as dynamic. Static otherwise.
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
echo "#  LLAMA PHASE A — GAP FILLER"
echo "#  Model: $MODEL"
echo "#  Draft: $DRAFT"
echo "#  Timestamp: $TIMESTAMP"
echo "################################################################"

# Gap 1: static_3_1_4 → only math500 + livecodebench (mtbench is already on disk)
run_one "static_3_1_4"  3 1 4    math500:20 livecodebench:20

# Gap 2: v3_dynamic → only livecodebench (mtbench + math500 already on disk)
run_one "v3_dynamic"    3 1 4    7 4 8 v3     livecodebench:20

# Gap 3: v6_dynamic → all 3 datasets (nothing on disk)
run_one "v6_dynamic"    3 1 4    10 5 16 v6   mtbench:20 math500:20 livecodebench:20

echo ""
echo "================================================================"
echo "  GAP FILLER COMPLETE  ($(date +%H:%M:%S))"
echo "  Check coverage:"
echo "    ls results/traces/*_${TAG}_*.json | wc -l   # expect 18"
echo "================================================================"
