#!/bin/bash
# Phase A — Cross-model dynamic speculative decoding trace collection.
#
# Runs 36 benchmarks total:
#   6 configs × 2 models × 3 datasets × 20 q/dataset
#   configs: static_3_1_4, static_7_1_8, static_7_4_8, static_6_10_60,
#            v3_dynamic, v6_dynamic
#   models : meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B
#   datasets: mtbench:20, math500:20, livecodebench:20
#
# Usage:
#   sbatch hpc/run_trace_collection.sh                     # both models
#   ONLY_MODEL=llama sbatch hpc/run_trace_collection.sh    # Llama only
#   ONLY_MODEL=qwen  sbatch hpc/run_trace_collection.sh    # Qwen only
#
# Or interactive (inside a live sintr):
#   conda activate sglang-dev
#   source hpc/unload_prepare.sh
#   ONLY_MODEL=llama bash hpc/run_trace_collection.sh --no-sbatch
#
# Expected runtime:
#   Per model: ~6 configs × (60s cold import + 3×2min bench + 5s kill) ≈ 45 min
#   Both models: ~1.5 h
#
# Outputs: results/traces/<config>_<model>_<dataset>.json
#          (test_signal_collection.py format — has per_turn_logs with signals +
#          token-level data. Post-process with analysis/extract_signal_traces.py)

#SBATCH -J phase_a_traces
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/phase_a_%j.out
#SBATCH --error=logs/phase_a_%j.err
#SBATCH --exclude=gpu-q-1,gpu-q-3

set -uo pipefail

REPO_ROOT="/rds/user/ou222/hpc-work/diss"
cd "$REPO_ROOT"
mkdir -p logs results results/traces

# Activate env only if NOT sourced from an already-prepared sintr shell
if [[ "${1:-}" != "--no-sbatch" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate sglang-dev
    source hpc/unload_prepare.sh
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PORT=30000
DATASETS=("mtbench:20" "math500:20" "livecodebench:20")

# ---------------------------------------------------------------------- #
# Helper: wait for server ready                                          #
# ---------------------------------------------------------------------- #
wait_for_server() {
    python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=900)
print('  Server ready.')
"
}

# ---------------------------------------------------------------------- #
# Helper: kill a server by PID with timeout                              #
# ---------------------------------------------------------------------- #
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
# Helper: run one config (static or dynamic) across 3 datasets           #
# Args: TAG MODEL DRAFT CONFIG_NAME NS TK NDT [NS_MAX TK_MAX NDT_MAX POLICY]
# If the 8th arg is present, treat as dynamic spec. Otherwise static.
# ---------------------------------------------------------------------- #
run_one_config() {
    local TAG="$1"; local MODEL="$2"; local DRAFT="$3"; local NAME="$4"
    local NS="$5"; local TK="$6"; local NDT="$7"
    local NS_MAX="${8:-}"; local TK_MAX="${9:-}"; local NDT_MAX="${10:-}"; local POLICY="${11:-}"

    local IS_DYNAMIC=0
    if [[ -n "$POLICY" ]]; then IS_DYNAMIC=1; fi

    local SERVER_LOG="logs/phase_a_${TAG}_${NAME}_${TIMESTAMP}.log"
    echo ""
    echo "================================================================"
    if [[ $IS_DYNAMIC -eq 1 ]]; then
        echo ">>> [$TAG / $NAME] dynamic policy=$POLICY  "
        echo "    start=($NS,$TK,$NDT)  max=($NS_MAX,$TK_MAX,$NDT_MAX)"
    else
        echo ">>> [$TAG / $NAME] static  ($NS,$TK,$NDT)"
    fi
    echo "================================================================"

    # Kill any stale server on our port
    pkill -9 -u ou222 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 2

    # Launch (from /tmp to avoid diss/sglang CWD shadow)
    local LAUNCH_CMD=(
        python3 -m sglang.launch_server
        --model "$MODEL"
        --speculative-algorithm EAGLE3
        --speculative-draft-model-path "$DRAFT"
        --speculative-num-steps "$NS" --speculative-eagle-topk "$TK" --speculative-num-draft-tokens "$NDT"
        --dynamic-spec-full-logging
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16
    )
    if [[ $IS_DYNAMIC -eq 1 ]]; then
        LAUNCH_CMD+=(
            --speculative-num-steps-max "$NS_MAX"
            --speculative-eagle-topk-max "$TK_MAX"
            --speculative-num-draft-tokens-max "$NDT_MAX"
            --enable-dynamic-speculative-decoding
            --dynamic-spec-policy "$POLICY"
        )
    fi

    ( cd /tmp && "${LAUNCH_CMD[@]}" > "$REPO_ROOT/$SERVER_LOG" 2>&1 ) &
    local SERVER_PID=$!
    echo "    Server PID: $SERVER_PID   log: $SERVER_LOG"

    # Wait for readiness
    if ! wait_for_server; then
        echo "    [SKIP] server never ready — see $SERVER_LOG"
        kill_server "$SERVER_PID"
        return 0
    fi

    # Run 3 benches against the same server
    for bench in "${DATASETS[@]}"; do
        local dset="${bench%:*}"
        local n="${bench#*:}"
        local out="results/traces/${NAME}_${TAG}_${dset}_${TIMESTAMP}.json"
        echo ""
        echo "    --- bench: $dset  (n=$n)  → $out"
        python3 test_signal_collection.py \
            --host localhost --port "$PORT" \
            --dataset "$dset" --num-samples "$n" \
            --output "$out" \
            || echo "    [WARN] $dset failed for $NAME/$TAG — continuing"
    done

    kill_server "$SERVER_PID"
}

# ---------------------------------------------------------------------- #
# Model-level driver                                                     #
# ---------------------------------------------------------------------- #
run_for_model() {
    local TAG="$1"; local MODEL="$2"; local DRAFT="$3"
    echo ""
    echo "################################################################"
    echo "#  MODEL: $TAG  ($MODEL)"
    echo "#  DRAFT: $DRAFT"
    echo "################################################################"

    # 4 static configs — (num_steps, topk, ndt)
    run_one_config "$TAG" "$MODEL" "$DRAFT" "static_3_1_4"    3 1 4
    run_one_config "$TAG" "$MODEL" "$DRAFT" "static_7_1_8"    7 1 8
    run_one_config "$TAG" "$MODEL" "$DRAFT" "static_7_4_8"    7 4 8
    run_one_config "$TAG" "$MODEL" "$DRAFT" "static_6_10_60"  6 10 60

    # 2 dynamic configs — (start ns, tk, ndt, max ns, tk, ndt, policy)
    run_one_config "$TAG" "$MODEL" "$DRAFT" "v3_dynamic"  3 1 4  7 4 8  v3
    run_one_config "$TAG" "$MODEL" "$DRAFT" "v6_dynamic"  3 1 4  10 5 16  v6
}

# ---------------------------------------------------------------------- #
# Main — select models per ONLY_MODEL env var                            #
# ---------------------------------------------------------------------- #
ONLY_MODEL="${ONLY_MODEL:-both}"
case "$ONLY_MODEL" in
    llama|Llama)
        run_for_model "llama" "meta-llama/Llama-3.1-8B-Instruct" "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
        ;;
    qwen|Qwen)
        run_for_model "qwen"  "Qwen/Qwen3-8B" "AngelSlim/Qwen3-8B_eagle3"
        ;;
    both|*)
        run_for_model "llama" "meta-llama/Llama-3.1-8B-Instruct" "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
        run_for_model "qwen"  "Qwen/Qwen3-8B" "AngelSlim/Qwen3-8B_eagle3"
        ;;
esac

echo ""
echo "================================================================"
echo "  PHASE A COMPLETE  ($(date +%H:%M:%S))"
echo "  Outputs: results/traces/*_${TIMESTAMP}.json"
echo "  Next: python3 analysis/extract_signal_traces.py results/traces/"
echo "================================================================"
