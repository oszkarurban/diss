#!/bin/bash
# Probe signal sweep — interactive GPU node (no SLURM queue).
# Usage:
#   source hpc/unload_prepare.sh   # (if not already done)
#   conda activate sglang-dev      # (if not already done)
#   bash run_probe_sweep_interactive.sh

set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
PORT=30000
NUM_SAMPLES=5          # increase for more stable results (max 80)
SERVER_TIMEOUT=600     # seconds to wait for server startup (5-10 min model load)
OUTPUT_DIR="probe_logs"

# Bidirectional policy: server launches with max, generation starts at startpoint
MAX_STEPS=5
MAX_TOPK=1
MAX_DTN=6
START_STEPS=3
START_DTN=4

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR" logs

BASE_SERVER_ARGS=(
    python3 -m sglang.launch_server
    --model "$MODEL"
    --speculative-algorithm EAGLE3
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-num-steps "$MAX_STEPS"
    --speculative-eagle-topk "$MAX_TOPK"
    --speculative-num-draft-tokens "$MAX_DTN"
    --speculative-num-steps-startpoint "$START_STEPS"
    --speculative-num-draft-tokens-startpoint "$START_DTN"
    --mem-fraction-static 0.75
    --cuda-graph-max-bs 1
    --tp 1
    --trust-remote-code
    --host 0.0.0.0
    --port "$PORT"
    --dtype bfloat16
)

SWEEP_CONFIGS=(vanilla all draft_entropy top1_prob top1_minus_top2 hidden_norm path_score target_entropy entropy_gap rolling_accept_rate)

# Save configs JSON (for reference / --compare)
python probe_signals.py --save-configs --output-dir "$OUTPUT_DIR"

for CONFIG_NAME in "${SWEEP_CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo " CONFIG: $CONFIG_NAME"
    echo "============================================================"

    # Get the --dynamic-spec-config JSON (prints "none" for vanilla)
    DYN_CFG=$(python probe_signals.py --get-config "$CONFIG_NAME" --output-dir "$OUTPUT_DIR")

    # Launch server in background
    if [ "$DYN_CFG" = "none" ]; then
        echo "Launching server (vanilla — no dynamic spec)..."
        "${BASE_SERVER_ARGS[@]}" &
    else
        echo "Launching server with --dynamic-spec-config '$DYN_CFG'..."
        "${BASE_SERVER_ARGS[@]}" \
            --enable-dynamic-spec \
            --dynamic-spec-config "$DYN_CFG" &
    fi
    SERVER_PID=$!

    # Wait for server ready (up to SERVER_TIMEOUT seconds — model load takes 5-10 min)
    echo "Waiting up to ${SERVER_TIMEOUT}s for server (PID $SERVER_PID)..."
    python -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=$SERVER_TIMEOUT)
print('Server ready.')
"

    # Run benchmark for this config
    python probe_signals.py \
        --host localhost \
        --port "$PORT" \
        --num-samples "$NUM_SAMPLES" \
        --output-dir "$OUTPUT_DIR" \
        --signal "$CONFIG_NAME"

    # Kill server and wait for cleanup (|| true prevents set -e from firing on SIGTERM exit code)
    echo "Killing server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    sleep 5   # let GPU memory fully free before next launch
done

echo ""
echo "============================================================"
echo " ALL CONFIGS DONE — COMPARISON"
echo "============================================================"
python probe_signals.py --compare --output-dir "$OUTPUT_DIR"
