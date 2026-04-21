#!/bin/bash
# Vanilla EAGLE3 chain benchmarks (no dynamic speculative decoding).
# Runs two static configs — (3,1,4) small chain and (7,1,8) deep chain —
# against all five benchmarks.  Produces the baseline numbers that
# the dynamic policy's throughput must meet or beat.
#
# Usage:  bash hpc/bench_chains.sh {llama|deepseek|qwen}
set -euo pipefail

MODEL_TAG="${1:-llama}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/bench_chains

export HF_HOME="/workspace/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCHMARKS="mtbench:80 gsm8k:100 math500:100 humaneval:100 livecodebench:100"

# Per-model GPU pinning + distinct port — matches calibrate_top1_dynamic.sh
# so three tags can run concurrently on a 3× A100 box.
case "$MODEL_TAG" in
    llama)
        MODEL="meta-llama/Llama-3.1-8B-Instruct"
        DRAFT="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
        GPU_ID=0
        PORT=30000
        ;;
    deepseek)
        MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        DRAFT="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
        GPU_ID=1
        PORT=30001
        ;;
    qwen)
        MODEL="Qwen/Qwen3-8B"
        DRAFT="AngelSlim/Qwen3-8B_eagle3"
        GPU_ID=2
        PORT=30002
        ;;
    *)
        echo "Unknown model tag: $MODEL_TAG (expected llama|deepseek|qwen)" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
TAG="${MODEL_TAG}"

wait_for_server() {
    echo "  Waiting for server (PID $1)..."
    python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=1800)
print('  Server ready.')
"
}

kill_server() {
    echo "  Killing server (PID $1)..."
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
    sleep 5
}

run_bench() {
    local label="$1"
    echo "  Running benchmarks: $label"
    for bench in $BENCHMARKS; do
        local bench_name="${bench%%:*}"
        echo "    >>> $bench"
        python SpecForge/benchmarks/bench_eagle3.py \
            --model-path "$MODEL" \
            --port "$PORT" \
            --config-list 1,0,0,0 \
            --benchmark-list "$bench" \
            --dtype bfloat16 \
            --skip-launch-server \
            --name "${TAG}_${label}_${bench_name}_${TIMESTAMP}" \
            --output-dir "results/bench_chains" \
            || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo "  Chain Benchmarks — $MODEL_TAG"
echo "  Model: $MODEL"
echo "  Draft: $DRAFT"
echo "  GPU:   $GPU_ID    Port: $PORT"
echo "  Timestamp: $TIMESTAMP"
echo "============================================================"

# 1. Static 3,1,4 (small chain)
echo ""
echo ">>> [1/2] Vanilla EAGLE3 3,1,4 (small chain)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_314"
kill_server $SERVER_PID

# 2. Static 7,1,8 (deep chain)
echo ""
echo ">>> [2/2] Vanilla EAGLE3 7,1,8 (deep chain)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_718"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo "  CHAIN BENCHMARKS COMPLETE — $TAG"
echo "  Results under results/bench_chains/"
echo "============================================================"
