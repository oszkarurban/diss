#!/bin/bash
# Vanilla EAGLE3 baseline — static (num_steps=7, topk=1, ndt=8) deep chain.
# Runs all five datasets so throughput is directly comparable to
# hpc/bench_dynamic.sh and hpc/bench_static_3_1_4.sh.
#
# Usage:  bash hpc/bench_static_7_1_8.sh {llama|deepseek|qwen}
set -euo pipefail

MODEL_TAG="${1:-llama}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/bench_chains

export HF_HOME="/workspace/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCHMARKS="${BENCHMARKS:-mtbench:80 gsm8k:100 math500:100 humaneval:100 livecodebench:100}"

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

GPU_ID="${GPU_OVERRIDE:-$GPU_ID}"
PORT="${PORT_OVERRIDE:-$PORT}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
TAG="${MODEL_TAG}"
LABEL="vanilla_718"

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
            --name "${TAG}_${LABEL}_${bench_name}_${TIMESTAMP}" \
            --output-dir "results/bench_chains" \
            || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo "  Static EAGLE3 (7,1,8) — $MODEL_TAG"
echo "  Model: $MODEL"
echo "  Draft: $DRAFT"
echo "  GPU:   $GPU_ID    Port: $PORT"
echo "  Timestamp: $TIMESTAMP"
echo "============================================================"

echo ""
echo ">>> Launching server"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo "  STATIC (7,1,8) COMPLETE — $TAG"
echo "  Results under results/bench_chains/"
echo "============================================================"
