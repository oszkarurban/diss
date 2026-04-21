#!/bin/bash
# Dynamic speculative decoding throughput benchmark.
# Runs the single top1-driven dynamic policy over all 5 datasets at the
# same sizes as hpc/bench_chains.sh so results are directly comparable
# to the vanilla EAGLE3 (3,1,4) and (7,1,8) baselines.
#
# Usage:  bash hpc/bench_dynamic.sh {llama|deepseek|qwen}
set -euo pipefail

MODEL_TAG="${1:-llama}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/bench_dynamic

export HF_HOME="/workspace/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCHMARKS="mtbench:80 gsm8k:100 math500:100 humaneval:100 livecodebench:100"

# Per-model GPU pinning + distinct port — matches bench_chains.sh exactly
# so a dynamic run can be executed in parallel with any static run for the
# other two models, and so the apples-to-apples comparison uses the same GPU.
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

# Match the envelope used during calibration (and what bench_chains.sh
# static configs span: start=(3,1,4), max=(7,1,8)).
SPEC_NS_START=3
SPEC_TK_START=1
SPEC_NDT_START=4
SPEC_NS_MAX=7
SPEC_TK_MAX=4
SPEC_NDT_MAX=8

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
            --output-dir "results/bench_dynamic" \
            || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo "  Dynamic Speculative Benchmark — $MODEL_TAG"
echo "  Model: $MODEL"
echo "  Draft: $DRAFT"
echo "  GPU:   $GPU_ID    Port: $PORT"
echo "  Policy envelope: start=($SPEC_TK_START,$SPEC_NS_START,$SPEC_NDT_START)"
echo "                   max=($SPEC_TK_MAX,$SPEC_NS_MAX,$SPEC_NDT_MAX)"
echo "  Timestamp: $TIMESTAMP"
echo "============================================================"

echo ""
echo ">>> Launching dynamic server (single top1-driven policy)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps "$SPEC_NS_START" \
    --speculative-eagle-topk "$SPEC_TK_START" \
    --speculative-num-draft-tokens "$SPEC_NDT_START" \
    --speculative-num-steps-max "$SPEC_NS_MAX" \
    --speculative-eagle-topk-max "$SPEC_TK_MAX" \
    --speculative-num-draft-tokens-max "$SPEC_NDT_MAX" \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "dynamic"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo "  DYNAMIC BENCHMARK COMPLETE — $TAG"
echo "  Results under results/bench_dynamic/"
echo "============================================================"
