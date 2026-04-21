#!/bin/bash
# Livecodebench-only calibration run.
# Mirrors hpc/calibrate_top1_dynamic.sh but skips mtbench/math500 so the
# livecodebench loader bug can be iterated on without re-running the working
# benchmarks each time.
#
# Usage:  bash hpc/calibrate_top1_livecodebench.sh {llama|deepseek|qwen}
set -euo pipefail

MODEL_TAG="${1:-llama}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/calibration

export HF_HOME="/workspace/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_Q=20

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

echo "============================================================"
echo "  LiveCodeBench-only calibration — $MODEL_TAG"
echo "  Model: $MODEL"
echo "  Draft: $DRAFT"
echo "  GPU:   $GPU_ID    Port: $PORT"
echo "  Timestamp: $TIMESTAMP"
echo "============================================================"

SPEC_NS_START=3
SPEC_TK_START=1
SPEC_NDT_START=4
SPEC_NS_MAX=7
SPEC_TK_MAX=4
SPEC_NDT_MAX=8

echo ">>> Launching server with single top1-driven dynamic policy"
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
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!

cleanup() {
    echo "  Killing server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "  Waiting for server (PID $SERVER_PID)..."
python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=1800)
print('  Server ready.')
"

BENCH="livecodebench:$NUM_Q"
OUT_TAG="${MODEL_TAG}_livecodebench_${TIMESTAMP}"
echo ""
echo ">>> Benchmark: $BENCH (output tag: $OUT_TAG)"
python SpecForge/benchmarks/bench_eagle3.py \
    --model-path "$MODEL" \
    --port "$PORT" \
    --config-list 1,0,0,0 \
    --benchmark-list "$BENCH" \
    --dtype bfloat16 \
    --skip-launch-server \
    --name "calibration_$OUT_TAG" \
    --output-dir "results/calibration"

echo ""
echo "============================================================"
echo "  LIVECODEBENCH RUN COMPLETE — $MODEL_TAG"
echo "  Results under results/calibration/"
echo "============================================================"
