#!/bin/bash
# FLy (training-free loosely speculative decoding) throughput benchmark.
# Mirrors hpc/bench_dynamic.sh but launches with --enable-fly on top of a
# static EAGLE3 chain.  Paper defaults for theta (0.3) and W (6). K=15 chain
# matches the FLy paper's draft-token count for 70B targets.
#
# Usage:  bash hpc/bench_fly.sh {llama|deepseek|qwen}
set -euo pipefail

MODEL_TAG="${1:-llama}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/bench_fly

# HF_HOME, TMPDIR, and JIT cache paths are set by hpc/unload_prepare.sh
# (HF weights live on RDS; JIT caches live on /local). Keep this script's
# env hygiene in sync with that file.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Benchmarks (override with BENCHMARKS env var for a subset, e.g.
#   BENCHMARKS="gsm8k:100 math500:100" bash hpc/bench_fly.sh llama
# )
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

export CUDA_VISIBLE_DEVICES="$GPU_ID"
TAG="${MODEL_TAG}"

# K=15 chain to match FLy paper's primary config for 70B targets (Table 4).
# Chain (topk=1) is required by FLy v1.  ndt = num_steps + 1.
SPEC_NS=15
SPEC_TK=1
SPEC_NDT=16

# FLy paper defaults (Table 4 ablation picks).
FLY_THETA=0.3
FLY_W=6

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
            --output-dir "results/bench_fly" \
            || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo "  FLy Benchmark — $MODEL_TAG"
echo "  Model: $MODEL"
echo "  Draft: $DRAFT"
echo "  GPU:   $GPU_ID    Port: $PORT"
echo "  Chain config:  num_steps=$SPEC_NS  topk=$SPEC_TK  ndt=$SPEC_NDT"
echo "  FLy params:    theta=$FLY_THETA  W=$FLY_W"
echo "  Timestamp: $TIMESTAMP"
echo "============================================================"

SERVER_LOG="logs/bench_fly_${TAG}_${TIMESTAMP}.log"

echo ""
echo ">>> Launching EAGLE3 server with --enable-fly"
echo "    server log: $SERVER_LOG"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps "$SPEC_NS" \
    --speculative-eagle-topk "$SPEC_TK" \
    --speculative-num-draft-tokens "$SPEC_NDT" \
    --enable-fly \
    --fly-entropy-threshold "$FLY_THETA" \
    --fly-window-length "$FLY_W" \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "fly"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo "  FLy BENCHMARK COMPLETE — $TAG"
echo "  Results under results/bench_fly/"
echo "  Server log at $SERVER_LOG"
echo "============================================================"
