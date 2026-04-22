#!/bin/bash
# Strict EAGLE3 baseline at the same K=15 chain config as bench_fly.sh.
# This is the FLy denominator for recovery% and the throughput reference
# for measuring FLy's speedup contribution.
#
# Usage:  bash hpc/bench_chain_k15.sh {llama|deepseek|qwen}
set -euo pipefail

MODEL_TAG="${1:-llama}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/bench_chain_k15

# HF_HOME, TMPDIR, and JIT cache paths are set by hpc/unload_prepare.sh.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Benchmarks (override with BENCHMARKS env var for a subset, e.g.
#   BENCHMARKS="gsm8k:100 math500:100" bash hpc/bench_chain_k15.sh llama
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

SPEC_NS=15
SPEC_TK=1
SPEC_NDT=16

wait_for_server() {
    python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=1800)
print('  Server ready.')
"
}

kill_server() {
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
            --output-dir "results/bench_chain_k15" \
            || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo "  Strict EAGLE3 baseline @ K=15 chain — $MODEL_TAG"
echo "  Model: $MODEL   Draft: $DRAFT"
echo "  Chain: num_steps=$SPEC_NS  topk=$SPEC_TK  ndt=$SPEC_NDT"
echo "  Timestamp: $TIMESTAMP"
echo "============================================================"

SERVER_LOG="logs/bench_chain_k15_${TAG}_${TIMESTAMP}.log"

python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps "$SPEC_NS" \
    --speculative-eagle-topk "$SPEC_TK" \
    --speculative-num-draft-tokens "$SPEC_NDT" \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server
run_bench "chain_k15"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo "  STRICT K=15 BASELINE COMPLETE — $TAG"
echo "  Results under results/bench_chain_k15/"
echo "============================================================"
