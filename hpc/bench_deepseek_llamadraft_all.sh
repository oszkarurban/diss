#!/bin/bash
# Combined: chains + trees + vanilla-no-spec for DeepSeek target with Llama EAGLE3 draft.
# Run directly (not via sbatch): ./hpc/bench_deepseek_llamadraft_all.sh

set -u

cd /rds/user/ou222/hpc-work/diss


REPO_ROOT="/rds/user/ou222/hpc-work/diss"

mkdir -p logs results

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# Mismatched draft: Llama-3.1 EAGLE3 draft on DeepSeek target
# (matches past signal_data_*_deepseek8b_llamadraft.json experiments)
DRAFT="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
TAG="deepseek_llamadraft"
TAG_NOSPEC="deepseek"
PORT=30000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LOG_OUT="logs/bench_${TAG}_all_${TIMESTAMP}.out"
LOG_ERR="logs/bench_${TAG}_all_${TIMESTAMP}.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)
echo "Logging stdout → $LOG_OUT"
echo "Logging stderr → $LOG_ERR"

BENCHMARKS="mtbench:80 gsm8k:200 math500:200 humaneval:164 livecodebench:200"

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
    local benches="$2"
    local tag="$3"
    echo "  Running benchmarks: $label"
    for bench in $benches; do
        echo "    >>> $bench"
        python SpecForge/benchmarks/bench_eagle3.py \
            --model-path "$MODEL" \
            --port "$PORT" \
            --config-list 1,0,0,0 \
            --benchmark-list "$bench" \
            --dtype bfloat16 \
            --skip-launch-server \
            --name "${tag}_${label}_${TIMESTAMP}" || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo " Full DeepSeek bench — $MODEL"
echo " Timestamp: $TIMESTAMP"
echo "============================================================"

########## VANILLA AUTOREGRESSIVE (NO SPEC) ##########
echo ""
echo ">>> [1/3] Vanilla autoregressive (NO spec)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_nospec" "$BENCHMARKS" "$TAG_NOSPEC"
kill_server $SERVER_PID

########## CHAINS ##########
echo ""
echo ">>> [2/3] Vanilla EAGLE3 3,1,4 (small chain)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_314" "$BENCHMARKS" "$TAG"
kill_server $SERVER_PID

echo ""
echo ">>> [3/3] Vanilla EAGLE3 7,1,8 (deep chain)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_718" "$BENCHMARKS" "$TAG"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo " ALL BENCHMARKS COMPLETE — $TAG / $TAG_NOSPEC"
echo "============================================================"
