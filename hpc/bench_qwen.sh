#!/bin/bash
#SBATCH -J bench_qwen
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/bench_qwen_%j.out
#SBATCH --error=logs/bench_qwen_%j.err

source hpc/unload_prepare.sh
conda activate sglang-dev

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results

MODEL="Qwen/Qwen3-8B"
DRAFT="AngelSlim/Qwen3-8B_eagle3"
TAG="qwen"
PORT=30000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BENCHMARKS="mtbench:80 gsm8k:200 math500:200 humaneval:164 livecodebench:200 gpqa:200 financeqa:200"

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
    python SpecForge/benchmarks/bench_eagle3.py \
        --model-path "$MODEL" \
        --port "$PORT" \
        --config-list 1,0,0,0 \
        --benchmark-list $BENCHMARKS \
        --dtype bfloat16 \
        --skip-launch-server \
        --name "${TAG}_${label}_${TIMESTAMP}"
}

echo "============================================================"
echo " Vanilla EAGLE3 Benchmarks — $MODEL"
echo " Timestamp: $TIMESTAMP"
echo "============================================================"

# 1. Static 3,1,4 (small chain)
echo ""
echo ">>> [1/4] Vanilla EAGLE3 3,1,4 (small chain)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_314"
kill_server $SERVER_PID

# 2. Static 7,1,8 (deep chain)
echo ""
echo ">>> [2/4] Vanilla EAGLE3 7,1,8 (deep chain)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_718"
kill_server $SERVER_PID

# 3. Static 7,4,8 (tree)
echo ""
echo ">>> [3/4] Vanilla EAGLE3 7,4,8 (tree)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 7 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_748"
kill_server $SERVER_PID

# 4. Static 10,6,60 (large tree, TALON-scale)
echo ""
echo ">>> [4/4] Vanilla EAGLE3 10,6,60 (TALON-scale tree)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 10 --speculative-eagle-topk 6 --speculative-num-draft-tokens 60 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_10_6_60"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo " ALL BENCHMARKS COMPLETE — $TAG"
echo " Results: SpecForge/benchmarks/results_*${TIMESTAMP}*.jsonl"
echo "============================================================"
