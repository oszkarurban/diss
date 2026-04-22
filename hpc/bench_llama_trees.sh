#!/bin/bash
#SBATCH -J bench_ll_trees
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/bench_ll_trees_%j.out
#SBATCH --error=logs/bench_ll_trees_%j.err

cd /rds/user/ou222/hpc-work/diss
eval "$(conda shell.bash hook)"
conda activate sglang-dev
source hpc/unload_prepare.sh
source hpc/unload_prepare.sh

REPO_ROOT="/rds/user/ou222/hpc-work/diss"

mkdir -p logs results

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
TAG="llama"
PORT=30000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
    echo "  Running benchmarks: $label"
    # Run each benchmark separately so one crash doesn't wipe the others
    for bench in $BENCHMARKS; do
        echo "    >>> $bench"
        python SpecForge/benchmarks/bench_eagle3.py \
            --model-path "$MODEL" \
            --port "$PORT" \
            --config-list 1,0,0,0 \
            --benchmark-list "$bench" \
            --dtype bfloat16 \
            --skip-launch-server \
            --name "${TAG}_${label}_${TIMESTAMP}" || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo " Tree Benchmarks — $MODEL"
echo " Timestamp: $TIMESTAMP"
echo "============================================================"

# 1. Static 7,4,8 (tree)
echo ""
echo ">>> [1/2] Vanilla EAGLE3 7,4,8 (tree)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 7 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_748"
kill_server $SERVER_PID

# 2. Static 10,6,60 (large tree, TALON-scale)
echo ""
echo ">>> [2/2] Vanilla EAGLE3 10,6,60 (TALON-scale tree)"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 10 --speculative-eagle-topk 6 --speculative-num-draft-tokens 60 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench "vanilla_10_6_60"
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo " TREE BENCHMARKS COMPLETE — $TAG"
echo "============================================================"
