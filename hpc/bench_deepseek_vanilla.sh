#!/bin/bash
#SBATCH -J bench_ds_van
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/bench_ds_van_%j.out
#SBATCH --error=logs/bench_ds_van_%j.err

cd /rds/user/ou222/hpc-work/diss
eval "$(conda shell.bash hook)"
conda activate sglang-dev
source hpc/unload_prepare.sh
source hpc/unload_prepare.sh

mkdir -p logs results

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TAG="deepseek"
PORT=30000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

BENCHMARKS="livecodebench:200"  # only re-run the one that hit wall time; other 4 datasets done
# BENCHMARKS="mtbench:80 gsm8k:200 math500:200 humaneval:164 livecodebench:200"

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
        echo "    >>> $bench"
        python SpecForge/benchmarks/bench_eagle3.py \
            --model-path "$MODEL" \
            --port "$PORT" \
            --config-list 1,0,0,0 \
            --benchmark-list "$bench" \
            --dtype bfloat16 \
            --skip-launch-server \
            --name "${TAG}_vanilla_nospec_${TIMESTAMP}" || echo "    ($bench failed, continuing)"
    done
}

echo "============================================================"
echo " Vanilla autoregressive (NO spec) — $MODEL"
echo "============================================================"

python3 -m sglang.launch_server \
    --model "$MODEL" \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
wait_for_server $SERVER_PID
run_bench
kill_server $SERVER_PID

echo ""
echo "============================================================"
echo " VANILLA NO-SPEC COMPLETE — $TAG"
echo "============================================================"
