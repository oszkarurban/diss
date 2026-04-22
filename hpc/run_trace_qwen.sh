#!/bin/bash
#SBATCH -J trace_qwen
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/trace_qwen_%j.out
#SBATCH --error=logs/trace_qwen_%j.err

# Phase A trace collection: 6 configs × 3 benchmarks × 20 q = 18 runs on Qwen.
# Mirror of hpc/run_trace_llama.sh with Qwen target + matched draft.

cd /rds/user/ou222/hpc-work/diss
eval "$(conda shell.bash hook)"
conda activate sglang-dev
source hpc/unload_prepare.sh
source hpc/unload_prepare.sh

mkdir -p logs results/traces

MODEL="Qwen/Qwen3-8B"
DRAFT="AngelSlim/Qwen3-8B_eagle3"
TAG="qwen"
PORT=30000
BENCHMARKS=("mtbench:20" "math500:20" "livecodebench:20")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

declare -a CONFIGS=(
    "static_3_1_4   --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
    "static_7_1_8   --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8"
    "static_7_4_8   --speculative-num-steps 7 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8"
    "static_6_10_60 --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 60"
    "v3_dynamic --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-num-steps-max 7 --speculative-eagle-topk-max 4 --speculative-num-draft-tokens-max 8 --enable-dynamic-speculative-decoding --dynamic-spec-policy v3"
    "v6_dynamic --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-num-steps-max 10 --speculative-eagle-topk-max 5 --speculative-num-draft-tokens-max 16 --enable-dynamic-speculative-decoding --dynamic-spec-policy v6"
)

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

for cfg in "${CONFIGS[@]}"; do
    label="$(echo "$cfg" | awk '{print $1}')"
    flags="$(echo "$cfg" | cut -d' ' -f2-)"

    echo ""
    echo "=========================================================="
    echo " Trace: $label @ $MODEL"
    echo "=========================================================="

    python3 -m sglang.launch_server \
        --model "$MODEL" \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$DRAFT" \
        $flags \
        --dynamic-spec-full-logging \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
        2>&1 | tee "logs/trace_${TAG}_${label}_${TIMESTAMP}.log" &
    SERVER_PID=$!
    wait_for_server

    for bench in "${BENCHMARKS[@]}"; do
        bench_name="${bench%:*}"
        echo "  >>> bench: $bench"
        python SpecForge/benchmarks/bench_eagle3.py \
            --model-path "$MODEL" \
            --port "$PORT" \
            --config-list 1,0,0,0 \
            --benchmark-list "$bench" \
            --dtype bfloat16 \
            --skip-launch-server \
            --name "trace_${TAG}_${label}" \
            2>&1 | tail -40 || echo "  ($bench bench failed, continuing)"

        out_file="results/traces/${TAG}_${label}_${bench_name}_${TIMESTAMP}_signals.jsonl.gz"
        python3 hpc/collect_traces.py \
            --benchmark "$bench" \
            --host 127.0.0.1 --port "$PORT" \
            --output "$out_file" \
            2>&1 | tail -20 || echo "  (trace extraction for $bench failed, continuing)"
    done

    kill_server $SERVER_PID
done

echo ""
echo "=========================================================="
echo " PHASE A TRACE COLLECTION COMPLETE — $TAG"
echo "=========================================================="
ls -la "results/traces/${TAG}_"*"${TIMESTAMP}"*.gz 2>/dev/null
