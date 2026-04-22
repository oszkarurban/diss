#!/bin/bash
#!#############################################################
#!#### SLURM Header Definitions                        ########
#!#############################################################

#SBATCH -J specforge_eval              # Name of the job
#SBATCH -A MASCOLO-SL2-GPU            # SL2 Account
#SBATCH -p ampere                      # Wilkes3 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/specforge_eval_%j.out
#SBATCH --error=logs/specforge_eval_%j.err

#! ############################################################
#! Environment Setup
#! ############################################################

source hpc/unload_prepare.sh
conda activate sglang-dev

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs results

PORT=30000
SERVER_TIMEOUT=600
BENCH_DIR="$REPO_ROOT/SpecForge/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

#! ############################################################
#! Model Pairs
#! ############################################################

declare -A DRAFT_MODELS
DRAFT_MODELS["meta-llama/Llama-3.1-8B-Instruct"]="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
DRAFT_MODELS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"

declare -A MODEL_TAGS
MODEL_TAGS["meta-llama/Llama-3.1-8B-Instruct"]="llama8b"
MODEL_TAGS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek8b"

#! ############################################################
#! Helper Functions
#! ############################################################

wait_for_server() {
    echo "  Waiting up to ${SERVER_TIMEOUT}s for server (PID $1)..."
    python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=$SERVER_TIMEOUT)
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
    local model_path="$1"
    local label="$2"
    echo "  Running SpecForge benchmark: $label"
    cd "$BENCH_DIR"
    python bench_eagle3.py \
        --model-path "$model_path" \
        --port "$PORT" \
        --config-list 1,0,0,0 \
        --benchmark-list mtbench:80 \
        --dtype bfloat16 \
        --skip-launch-server
    cd "$REPO_ROOT"
    # Copy result file to named location
    local latest=$(ls -t "$BENCH_DIR"/results_*.jsonl 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        cp "$latest" "results/specforge_${label}_${TIMESTAMP}.jsonl"
        echo "  Result saved: results/specforge_${label}_${TIMESTAMP}.jsonl"
    fi
}

launch_vanilla() {
    local model="$1"
    local draft="$2"
    local steps="$3"
    local topk="$4"
    local ndt="$5"

    python3 -m sglang.launch_server \
        --model "$model" \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$draft" \
        --speculative-num-steps "$steps" \
        --speculative-eagle-topk "$topk" \
        --speculative-num-draft-tokens "$ndt" \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
}

launch_dynamic() {
    local model="$1"
    local draft="$2"
    local steps="$3"
    local topk="$4"
    local ndt="$5"
    local steps_max="$6"
    local topk_max="$7"
    local ndt_max="$8"

    python3 -m sglang.launch_server \
        --model "$model" \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$draft" \
        --speculative-num-steps "$steps" \
        --speculative-eagle-topk "$topk" \
        --speculative-num-draft-tokens "$ndt" \
        --speculative-num-steps-max "$steps_max" \
        --speculative-eagle-topk-max "$topk_max" \
        --speculative-num-draft-tokens-max "$ndt_max" \
        --enable-dynamic-speculative-decoding \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
}

#! ############################################################
#! Benchmark Matrix
#!
#! For each model pair we run:
#!
#! BASELINES:
#!   1. Vanilla EAGLE3 3,1,4  — starting config (SpecForge default)
#!   2. Best SpecForge chain  — 5,1,6 (best throughput from SpecForge)
#!   3. Best TALON-scale tree — 7,4,8 (topk=4 tree, closest to
#!      TALON's K=10,D=8,N=60 within our CUDA graph constraints)
#!
#! OUR METHOD (V3 dynamic, adapts tree shape per decode step):
#!   4. V3 start=3,1,4 max=5,1,6  — can V3 match best chain?
#!   5. V3 start=3,1,4 max=7,4,8  — V3 with full range (our best)
#!   6. V3 start=5,1,6 max=7,4,8  — V3 from higher starting point
#!
#! ############################################################

echo ""
echo "============================================================"
echo " SpecForge Official Evaluation"
echo " Timestamp: $TIMESTAMP"
echo " Benchmark: MT-Bench, 80 questions, batch_size=1, temp=0"
echo "============================================================"

for MODEL in "meta-llama/Llama-3.1-8B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"; do
    DRAFT="${DRAFT_MODELS[$MODEL]}"
    TAG="${MODEL_TAGS[$MODEL]}"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Model: $MODEL"
    echo "  Draft: $DRAFT"
    echo "══════════════════════════════════════════════════"

    # ── BASELINES ──

    # 1. Vanilla EAGLE3 3,1,4 (starting config)
    echo ""; echo ">>> [1/6] Vanilla EAGLE3 3,1,4 (baseline)"
    launch_vanilla "$MODEL" "$DRAFT" 3 1 4
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    run_bench "$MODEL" "${TAG}_vanilla_314"
    kill_server $SERVER_PID

    # 2. Best SpecForge chain: 5,1,6
    echo ""; echo ">>> [2/6] Vanilla EAGLE3 5,1,6 (best SpecForge chain)"
    launch_vanilla "$MODEL" "$DRAFT" 5 1 6
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    run_bench "$MODEL" "${TAG}_vanilla_516"
    kill_server $SERVER_PID

    # 3. TALON-scale tree: 7,4,8
    echo ""; echo ">>> [3/6] Vanilla EAGLE3 7,4,8 (TALON-scale tree)"
    launch_vanilla "$MODEL" "$DRAFT" 7 4 8
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    run_bench "$MODEL" "${TAG}_vanilla_748"
    kill_server $SERVER_PID

    # ── OUR METHOD (V3 Dynamic) ──

    # 4. V3 start=3,1,4 max=5,1,6 (can V3 match best chain?)
    echo ""; echo ">>> [4/6] V3 Dynamic 3,1,4 → 5,1,6"
    launch_dynamic "$MODEL" "$DRAFT" 3 1 4 5 1 6
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    run_bench "$MODEL" "${TAG}_v3_314_516"
    kill_server $SERVER_PID

    # 5. V3 start=3,1,4 max=7,4,8 (full range — our best result)
    echo ""; echo ">>> [5/6] V3 Dynamic 3,1,4 → 7,4,8"
    launch_dynamic "$MODEL" "$DRAFT" 3 1 4 7 4 8
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    run_bench "$MODEL" "${TAG}_v3_314_748"
    kill_server $SERVER_PID

    # 6. V3 start=5,1,6 max=7,4,8 (from higher starting point)
    echo ""; echo ">>> [6/6] V3 Dynamic 5,1,6 → 7,4,8"
    launch_dynamic "$MODEL" "$DRAFT" 5 1 6 7 4 8
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    run_bench "$MODEL" "${TAG}_v3_516_748"
    kill_server $SERVER_PID

done

#! ############################################################
#! Summary
#! ############################################################

echo ""
echo "============================================================"
echo " ALL BENCHMARKS COMPLETE"
echo " Results in: results/specforge_*_${TIMESTAMP}.jsonl"
echo "============================================================"

echo ""
echo "Results summary:"
echo "──────────────────────────────────────────────────────────────────────────"
printf "  %-40s %10s %10s %10s\n" "Config" "TP(tok/s)" "AccLen" "Latency(s)"
echo "──────────────────────────────────────────────────────────────────────────"
for f in results/specforge_*_${TIMESTAMP}.jsonl; do
    label=$(basename "$f" .jsonl | sed "s/specforge_//" | sed "s/_${TIMESTAMP}//")
    tp=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['mtbench'][0]['metrics'][0]['output_throughput']:.1f}\")" 2>/dev/null || echo "ERR")
    al=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['mtbench'][0]['metrics'][0]['accept_length']:.3f}\")" 2>/dev/null || echo "ERR")
    lat=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['mtbench'][0]['metrics'][0]['latency']:.1f}\")" 2>/dev/null || echo "ERR")
    printf "  %-40s %10s %10s %10s\n" "$label" "$tp" "$al" "$lat"
done
