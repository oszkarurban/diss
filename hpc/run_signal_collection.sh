#!/bin/bash
#!#############################################################
#!#### Signal Data Collection for V3 Analysis             #####
#!#############################################################

#SBATCH -J signal_collect               # Name of the job
#SBATCH -A MASCOLO-SL2-GPU             # SL2 Account
#SBATCH -p ampere                       # Wilkes3 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/signal_collect_%j.out
#SBATCH --error=logs/signal_collect_%j.err

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
NUM_SAMPLES=10
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

#! ############################################################
#! Model Pairs
#! ############################################################

declare -A DRAFT_MODELS
DRAFT_MODELS["meta-llama/Llama-3.1-8B-Instruct"]="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
DRAFT_MODELS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
DRAFT_MODELS["Qwen/Qwen3-8B"]="AngelSlim/Qwen3-8B_eagle3"

declare -A MODEL_TAGS
MODEL_TAGS["meta-llama/Llama-3.1-8B-Instruct"]="llama"
MODEL_TAGS["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek"
MODEL_TAGS["Qwen/Qwen3-8B"]="qwen"

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

launch_static() {
    local model="$1"
    local draft="$2"

    python3 -m sglang.launch_server \
        --model "$model" \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$draft" \
        --speculative-num-steps 7 \
        --speculative-eagle-topk 4 \
        --speculative-num-draft-tokens 8 \
        --enable-dynamic-speculative-decoding \
        --speculative-num-steps-max 7 \
        --speculative-eagle-topk-max 4 \
        --speculative-num-draft-tokens-max 8 \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
}

launch_dynamic() {
    local model="$1"
    local draft="$2"

    python3 -m sglang.launch_server \
        --model "$model" \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path "$draft" \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --speculative-num-steps-max 7 \
        --speculative-eagle-topk-max 4 \
        --speculative-num-draft-tokens-max 8 \
        --enable-dynamic-speculative-decoding \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
}

collect_signals() {
    local output="$1"
    echo "  Collecting signals → $output"
    python3 test_signal_collection.py \
        --port "$PORT" \
        --num-samples "$NUM_SAMPLES" \
        --output "$output"
}

#! ############################################################
#! Data Collection Matrix
#!
#! For each model pair: static 7,4,8 + dynamic V3 3,1,4→7,4,8
#! Static uses --enable-dynamic-speculative-decoding with
#! start=max=7,4,8 so signals are collected but config is fixed.
#! ############################################################

echo ""
echo "============================================================"
echo " Signal Data Collection for V3 Analysis"
echo " Timestamp: $TIMESTAMP"
echo " Samples: $NUM_SAMPLES MT-Bench questions per run"
echo "============================================================"

for MODEL in "meta-llama/Llama-3.1-8B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "Qwen/Qwen3-8B"; do
    DRAFT="${DRAFT_MODELS[$MODEL]}"
    TAG="${MODEL_TAGS[$MODEL]}"

    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Model: $MODEL"
    echo "  Draft: $DRAFT"
    echo "══════════════════════════════════════════════════"

    # 1. Static 7,4,8 with signal logging
    echo ""; echo ">>> Static 7,4,8 — $TAG"
    launch_static "$MODEL" "$DRAFT"
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    collect_signals "signal_data_analysis_static_${TAG}_${TIMESTAMP}.json"
    kill_server $SERVER_PID

    # 2. Dynamic V3 3,1,4 → 7,4,8
    echo ""; echo ">>> Dynamic V3 3,1,4→7,4,8 — $TAG"
    launch_dynamic "$MODEL" "$DRAFT"
    SERVER_PID=$!
    wait_for_server $SERVER_PID
    collect_signals "signal_data_analysis_dynamic_${TAG}_${TIMESTAMP}.json"
    kill_server $SERVER_PID

done

echo ""
echo "============================================================"
echo " ALL SIGNAL COLLECTION COMPLETE"
echo " Files: signal_data_analysis_*_${TIMESTAMP}.json"
echo "============================================================"

# Run analysis on collected data
echo ""
echo "Running analysis..."

LLAMA_DYN="signal_data_analysis_dynamic_llama_${TIMESTAMP}.json"
LLAMA_STA="signal_data_analysis_static_llama_${TIMESTAMP}.json"
DS_DYN="signal_data_analysis_dynamic_deepseek_${TIMESTAMP}.json"
DS_STA="signal_data_analysis_static_deepseek_${TIMESTAMP}.json"
QWEN_DYN="signal_data_analysis_dynamic_qwen_${TIMESTAMP}.json"
QWEN_STA="signal_data_analysis_static_qwen_${TIMESTAMP}.json"

python3 analysis_v3_deep.py \
    --dynamic "$LLAMA_DYN" "$DS_DYN" "$QWEN_DYN" \
    --static "$LLAMA_STA" "$DS_STA" "$QWEN_STA" \
    --label "Llama 3.1 8B" "DeepSeek-R1 8B" "Qwen3 8B" \
    --output "results/analysis_v3_deep_${TIMESTAMP}.md"

echo "Analysis report: results/analysis_v3_deep_${TIMESTAMP}.md"
echo "Done."
