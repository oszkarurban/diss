#!/bin/bash
# Demo: run a 2-question mtbench under the dynamic policy and print the bucket
# distribution of chosen (topk, num_steps, ndt) triples from the resulting JSONL.
# Proves the SpecForge patch is forwarding the server's spec_signal_log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results/demo

export HF_HOME="/workspace/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export CUDA_VISIBLE_DEVICES=0

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT="lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
PORT=30000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ">>> Launching server"
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-num-steps-max 7 \
    --speculative-eagle-topk-max 4 \
    --speculative-num-draft-tokens-max 8 \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=1800)
print('  Server ready.')
"

NAME="demo_dynspec_${TIMESTAMP}"
python SpecForge/benchmarks/bench_eagle3.py \
    --model-path "$MODEL" \
    --port "$PORT" \
    --config-list 1,0,0,0 \
    --benchmark-list "mtbench:2" \
    --dtype bfloat16 \
    --skip-launch-server \
    --name "$NAME" \
    --output-dir "results/demo"

echo ""
echo "=============================================================="
echo "Bucket distribution from $NAME"
echo "=============================================================="
python3 - <<PY
import json, glob
from collections import Counter
paths = sorted(glob.glob("results/demo/${NAME}_*.jsonl"))
if not paths:
    print("ERROR: no JSONL produced"); exit(1)
blob = json.load(open(paths[-1]))
steps = []
for bench, entries in blob.items():
    if not isinstance(entries, list):
        continue
    for entry in entries:
        for m in entry.get("metrics", []):
            for req in (m.get("spec_signal_log") or []):
                steps.extend(req)
if not steps:
    print("FAILURE: spec_signal_log is empty in the JSONL.")
    print("         → the SpecForge patch is not forwarding the log.")
    exit(2)
print(f"Parsed {len(steps)} per-step decisions.")
hist = Counter((s['chosen_topk'], s['chosen_num_steps'], s['chosen_num_draft_tokens']) for s in steps)
for bucket, n in sorted(hist.items(), key=lambda kv: -kv[1]):
    pct = 100.0 * n / len(steps)
    print(f"  (topk={bucket[0]}, num_steps={bucket[1]}, ndt={bucket[2]})  {n:5d}  {pct:5.1f}%")
distinct = len(hist)
print()
print(f"Distinct tree configs seen: {distinct}")
print("  → policy IS altering tree size." if distinct > 1 else "  → policy chose a single config on this small run (expected if run was too short).")
PY
