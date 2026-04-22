#!/bin/bash
# Smoke test for --dynamic-spec-full-logging on a STATIC EAGLE3 run.
# Expected runtime on a warm node: ~5 min. Cold node: ~8 min (import).
#
# Validates the 4 code patches (server_args, eagle_info, dynamic_spec,
# eagle_worker) by launching static (3,1,4) Llama EAGLE3 + Llama matched
# draft with --dynamic-spec-full-logging, sending 5 MT-Bench questions,
# and confirming per-step spec_signal_log entries emerge with non-zero
# top1_prob / target_top1_prob / rolling_accept_rate.
#
# Run inside an active sintr session:
#   conda activate sglang-dev
#   source hpc/unload_prepare.sh
#   bash hpc/smoke_test_signal_logging.sh
#
# Exit codes:
#   0 — signals emitted and pass sanity checks
#   1 — failure (see stdout / server log)

set -euo pipefail

REPO_ROOT="/rds/user/ou222/hpc-work/diss"
cd "$REPO_ROOT"
mkdir -p logs results

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SERVER_LOG="logs/smoke_static_314_${TIMESTAMP}.log"
OUT_JSON="results/smoke_static_314_llama_${TIMESTAMP}.json"
PORT=30000

echo "=== Smoke test: static (3,1,4) + --dynamic-spec-full-logging ==="
echo "Repo root: $REPO_ROOT"
echo "Server log: $SERVER_LOG"
echo "Signal output: $OUT_JSON"
echo ""

# Kill anything stale on our port
pkill -9 -u ou222 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 2

# Launch static EAGLE3 server with signal logging on (NO --enable-dynamic-speculative-decoding)
echo ">>> Launching server..."
cd /tmp  # avoid CWD shadowing of diss/sglang/
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --dynamic-spec-full-logging \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
    > "$REPO_ROOT/$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cd "$REPO_ROOT"
echo "Server PID: $SERVER_PID"

# Wait for the server to be ready (15 min — tolerates cold-Lustre + JIT compile)
echo ">>> Waiting for server ready (timeout: 15 min)..."
python3 -c "
from sglang.utils import wait_for_server
wait_for_server('http://localhost:$PORT', timeout=900)
print('  Server ready.')
"

# Run signal collection on 5 mtbench questions
echo ">>> Running signal collection (5 mtbench questions)..."
python3 test_signal_collection.py \
    --host localhost --port "$PORT" \
    --dataset mtbench --num-samples 5 \
    --output "$OUT_JSON" \
    || { echo "[FAIL] signal collection errored"; kill -9 $SERVER_PID; exit 1; }

# Kill server
echo ">>> Killing server..."
kill -9 $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 2

# Post-check: the JSON output must exist and contain non-empty signal logs
echo ">>> Sanity check output..."
python3 - <<PYEOF
import json, sys
with open("$OUT_JSON") as f:
    data = json.load(f)

# test_signal_collection.py writes {"per_turn_logs": [{"signals": [...], ...}, ...]}
per_turn = [t.get("signals", []) for t in data.get("per_turn_logs", [])]
total = sum(len(t) for t in per_turn)
turns = len(per_turn)
turns_nonempty = sum(1 for t in per_turn if t)

print(f"  Turns: {turns} ({turns_nonempty} non-empty)")
print(f"  Total logged steps: {total}")

if total == 0:
    print("  [FAIL] No signal log entries collected.")
    print("         Likely cause: code patch didn't hit the static path,")
    print("         OR --dynamic-spec-full-logging not recognised by the build.")
    sys.exit(1)

# Inspect first non-empty entry
for turn in per_turn:
    if turn:
        step0 = turn[0]
        print(f"  Sample step[0]: {json.dumps(step0, indent=None)[:200]}")
        for k in ("top1_prob", "target_top1_prob", "rolling_accept_rate",
                  "chosen_topk", "chosen_num_steps", "chosen_num_draft_tokens"):
            if k not in step0:
                print(f"  [FAIL] Missing key: {k}")
                sys.exit(1)
        # On a static (3,1,4) run, chosen_* must equal 1/3/4 (policy did not fire)
        if (step0["chosen_topk"], step0["chosen_num_steps"], step0["chosen_num_draft_tokens"]) != (1, 3, 4):
            print(f"  [FAIL] chosen_* should be (1,3,4) on static run, got "
                  f"({step0['chosen_topk']},{step0['chosen_num_steps']},{step0['chosen_num_draft_tokens']})")
            sys.exit(1)
        break

print("  [OK] spec_signal_log populated on static run with expected (1,3,4) config.")
PYEOF

echo ""
echo "=== Smoke test PASSED ==="
echo "Signal JSON: $OUT_JSON"
echo "Server log:  $SERVER_LOG"
