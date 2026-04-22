#!/usr/bin/env python3
"""
test_signal_collection.py — Signal and token-level data collection for V6
dynamic speculative decoding.

Sends MT-Bench requests to a running SGLang server launched with
--enable-dynamic-speculative-decoding and extracts:
  1. Per-step signal logs.  V6 layout per step:
       * 3 signals: top1_prob, target_top1_prob, rolling_accept_rate
       * 2 adaptive thresholds: top1_threshold, target_threshold
       * 3 chosen_* fields: chosen_topk, chosen_num_steps, chosen_num_draft_tokens
       * accept_length (appended by the server pipeline after verify)
  2. Per-step token-level data (draft/accepted/rejected tokens, tree structure)

Usage (server must be running):
    python test_signal_collection.py --port 30000 --num-samples 10 --output signal_data.json
"""

import argparse
import json
import os
import sys
import time

# ── Path setup (must precede all sglang / benchmarker imports) ──────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_REPO_ROOT, "sglang", "python"),
    os.path.join(_REPO_ROOT, "SpecForge", "benchmarks"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from argparse import Namespace
from typing import Any, Dict, List

import numpy as np
import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.test.test_utils import select_sglang_backend

from benchmarker import BENCHMARKS

# V6 signals (must match sglang/python/sglang/srt/speculative/dynamic_spec.py:V6Signals).
V6_SIGNAL_KEYS = [
    "top1_prob",
    "target_top1_prob",
    "rolling_accept_rate",
]

# V6 adaptive thresholds (one running-median per axis of the 2×2).
V6_THRESHOLD_KEYS = [
    "top1_threshold",
    "target_threshold",
]

CONFIG_KEYS = [
    "chosen_topk",
    "chosen_num_steps",
    "chosen_num_draft_tokens",
]

# Token-level meta_info keys (already returned by server, just need extraction)
TOKEN_LEVEL_KEYS = [
    "spec_draft_tokens",
    "spec_accepted_tokens_log",
    "spec_rejected_tokens_log",
    "spec_accept_index_log",
    "spec_topk",
    "spec_num_steps",
    "spec_draft_token_num",
    "spec_retrive_next_token",
    "spec_retrive_next_sibling",
]


# ── Extraction ──────────────────────────────────────────────────────────────


def extract_signal_logs(
    states: List[Any], answer_keys: List[str]
) -> List[List[Dict]]:
    """Extract spec_signal_log from every (state, answer_key) pair.

    Returns a list of per-turn signal logs.  Each entry is a list of step
    dicts (may be empty if the turn produced few tokens or was in warmup).
    """
    logs = []
    for state in states:
        for key in answer_keys:
            meta = state.get_meta_info(key)
            if meta is None:
                logs.append([])
                continue
            sig_log = meta.get("spec_signal_log", [])
            logs.append(sig_log)
    return logs


def extract_token_level_data(
    states: List[Any], answer_keys: List[str]
) -> List[Dict[str, Any]]:
    """Extract per-step token-level data from every (state, answer_key) pair.

    Returns a list of per-turn dicts, each containing lists of per-step data
    for draft tokens, accepted/rejected tokens, tree structure, and server
    config. Keys match TOKEN_LEVEL_KEYS from the HTTP response meta_info.
    """
    all_turns = []
    for state in states:
        for key in answer_keys:
            meta = state.get_meta_info(key)
            if meta is None:
                all_turns.append({k: [] for k in TOKEN_LEVEL_KEYS})
                continue
            turn_data = {}
            for tk in TOKEN_LEVEL_KEYS:
                turn_data[tk] = meta.get(tk, [])
            all_turns.append(turn_data)
    return all_turns


# ── Analysis ────────────────────────────────────────────────────────────────


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values) if values else np.array([0.0])
    return {
        "n": len(values),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def print_signal_stats(all_steps: List[Dict]):
    print(f"\n  Extracted {len(all_steps)} signal log entries.\n")
    header = f"  {'Signal':<28s} {'N':>5s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'Std':>10s}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for key in V6_SIGNAL_KEYS + V6_THRESHOLD_KEYS + CONFIG_KEYS:
        values = [step[key] for step in all_steps if key in step]
        if not values:
            print(f"  {key:<28s}     0       —        —        —        —")
            continue
        s = compute_stats(values)
        print(
            f"  {key:<28s} {s['n']:>5d} {s['min']:>10.4f} {s['max']:>10.4f} "
            f"{s['mean']:>10.4f} {s['std']:>10.4f}"
        )
    print()


def print_sample_logs(per_turn_logs: List[List[Dict]], n: int = 3):
    shown = 0
    for i, log in enumerate(per_turn_logs):
        if not log:
            continue
        print(f"  Turn {i}: {len(log)} steps")
        # Show first and last step
        print(f"    step 0:  {json.dumps(log[0], indent=None)}")
        if len(log) > 1:
            print(f"    step -1: {json.dumps(log[-1], indent=None)}")
        print()
        shown += 1
        if shown >= n:
            break


# ── Verification ────────────────────────────────────────────────────────────


def verify_signals(all_steps: List[Dict]) -> bool:
    if not all_steps:
        print("  [FAIL] No signal log entries collected at all!")
        print("         Is the server running with --enable-dynamic-speculative-decoding?")
        return False

    all_ok = True
    n = len(all_steps)

    # Check 1: V6 signals non-zero in >= 80% of steps (post-warmup).
    # top1_prob and target_top1_prob can legitimately be 0.0 on the very
    # first step (before any prior draft/verify produces values); RAR is
    # initialised to 0.5 per-Req so it's always non-zero.
    print("  ── Non-zero checks (V6 signals) ──")
    for key in V6_SIGNAL_KEYS:
        values = [step.get(key, 0.0) for step in all_steps]
        nonzero = sum(1 for v in values if abs(v) > 1e-10)
        pct = 100.0 * nonzero / n if n else 0
        passed = pct >= 80.0
        tag = "[OK]" if passed else "[FAIL]"
        print(f"    {tag}  {key}: non-zero in {nonzero}/{n} steps ({pct:.1f}%)")
        all_ok &= passed

    # Check 2: V6 signals have std > 0 (vary between steps).
    print("  ── Variation checks (V6 signals) ──")
    for key in V6_SIGNAL_KEYS:
        values = [step.get(key, 0.0) for step in all_steps]
        std = float(np.std(values)) if len(values) > 1 else 0.0
        passed = std > 1e-10
        tag = "[OK]" if passed else "[WARN]"
        print(f"    {tag}  {key}: std = {std:.6f}")

    # Check 3: adaptive thresholds stay in [0, 1] and move over the trace.
    print("  ── Adaptive-threshold checks ──")
    for key in V6_THRESHOLD_KEYS:
        values = [step.get(key) for step in all_steps if key in step]
        if not values:
            print(f"    [FAIL]  {key}: not present in log")
            all_ok = False
            continue
        in_range = all(0.0 <= v <= 1.0 for v in values)
        std = float(np.std(values)) if len(values) > 1 else 0.0
        tag_rng = "[OK]" if in_range else "[FAIL]"
        tag_var = "[OK]" if std > 1e-10 else "[WARN]"
        print(
            f"    {tag_rng}  {key}: in [0,1] "
            f"(min={min(values):.4f} max={max(values):.4f})   "
            f"{tag_var}  std={std:.6f}"
        )
        all_ok &= in_range

    # Check 4: chosen params have > 1 distinct value (policy exercised all cells).
    print("  ── Config variation checks ──")
    for key in CONFIG_KEYS:
        values = set(step.get(key) for step in all_steps if key in step)
        passed = len(values) > 1
        tag = "[OK]" if passed else "[WARN]"
        print(f"    {tag}  {key}: {len(values)} distinct values: {sorted(values)}")

    return all_ok


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Test dynamic spec signal collection against a running server."
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of questions from the dataset to use.",
    )
    p.add_argument(
        "--dataset", default="mtbench",
        choices=["mtbench", "math500", "livecodebench", "humaneval", "gsm8k", "gpqa"],
        help="Which benchmark dataset to run signal collection against.",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of concurrent requests (matches SpecForge bench_eagle3.py batch_size).",
    )
    p.add_argument(
        "--output", default="signal_data.json",
        help="Path to save raw signal data.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  SIGNAL COLLECTION TEST")
    print("=" * 60)
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Dataset: {args.dataset}, samples: {args.num_samples}")
    print()

    # 1. Load data via SpecForge
    benchmarker = BENCHMARKS.get(args.dataset)(num_samples=args.num_samples)
    questions, labels = benchmarker.load_data()
    sgl_function = benchmarker.create_sgl_function()
    answer_keys = benchmarker.get_answer_keys()

    # 2. Connect to running server (identical to benchmarker/base.py)
    sgl_args = Namespace(
        host=f"http://{args.host}", port=args.port, backend="srt-no-parallel"
    )
    sgl.set_default_backend(select_sglang_backend(sgl_args))

    # 3. Run inference
    print(f"  Sending {len(questions)} requests...")
    tic = time.perf_counter()
    states = sgl_function.run_batch(
        questions,
        temperature=0,
        max_new_tokens=2048,
        num_threads=args.batch_size,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - tic
    print(f"  Done in {elapsed:.1f}s.\n")

    # 4. Extract signal logs and token-level data
    # Single-turn benchmarks (math500, livecodebench, humaneval, gsm8k) return
    # None from get_answer_keys(); base.py handles this with default "answer".
    effective_keys = answer_keys if answer_keys else ["answer"]
    per_turn_logs = extract_signal_logs(states, effective_keys)
    per_turn_tokens = extract_token_level_data(states, effective_keys)
    all_steps = [step for turn_log in per_turn_logs for step in turn_log]

    print("─" * 60)
    print("  SIGNAL STATISTICS")
    print("─" * 60)
    print_signal_stats(all_steps)

    print("─" * 60)
    print("  SAMPLE LOGS (first 3 turns with data)")
    print("─" * 60)
    print_sample_logs(per_turn_logs, n=3)

    # Token-level data summary
    print("─" * 60)
    print("  TOKEN-LEVEL DATA SUMMARY")
    print("─" * 60)
    total_token_steps = sum(
        len(t.get("spec_draft_tokens", [])) for t in per_turn_tokens
    )
    turns_with_tokens = sum(
        1 for t in per_turn_tokens if t.get("spec_draft_tokens")
    )
    print(f"  Turns with token data: {turns_with_tokens}/{len(per_turn_tokens)}")
    print(f"  Total steps with draft tokens: {total_token_steps}")
    if turns_with_tokens > 0:
        # Show a sample
        for i, t in enumerate(per_turn_tokens):
            drafts = t.get("spec_draft_tokens", [])
            accepted = t.get("spec_accepted_tokens_log", [])
            if drafts and len(drafts) > 0:
                print(f"  Turn {i}: {len(drafts)} steps, "
                      f"first step: {len(drafts[0])} draft tokens, "
                      f"{len(accepted[0]) if accepted else 0} accepted")
                break
    print()

    print("─" * 60)
    print("  VERIFICATION")
    print("─" * 60)
    passed = verify_signals(all_steps)

    # 5. Save raw data (signals + token-level combined per turn)
    combined_turns = []
    for i in range(len(per_turn_logs)):
        turn_entry = {
            "signals": per_turn_logs[i],
        }
        if i < len(per_turn_tokens):
            turn_entry.update(per_turn_tokens[i])
        combined_turns.append(turn_entry)

    output = {
        "server": f"{args.host}:{args.port}",
        "num_questions": len(questions),
        "num_turns": len(per_turn_logs),
        "num_steps_total": len(all_steps),
        "elapsed_seconds": elapsed,
        "per_turn_logs": combined_turns,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw data saved to: {args.output}")

    print(f"\n{'=' * 60}")
    print(f"  OVERALL: {'PASS' if passed else 'FAIL'}")
    print(f"{'=' * 60}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
