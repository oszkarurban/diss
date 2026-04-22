#!/usr/bin/env python3
"""
collect_traces.py — benchmark-agnostic spec_signal_log extractor.

Runs one benchmark against a running sglang server (launched with
--dynamic-spec-full-logging, with or without dynamic spec enabled),
extracts per-step spec_signal_log + per-step token-level data from
each request's meta_info, and writes a gzipped JSONL to disk.

Usage:
    python hpc/collect_traces.py \\
        --benchmark mtbench:20 \\
        --host 127.0.0.1 --port 30000 \\
        --output results/traces/llama_static_3_1_4_mtbench_signals.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from typing import Any, Dict, List

# Path setup (must precede sglang / benchmarker imports).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    os.path.join(_REPO_ROOT, "sglang", "python"),
    os.path.join(_REPO_ROOT, "SpecForge", "benchmarks"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from argparse import Namespace

from benchmarker import BENCHMARKS
from sglang import set_default_backend
from sglang.test.test_utils import select_sglang_backend

V6_SIGNAL_KEYS = ["top1_prob", "target_top1_prob", "rolling_accept_rate"]
V6_THRESHOLD_KEYS = ["top1_threshold", "target_threshold"]
CONFIG_KEYS = ["chosen_topk", "chosen_num_steps", "chosen_num_draft_tokens"]
TOKEN_KEYS = [
    "spec_draft_tokens",
    "spec_accepted_tokens_log",
    "spec_rejected_tokens_log",
    "spec_accept_index_log",
    "spec_topk",
    "spec_num_steps",
    "spec_draft_token_num",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   help="e.g. 'mtbench:20', 'math500:20', 'livecodebench:20'")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override benchmark's default")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def split_bench_spec(spec: str):
    parts = spec.split(":")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], int(parts[1]), None
    name, n, subset = parts
    return name, int(n), subset.split(",")


def extract_per_request(
    states: List[Any], answer_keys: List[str]
) -> List[Dict[str, Any]]:
    rows = []
    for i, state in enumerate(states):
        for key in answer_keys:
            meta = state.get_meta_info(key)
            if meta is None:
                continue
            rows.append({
                "request_idx": i,
                "answer_key": key,
                "completion_tokens": meta.get("completion_tokens"),
                "spec_verify_ct": meta.get("spec_verify_ct"),
                "spec_signal_log": meta.get("spec_signal_log", []),
                **{k: meta.get(k, []) for k in TOKEN_KEYS},
            })
    return rows


def main() -> None:
    args = parse_args()
    bench_name, num_samples, subset = split_bench_spec(args.benchmark)
    if bench_name not in BENCHMARKS.benchmarks:
        raise SystemExit(
            f"Unknown benchmark {bench_name!r}. "
            f"Available: {list(BENCHMARKS.benchmarks.keys())}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # Initialize backend identically to benchmarker.run()
    sglang_args = Namespace(
        host=f"http://{args.host}",
        port=args.port,
        backend="srt-no-parallel",
    )
    set_default_backend(select_sglang_backend(sglang_args))

    # Instantiate the benchmarker
    cls = BENCHMARKS.get(bench_name)
    if subset is None:
        bench = cls(num_samples=num_samples)
    else:
        bench = cls(num_samples=num_samples, subset=subset)

    questions, labels = bench.load_data()
    if not questions:
        raise SystemExit(f"Benchmark {bench_name} returned no questions")

    sgl_fn = bench.create_sgl_function()
    answer_keys = bench.get_answer_keys() or ["answer"]
    max_new = args.max_new_tokens or bench.get_max_new_tokens()

    print(f"[collect_traces] benchmark={bench_name} n={len(questions)} "
          f"bs={args.batch_size} max_new={max_new}")

    tic = time.perf_counter()
    states = sgl_fn.run_batch(
        questions,
        temperature=0,
        max_new_tokens=max_new,
        num_threads=args.batch_size,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - tic
    print(f"[collect_traces] inference completed in {elapsed:.1f}s")

    rows = extract_per_request(states, answer_keys)
    total_steps = sum(len(r["spec_signal_log"]) for r in rows)
    rows_with_signals = sum(1 for r in rows if r["spec_signal_log"])
    print(f"[collect_traces] extracted {len(rows)} request rows, "
          f"{rows_with_signals} with spec_signal_log (total steps: {total_steps})")

    # Header line with run-level metadata, then one line per request
    header = {
        "__meta__": True,
        "benchmark": bench_name,
        "num_samples": len(questions),
        "batch_size": args.batch_size,
        "max_new_tokens": max_new,
        "elapsed_s": elapsed,
        "generated_at": time.time(),
    }
    with gzip.open(args.output, "wt") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[collect_traces] wrote {args.output}")


if __name__ == "__main__":
    main()
