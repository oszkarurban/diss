#!/usr/bin/env python3
"""
compute_trace_metrics.py — derive throughput + accept_length from Phase A traces.

`test_signal_collection.py` doesn't emit the SpecForge bench_eagle3.py
summary format (output_throughput / accept_length). This helper computes
those numbers from the raw per-step data present in every trace:

  accept_length  = mean(len(spec_accepted_tokens_log[step])) across all steps
  output_tokens  = sum(len(spec_accepted_tokens_log[step]))  across all steps
  throughput     = output_tokens / elapsed_seconds          (tok/s, aggregate)

For static configs these should match the CLAUDE.md/static_eagle3_baselines
numbers (within ~1% of bench_eagle3.py's MT-Bench:80 runs, since our n=20
is a random subset and temperature=0 is deterministic per prompt).

Usage:
    python3 analysis/compute_trace_metrics.py results/traces/
    python3 analysis/compute_trace_metrics.py results/traces/ --csv metrics.csv
    python3 analysis/compute_trace_metrics.py results/traces/ --markdown

Outputs a table per (config, model, dataset) cell. Picks the latest
timestamp per cell (same heuristic as extract_signal_traces.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Tuple


_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")
_DATASETS = ("mtbench", "math500", "livecodebench", "humaneval", "gsm8k", "gpqa")
_MODELS = ("llama", "qwen", "deepseek", "deepseek_ld", "deepseek_llamadraft")


def parse_filename(path: str) -> Optional[Tuple[str, str, str, str]]:
    base = os.path.basename(path)
    m = _TIMESTAMP_RE.search(base)
    if not m:
        return None
    ts = m.group(1)
    stem = base[: m.start()]
    for dset in _DATASETS:
        if stem.endswith("_" + dset):
            rem = stem[: -(len(dset) + 1)]
            for mdl in _MODELS:
                if rem.endswith("_" + mdl):
                    return rem[: -(len(mdl) + 1)], mdl, dset, ts
    return None


def metrics_for(trace: dict) -> Dict[str, float]:
    """Aggregate throughput (tok/s) + accept_length from one trace JSON."""
    per_turn = trace.get("per_turn_logs", [])
    elapsed = float(trace.get("elapsed_seconds", 0.0)) or 1e-9

    total_accepted = 0
    step_als: List[int] = []
    step_draft_counts: List[int] = []

    for turn in per_turn:
        accepted_log = turn.get("spec_accepted_tokens_log", []) or []
        draft_log = turn.get("spec_draft_tokens", []) or []
        for step_accepted in accepted_log:
            al = len(step_accepted) if step_accepted is not None else 0
            step_als.append(al)
            total_accepted += al
        for step_draft in draft_log:
            step_draft_counts.append(
                len(step_draft) if step_draft is not None else 0
            )

    num_steps = len(step_als)
    num_turns = len(per_turn)
    num_turns_nonempty = sum(1 for t in per_turn if t.get("spec_draft_tokens"))

    if num_steps == 0:
        return {
            "throughput": 0.0,
            "accept_length": 0.0,
            "output_tokens": 0,
            "num_steps": 0,
            "num_turns": num_turns,
            "num_turns_nonempty": num_turns_nonempty,
            "elapsed_s": elapsed,
        }

    return {
        "throughput": total_accepted / elapsed,
        "accept_length": statistics.mean(step_als),
        "accept_length_std": statistics.pstdev(step_als) if num_steps > 1 else 0.0,
        "output_tokens": total_accepted,
        "num_steps": num_steps,
        "num_turns": num_turns,
        "num_turns_nonempty": num_turns_nonempty,
        "mean_draft_per_step": (
            statistics.mean(step_draft_counts) if step_draft_counts else 0.0
        ),
        "elapsed_s": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace_dir", help="Directory with Phase A *.json trace files.")
    ap.add_argument("--pattern", default="*.json", help="Glob inside trace_dir.")
    ap.add_argument("--csv", default=None, help="Optional output CSV path.")
    ap.add_argument(
        "--markdown", action="store_true",
        help="Print results as a pivoted markdown table (dataset columns).",
    )
    args = ap.parse_args()

    paths = sorted(glob(os.path.join(args.trace_dir, args.pattern)))
    if not paths:
        print(f"[ERROR] No files matching {args.pattern} in {args.trace_dir}", file=sys.stderr)
        return 1

    # Keep newest timestamp per (config, model, dataset)
    latest: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for p in paths:
        parsed = parse_filename(p)
        if parsed is None:
            continue
        config, model, dset, ts = parsed
        key = (config, model, dset)
        prev = latest.get(key)
        if prev is None or ts > prev[1]:
            latest[key] = (p, ts)

    rows = []
    for (config, model, dset), (p, ts) in sorted(latest.items()):
        try:
            with open(p) as f:
                trace = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        m = metrics_for(trace)
        rows.append({
            "config": config, "model": model, "dataset": dset,
            "timestamp": ts, "file": os.path.basename(p),
            **m,
        })

    if not rows:
        print("[ERROR] No parseable trace files found.", file=sys.stderr)
        return 1

    # Print long-form table
    print(f"\nMetrics derived from {len(rows)} trace files:\n")
    hdr = f"{'config':<16} {'model':<6} {'dataset':<15} " \
          f"{'tput':>8} {'AL':>6} {'AL_std':>7} {'n_steps':>8} {'n_turns':>8} {'elap':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['config']:<16} {r['model']:<6} {r['dataset']:<15} "
            f"{r['throughput']:>8.2f} {r['accept_length']:>6.2f} "
            f"{r.get('accept_length_std', 0):>7.2f} "
            f"{r['num_steps']:>8d} {r['num_turns']:>8d} {r['elapsed_s']:>6.1f}"
        )

    # Pivoted markdown table
    if args.markdown:
        print("\n### Pivoted: throughput (tok/s) — rows=config, cols=dataset")
        models = sorted({r["model"] for r in rows})
        configs = sorted({r["config"] for r in rows})
        datasets = sorted({r["dataset"] for r in rows})
        for mdl in models:
            print(f"\n**Model: {mdl}**\n")
            print("| config | " + " | ".join(datasets) + " |")
            print("|" + "---|" * (len(datasets) + 1))
            for cfg in configs:
                cells = []
                for ds in datasets:
                    match = next(
                        (r for r in rows
                         if r["config"] == cfg and r["model"] == mdl and r["dataset"] == ds),
                        None,
                    )
                    if match is None:
                        cells.append("—")
                    else:
                        cells.append(
                            f"{match['throughput']:.1f} / {match['accept_length']:.2f}"
                        )
                print(f"| {cfg} | " + " | ".join(cells) + " |")

    # Optional CSV
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV written to: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
