#!/usr/bin/env python3
"""
extract_signal_traces.py — flatten Phase A trace files into tidy per-step rows.

Reads all ``results/traces/*.json`` files produced by
``test_signal_collection.py`` (via ``hpc/run_trace_collection.sh``) and
emits one long-form JSONL.gz per (config, model, dataset) combination.

Filename convention expected (from run_trace_collection.sh):
    <config_name>_<model_tag>_<dataset>_<timestamp>.json

E.g. ``static_3_1_4_llama_mtbench_20260418_120115.json`` →
     ``static_3_1_4_llama_mtbench_signals.jsonl.gz``

Each row of the output is one decode step::
    {
        "config": "static_3_1_4",
        "model":  "llama",
        "dataset": "mtbench",
        "turn":   12,          # 0-indexed turn within this file
        "step":   5,           # 0-indexed step within that turn
        "top1_prob": ...,
        "target_top1_prob": ...,
        "rolling_accept_rate": ...,
        "top1_threshold": ...,
        "target_threshold": ...,
        "chosen_topk": ...,
        "chosen_num_steps": ...,
        "chosen_num_draft_tokens": ...,
        "accept_length": ...   # optional, present if server logged it
    }

Usage:
    python3 analysis/extract_signal_traces.py results/traces/
    python3 analysis/extract_signal_traces.py results/traces/ --overwrite
    python3 analysis/extract_signal_traces.py results/traces/ --pattern '*llama*20260418*.json'

Writes to ``results/traces/<config>_<model>_<dataset>_signals.jsonl.gz``
(dropping the timestamp).  If multiple timestamped inputs collide on the
same output name, the newest by mtime wins unless ``--append`` is set.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional, Tuple


# Expected filename pattern: <config>_<model>_<dataset>_<YYYYMMDD>_<HHMMSS>.json
# config may contain underscores (e.g. "static_3_1_4", "v6_dynamic").
# So we anchor on the trailing timestamp + .json suffix and back-parse.
_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")
# Known dataset tags — the segment directly before the timestamp.
_DATASETS = ("mtbench", "math500", "livecodebench", "humaneval", "gsm8k", "gpqa")
# Known model tags — the segment directly before the dataset.
_MODELS = ("llama", "qwen", "deepseek", "deepseek_ld", "deepseek_llamadraft")


def parse_filename(path: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (config, model, dataset, timestamp) or None if not parseable."""
    base = os.path.basename(path)
    m = _TIMESTAMP_RE.search(base)
    if not m:
        return None
    ts = m.group(1)
    stem = base[: m.start()]  # e.g. "static_3_1_4_llama_mtbench"
    # Match dataset tag at the end, then model tag, rest = config.
    for dset in _DATASETS:
        if stem.endswith("_" + dset):
            remainder = stem[: -(len(dset) + 1)]  # strip "_<dset>"
            for mdl in _MODELS:
                if remainder.endswith("_" + mdl):
                    config = remainder[: -(len(mdl) + 1)]
                    return config, mdl, dset, ts
    return None


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [WARN] skip {path}: {e}", file=sys.stderr)
        return None


def flatten_turns(
    data: Dict[str, Any],
    config: str,
    model: str,
    dataset: str,
) -> List[Dict[str, Any]]:
    """One row per (turn, step) with metadata cols merged in."""
    rows: List[Dict[str, Any]] = []
    per_turn = data.get("per_turn_logs", []) or []
    for turn_idx, turn in enumerate(per_turn):
        signals = turn.get("signals", []) if isinstance(turn, dict) else []
        for step_idx, step in enumerate(signals):
            if not isinstance(step, dict):
                continue
            row = {
                "config": config,
                "model": model,
                "dataset": dataset,
                "turn": turn_idx,
                "step": step_idx,
                **step,  # spreads top1_prob, chosen_*, accept_length, ...
            }
            rows.append(row)
    return rows


def group_latest(paths: List[str]) -> Dict[Tuple[str, str, str], str]:
    """For each (config, model, dataset), keep only the path with latest timestamp."""
    groups: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for p in paths:
        parsed = parse_filename(p)
        if parsed is None:
            continue
        config, model, dset, ts = parsed
        key = (config, model, dset)
        prev = groups.get(key)
        if prev is None or ts > prev[1]:
            groups[key] = (p, ts)
    return {k: v[0] for k, v in groups.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "trace_dir",
        help="Directory containing Phase A *.json trace files.",
    )
    ap.add_argument(
        "--pattern", default="*.json",
        help="Glob pattern within trace_dir (default: *.json).",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing *_signals.jsonl.gz files (default: skip).",
    )
    ap.add_argument(
        "--append", action="store_true",
        help="If multiple timestamped inputs exist for the same "
             "(config, model, dataset), concat rows instead of picking latest.",
    )
    args = ap.parse_args()

    trace_dir = os.path.abspath(args.trace_dir)
    if not os.path.isdir(trace_dir):
        print(f"[ERROR] Not a directory: {trace_dir}", file=sys.stderr)
        return 1

    inputs = sorted(glob(os.path.join(trace_dir, args.pattern)))
    if not inputs:
        print(f"[ERROR] No files matching {args.pattern} in {trace_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(inputs)} candidate json files in {trace_dir}")

    if args.append:
        # Group: one output file per (config, model, dataset); merge all timestamps.
        grouped: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        for p in inputs:
            parsed = parse_filename(p)
            if parsed:
                grouped[(parsed[0], parsed[1], parsed[2])].append(p)
    else:
        # Pick latest timestamp per group.
        latest = group_latest(inputs)
        grouped = {k: [v] for k, v in latest.items()}

    if not grouped:
        print("[ERROR] No files matched the expected "
              "<config>_<model>_<dataset>_<timestamp>.json pattern.", file=sys.stderr)
        return 1

    print(f"Identified {len(grouped)} (config, model, dataset) groups.")

    total_rows = 0
    for (config, model, dset), paths in sorted(grouped.items()):
        out_name = f"{config}_{model}_{dset}_signals.jsonl.gz"
        out_path = os.path.join(trace_dir, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [SKIP] {out_name} exists (use --overwrite to replace)")
            continue

        rows: List[Dict[str, Any]] = []
        for p in paths:
            data = load_json(p)
            if data is None:
                continue
            rows.extend(flatten_turns(data, config, model, dset))

        if not rows:
            print(f"  [WARN] {out_name}: 0 rows (signal log empty in source)")
            continue

        with gzip.open(out_path, "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        total_rows += len(rows)
        print(f"  [OK]   {out_name:<60s} {len(rows):>6d} steps   "
              f"(from {len(paths)} input{'s' if len(paths) > 1 else ''})")

    print(f"\nTotal: {total_rows} rows across {len(grouped)} output files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
