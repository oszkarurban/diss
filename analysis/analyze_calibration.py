"""
Phase 1 calibration analysis.

Reads the per-step spec_signal_log entries from the SpecForge-produced
JSONLs under results/calibration/ (produced by hpc/calibrate_top1_dynamic.sh)
and reports:

  1. Per-(model, dataset) distribution of top1_prob under the dynamic
     policy.
  2. Per-(model, dataset) bucket counts of the chosen
     (topk, num_steps, ndt) triples — the user's stated analysis goal.
  3. Per-(model, dataset) fraction of steps that hit the small-chain
     zone (``chosen == (1, 2, 3)``).
  4. Suggested refined values for TOP1_UNCONFIDENT / TOP1_DIVISOR based
     on the observed distribution.  The Phase 0 constants live in
     ``sglang/.../speculative/dynamic_spec.py``.

The SpecForge bench_eagle3 output format wraps per-request meta_info in
one entry per benchmark.  Each entry's meta_info['spec_signal_log'] is
a list of dicts as produced by eagle_worker._apply_dynamic_spec_config.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
CAL_DIR = REPO_ROOT / "results/calibration"
SMALL_CHAIN_TRIPLE = (1, 2, 3)  # (topk, num_steps, ndt)


def _iter_signal_logs(jsonl_paths: Iterable[Path]):
    """Yield (model_tag, dataset_tag, signal_log) tuples from each JSONL."""
    for path in jsonl_paths:
        basename = path.name
        tag_match = basename.split("_")
        if len(tag_match) < 4 or tag_match[0] != "calibration":
            continue
        model_tag = tag_match[1]
        dataset_tag = tag_match[2]
        with open(path) as fh:
            blob = json.load(fh)
        if not isinstance(blob, dict):
            continue
        for bench_name, bench_entries in blob.items():
            if bench_name in {"model"}:
                continue
            if not isinstance(bench_entries, list):
                continue
            for entry in bench_entries:
                sigs = (
                    entry.get("metrics", [{}])[0].get("spec_signal_log")
                    or entry.get("spec_signal_log")
                )
                if not sigs:
                    continue
                yield model_tag, dataset_tag, sigs


def flatten_signals(records: Iterable) -> pd.DataFrame:
    rows = []
    for model, dataset, sig_list in records:
        if isinstance(sig_list, list) and sig_list and isinstance(sig_list[0], list):
            for req_steps in sig_list:
                for step in req_steps:
                    rows.append({"model": model, "dataset": dataset, **step})
        else:
            for step in sig_list:
                if isinstance(step, dict):
                    rows.append({"model": model, "dataset": dataset, **step})
    return pd.DataFrame(rows)


def _latest_per_model_dataset(paths: List[Path]) -> List[Path]:
    """Keep only the latest JSONL per (model, dataset) by the run timestamp.

    Filename pattern: ``calibration_<model>_<dataset>_<run_ts>_results_<save_ts>.jsonl``
    where run_ts = YYYYMMDD_HHMMSS.  Older runs are preserved on disk but
    excluded from analysis so historical (pre-patch empty-log) files don't
    drown the latest good data.
    """
    latest: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    for p in paths:
        parts = p.stem.split("_")
        if len(parts) < 6 or parts[0] != "calibration":
            continue
        model, dataset = parts[1], parts[2]
        run_ts = f"{parts[3]}_{parts[4]}"
        key = (model, dataset)
        if key not in latest or run_ts > latest[key][0]:
            latest[key] = (run_ts, p)
    return [p for _, p in latest.values()]


def main() -> None:
    all_paths = sorted(CAL_DIR.glob("*.jsonl"))
    if not all_paths:
        print(f"No calibration JSONLs under {CAL_DIR}.  Run hpc/calibrate_top1_dynamic.sh first.")
        return

    paths = _latest_per_model_dataset(all_paths)
    print(f"Found {len(all_paths)} JSONL(s); using {len(paths)} latest per (model, dataset).")
    for p in sorted(paths):
        print(f"  {p.name}")
    print()

    df = flatten_signals(_iter_signal_logs(paths))
    if df.empty:
        print(f"Loaded {len(paths)} JSONL(s) but no spec_signal_log entries parsed.")
        return

    print(f"Loaded {len(df)} per-step rows from {len(paths)} JSONL(s)\n")

    print("=" * 70)
    print("top1_prob distribution (quantiles)")
    print("=" * 70)
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    print(
        df.groupby(["model", "dataset"], observed=True)["top1_prob"]
        .quantile(quantiles)
        .unstack()
        .round(3)
        .to_string()
    )

    print()
    print("=" * 70)
    print("Chosen (topk, num_steps, ndt) bucket distribution per (model, dataset)")
    print("=" * 70)
    df["bucket"] = list(
        zip(df["chosen_topk"], df["chosen_num_steps"], df["chosen_num_draft_tokens"])
    )
    for (m, d), g in df.groupby(["model", "dataset"], observed=True):
        print(f"\n--- {m} / {d}  (n={len(g)}) ---")
        hist = Counter(g["bucket"])
        for bucket, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / len(g)
            print(f"  (topk={bucket[0]}, num_steps={bucket[1]}, ndt={bucket[2]})  "
                  f"{n:6d}  {pct:5.1f}%")

    print()
    print("=" * 70)
    print("Small-chain zone firing rate  (chosen == (1, 2, 3))")
    print("=" * 70)
    df["is_small"] = df["bucket"] == SMALL_CHAIN_TRIPLE
    print(
        df.groupby(["model", "dataset"], observed=True)["is_small"]
        .mean()
        .mul(100)
        .round(2)
        .to_string()
    )

    print()
    print("=" * 70)
    print("Threshold refinement")
    print("=" * 70)
    for (m, d), g in df.groupby(["model", "dataset"], observed=True):
        small = g[g["is_small"]]
        if small.empty:
            upper = None
        else:
            upper = float(small["top1_prob"].quantile(0.99))
        pct = 100.0 * g["is_small"].mean()
        print(
            f"  {m}/{d}: small-chain fires on {pct:5.1f}% of steps; "
            f"top1_prob 99th pct among small-chain steps = "
            f"{('N/A' if upper is None else f'{upper:.4f}')}"
        )

    upper_bounds: List[float] = []
    for (m, d), g in df.groupby(["model", "dataset"], observed=True):
        small = g[g["is_small"]]
        if not small.empty:
            upper_bounds.append(float(small["top1_prob"].quantile(0.99)))
    if upper_bounds:
        recommended = float(pd.Series(upper_bounds).median())
        current = 0.10
        drift = abs(recommended - current) / max(current, 1e-9)
        flag = "  ⇒  refine" if drift > 0.20 else "  ⇒  keep"
        print(
            f"\n  Suggested TOP1_UNCONFIDENT = {recommended:.4f} "
            f"(current = {current:.4f}, drift = {100*drift:.1f}%){flag}"
        )
    else:
        print("\n  No small-chain firings observed — TOP1_UNCONFIDENT can be "
              "lowered until zone 1 is reachable, or left as a safety net.")


if __name__ == "__main__":
    main()
