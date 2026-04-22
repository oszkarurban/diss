#!/usr/bin/env python3
"""
phase_b_build_master.py — Build the Phase B master DataFrame.

Joins the flattened signal JSONL.gz files (one row per decode step) with
per-step token-level data extracted from the raw trace JSON files.

Output: ``results/traces/master.parquet`` — one row per (config, model,
dataset, turn, step) with columns:

    config                    str    e.g. "static_3_1_4"
    model                     str    "llama" | "qwen"
    dataset                   str    "mtbench" | "math500" | "livecodebench"
    turn                      int    per-file turn index
    step                      int    per-turn step index
    top1_prob                 float  draft softmax top-1
    target_top1_prob          float  target softmax top-1
    rolling_accept_rate       float  per-request EMA, α=0.3
    top1_threshold            float  logged running percentile (0 for static)
    target_threshold          float  logged running percentile (0 for static)
    chosen_topk               int    the config that was active this step
    chosen_num_steps          int
    chosen_num_draft_tokens   int
    accept_length             int    tokens accepted from draft this step
    draft_tree_size           int    len(spec_draft_tokens[step]) — actual budget
    first_reject_pos          int    max(accept_index) + 1, or 0 if all rejected
    DOG                       float  top1_prob × rolling_accept_rate  (V3 scalar)
    t                         float  0.7·top1 + 0.3·target_top1        (V6 scalar)
    yield_per_compute         float  accept_length / draft_tree_size

Usage:
    python3 analysis/phase_b_build_master.py
    python3 analysis/phase_b_build_master.py --overwrite
    python3 analysis/phase_b_build_master.py --trace-dir results/traces/
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
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd


_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.json$")
_DATASETS = ("mtbench", "math500", "livecodebench", "humaneval", "gsm8k", "gpqa")
_MODELS = ("llama", "qwen", "deepseek", "deepseek_ld", "deepseek_llamadraft")


def parse_raw_filename(path: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse results/traces/<config>_<model>_<dataset>_<YYYYMMDD>_<HHMMSS>.json."""
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


def load_signals(trace_dir: str) -> pd.DataFrame:
    """Load every *_signals.jsonl.gz into one long DataFrame."""
    paths = sorted(glob(os.path.join(trace_dir, "*_signals.jsonl.gz")))
    if not paths:
        raise FileNotFoundError(f"No *_signals.jsonl.gz in {trace_dir}")

    rows: List[Dict] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for line in f:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Strict dtypes so downstream joins / filters don't silently cast
    int_cols = ["turn", "step", "chosen_topk", "chosen_num_steps",
                "chosen_num_draft_tokens"]
    for c in int_cols:
        df[c] = df[c].astype("int32")
    float_cols = ["top1_prob", "target_top1_prob", "rolling_accept_rate",
                  "top1_threshold", "target_threshold"]
    for c in float_cols:
        df[c] = df[c].astype("float32")
    for c in ("config", "model", "dataset"):
        df[c] = df[c].astype("category")

    print(f"  Loaded signals: {len(df):,} rows from {len(paths)} files")
    return df


def iter_token_rows(trace_dir: str) -> Iterator[Dict]:
    """Stream one dict per (config,model,dataset,turn,step) with token-level fields.

    Picks the latest timestamp per (config, model, dataset) when multiple
    raw JSONs exist for the same cell.
    """
    latest: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    for p in glob(os.path.join(trace_dir, "*.json")):
        # Exclude any leftover aggregate / analysis files
        if os.path.basename(p).startswith(("smoke_", "metrics_")):
            continue
        parsed = parse_raw_filename(p)
        if parsed is None:
            continue
        config, model, dset, ts = parsed
        key = (config, model, dset)
        prev = latest.get(key)
        if prev is None or ts > prev[1]:
            latest[key] = (p, ts)

    n_cells = 0
    for (config, model, dset), (path, _ts) in sorted(latest.items()):
        n_cells += 1
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] skip {path}: {e}", file=sys.stderr)
            continue

        per_turn = data.get("per_turn_logs", []) or []
        for turn_idx, turn in enumerate(per_turn):
            if not isinstance(turn, dict):
                continue
            accepted_log = turn.get("spec_accepted_tokens_log", []) or []
            draft_log = turn.get("spec_draft_tokens", []) or []
            accept_idx_log = turn.get("spec_accept_index_log", []) or []

            n_steps = max(len(accepted_log), len(draft_log), len(accept_idx_log))
            for step_idx in range(n_steps):
                accepted = (accepted_log[step_idx]
                            if step_idx < len(accepted_log) else [])
                draft = (draft_log[step_idx]
                         if step_idx < len(draft_log) else [])
                accept_idx = (accept_idx_log[step_idx]
                              if step_idx < len(accept_idx_log) else [])

                yield {
                    "config": config,
                    "model": model,
                    "dataset": dset,
                    "turn": turn_idx,
                    "step": step_idx,
                    "accept_length": len(accepted) if accepted else 0,
                    "draft_tree_size": len(draft) if draft else 0,
                    # max(accept_idx) + 1 when non-empty; 0 if all drafts rejected.
                    # Works for both chains (positions = 0..num_steps-1) and
                    # trees (positions = indices into the flat draft array).
                    "first_reject_pos": (max(accept_idx) + 1
                                         if accept_idx else 0),
                }
    print(f"  Scanned {n_cells} raw trace files (latest per cell)")


def load_tokens(trace_dir: str) -> pd.DataFrame:
    rows = list(iter_token_rows(trace_dir))
    if not rows:
        raise FileNotFoundError(
            f"No token data extracted from {trace_dir}/*.json"
        )
    df = pd.DataFrame(rows)
    for c in ("config", "model", "dataset"):
        df[c] = df[c].astype("category")
    for c in ("turn", "step", "accept_length", "draft_tree_size",
              "first_reject_pos"):
        df[c] = df[c].astype("int32")
    print(f"  Loaded tokens:  {len(df):,} rows")
    return df


def build_master(trace_dir: str) -> pd.DataFrame:
    signals = load_signals(trace_dir)
    tokens = load_tokens(trace_dir)

    # Some dynamic log entries include "accept_length" (appended in verify
    # pipeline — see eagle_info.py:523); static rows collected by
    # _log_signals_static don't include it. Tokens df's accept_length is
    # len(spec_accepted_tokens_log[step]) — the SAME quantity, measured
    # directly from the token log, and always populated.
    #
    # Cross-check: where signals.accept_length exists, verify it matches
    # tokens.accept_length to within ±0 (should be exactly equal — they're
    # derived from the same underlying accept_index array). Then drop the
    # signals column so merge keeps the complete tokens version.
    if "accept_length" in signals.columns:
        # The signals-side accept_length is appended by eagle_info.verify()
        # AFTER draft() has already added the current step's log entry. On
        # dynamic runs this creates an off-by-one: signals.accept_length
        # on step N reflects step (N-1)'s verify outcome, because verify for
        # the current step hasn't run yet when we log the per-step signals.
        # The tokens-side accept_length (len of spec_accepted_tokens_log[step])
        # is the authoritative per-step outcome. We use that; signals column
        # is dropped. Cross-check is informational only — expect disagreement
        # until the off-by-one is fixed upstream.
        key_cols_check = ["config", "model", "dataset", "turn", "step"]
        cmp = signals[key_cols_check + ["accept_length"]].rename(
            columns={"accept_length": "accept_length_signals"}
        ).merge(
            tokens[key_cols_check + ["accept_length"]],
            on=key_cols_check, how="inner",
        )
        both_present = cmp["accept_length_signals"].notna()
        if both_present.any():
            agree = (
                cmp.loc[both_present, "accept_length_signals"]
                == cmp.loc[both_present, "accept_length"]
            )
            print(f"  accept_length cross-check (informational): "
                  f"{int(agree.sum())}/{int(both_present.sum())} rows agree. "
                  f"Expected low agreement due to verify-side append "
                  f"off-by-one. We use tokens.accept_length (definitive).")
        signals = signals.drop(columns=["accept_length"])

    # draft_tree_size and first_reject_pos only ever come from the tokens df;
    # defensively drop any legacy pre-existing columns in signals.
    for col in ("draft_tree_size", "first_reject_pos"):
        if col in signals.columns:
            signals = signals.drop(columns=[col])

    key_cols = ["config", "model", "dataset", "turn", "step"]
    master = signals.merge(tokens, on=key_cols, how="inner", validate="one_to_one")

    # Quick sanity: join should preserve row count of signals (every signal row
    # has token data). If not, something is off.
    if len(master) != len(signals):
        lost = len(signals) - len(master)
        print(f"  [WARN] join lost {lost:,} signal rows "
              f"({100 * lost / len(signals):.1f}%). "
              "Likely timestamp mismatch between signals and raw JSON.",
              file=sys.stderr)

    # Derived signals
    master["DOG"] = (
        master["top1_prob"] * master["rolling_accept_rate"]
    ).astype("float32")
    master["t"] = (
        0.7 * master["top1_prob"] + 0.3 * master["target_top1_prob"]
    ).astype("float32")
    # Per-step acceptance rate — derived, not logged.
    # Matches the server's own formula (eagle_info.py:479): the raw step_rate
    # that gets fed into the rolling_accept_rate EMA.  accept_length is
    # capped at (ndt - 1) by EAGLE3 (the bonus token isn't counted in AL),
    # so denom = max(ndt - 1, 1) keeps this in [0, 1].
    master["accept_rate"] = (
        master["accept_length"]
        / (master["chosen_num_draft_tokens"].clip(lower=2) - 1)
    ).clip(0, 1).astype("float32")
    # Yield per unit of draft compute — used by Plot 4 (buy zone).
    master["yield_per_compute"] = (
        master["accept_length"] / master["draft_tree_size"].clip(lower=1)
    ).astype("float32")

    # Final column order
    master = master[[
        "config", "model", "dataset", "turn", "step",
        # Raw signals logged by the server per decode step
        "top1_prob", "target_top1_prob", "rolling_accept_rate",
        "top1_threshold", "target_threshold",
        # The config active on this step (varies for dynamic, constant for static)
        "chosen_topk", "chosen_num_steps", "chosen_num_draft_tokens",
        # Outcome: how many draft tokens the target accepted
        "accept_length", "accept_rate",
        "draft_tree_size", "first_reject_pos",
        # Derived decision signals (what V3 and V6 consume)
        "DOG", "t",
        "yield_per_compute",
    ]]

    return master


def sanity_checks(master: pd.DataFrame, trace_dir: str) -> None:
    print("\n=== Sanity checks ===")

    # 1. Row counts per cell
    by_cell = master.groupby(
        ["config", "model", "dataset"], observed=True
    ).size().reset_index(name="rows")
    print(f"  Cells covered: {len(by_cell)}")

    # 2. AL per cell — compare with metrics_final.csv if present
    metrics_csv = os.path.join(trace_dir, "metrics_final.csv")
    if os.path.exists(metrics_csv):
        metrics = pd.read_csv(metrics_csv)
        computed = master.groupby(
            ["config", "model", "dataset"], observed=True
        )["accept_length"].mean().reset_index(name="AL_computed")
        check = metrics.merge(
            computed, on=["config", "model", "dataset"]
        )
        check["delta"] = (check["accept_length"] - check["AL_computed"]).abs()
        worst = check.loc[check["delta"].idxmax()]
        print(f"  AL check vs metrics_final.csv: "
              f"worst |delta| = {worst['delta']:.4f} "
              f"at {worst['config']}/{worst['model']}/{worst['dataset']}")
        if worst["delta"] > 0.01:
            print(f"  [WARN] AL mismatch > 0.01 tokens. Investigate.",
                  file=sys.stderr)

    # 3. Coverage matrix
    print("\n  Coverage matrix (rows per cell):")
    pivot = by_cell.pivot_table(
        index=["config", "model"], columns="dataset",
        values="rows", aggfunc="sum", fill_value=0,
    )
    print(pivot.to_string())

    # 4. Signal sanity
    print(f"\n  Signal ranges:")
    for col in ("top1_prob", "target_top1_prob", "rolling_accept_rate",
                "DOG", "t"):
        s = master[col]
        print(f"    {col:25s}  min={s.min():.3f}  "
              f"max={s.max():.3f}  mean={s.mean():.3f}  std={s.std():.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="results/traces/",
                    help="Directory with *_signals.jsonl.gz + *.json files")
    ap.add_argument("--output", default=None,
                    help="Parquet path (default: {trace-dir}/master.parquet)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing parquet")
    args = ap.parse_args()

    default_out = os.path.join(args.trace_dir, "master.parquet")
    out = args.output or default_out
    if os.path.exists(out) and not args.overwrite:
        print(f"[SKIP] {out} exists (use --overwrite)")
        return 0

    print(f"Building master df from {args.trace_dir}")
    master = build_master(args.trace_dir)
    print(f"\nMaster df: {len(master):,} rows × {len(master.columns)} cols")

    sanity_checks(master, args.trace_dir)

    # Try parquet first (compact + fast); fall back to pickle if neither
    # pyarrow nor fastparquet is installed.
    def _write(path: str) -> None:
        master.to_parquet(path, compression="snappy", index=False)

    try:
        _write(out)
        print(f"\nWrote {out}  ({os.path.getsize(out)/1e6:.1f} MB) [parquet]")
    except ImportError:
        fallback = os.path.splitext(out)[0] + ".pkl"
        master.to_pickle(fallback, compression="gzip")
        print(f"\n[INFO] pyarrow/fastparquet not installed — wrote pickle instead")
        print(f"Wrote {fallback}  ({os.path.getsize(fallback)/1e6:.1f} MB) [pickle.gz]")
        print(f"Note: notebook loader tries parquet first, falls back to pickle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
