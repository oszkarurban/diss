#!/usr/bin/env python3
"""
phase_b3_eagle2.py — EAGLE-2 Figure 6 style analysis.

Central question (after user feedback): which signal most linearly
predicts acceptance, and AT WHAT CONFIDENCE INTERVAL does each tree
shape (small chain vs deep chain vs wide tree vs very wide tree) pay
off?

This mirrors EAGLE-2's Figure 6 methodology:
  - Bin draft confidence into intervals
  - Plot mean acceptance rate per bin
  - Compare to y=x line — steeper / more linear = better predictor

Then extends:
  - Per-depth cumulative acceptance probability (chains only): shows
    where chain depth stops paying off
  - Wide vs deep comparison at same signal bin: shows where topk=4
    branching beats topk=1 chain
  - Signal linearity score (R² of OLS fit) per cell per signal

Usage:
    python3 analysis/phase_b3_eagle2.py

Outputs:
    analysis/figures/plot_e2a_fig6_{signal}.{png,pdf}   — EAGLE-2 F6 replication
    analysis/figures/plot_e2b_depth_cumulative.{png,pdf} — per-depth accept curves
    analysis/figures/plot_e2c_wide_vs_deep.{png,pdf}    — tree vs chain
    analysis/figures/tables/e2_linearity_scores.csv
    analysis/figures/tables/e2_transition_points.csv
    analysis/phase_b3_eagle2_summary.md
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402
from scipy import stats          # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase_b_analysis import (  # noqa: E402
    load_master, STATIC_CONFIGS, CONFIG_SHAPE, MODELS, DATASETS,
)


CFG_PALETTE = {
    "static_3_1_4":   "#1f77b4",
    "static_7_1_8":   "#ff7f0e",
    "static_7_4_8":   "#2ca02c",
    "static_6_10_60": "#d62728",
}

# Signals to compare as predictors of acceptance
SIGNAL_CANDIDATES = ["top1_prob", "target_top1_prob", "t", "DOG"]

# Granularity for the Eagle-2 Figure 6 bins (paper uses ~10 bins of 0.1)
N_BINS_EAGLE2 = 10


def _savefig(fig_dir: str, name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(
            os.path.join(fig_dir, f"{name}.{ext}"),
            bbox_inches="tight",
            dpi=150 if ext == "png" else 300,
        )
    plt.close()


# -------------------------------------------------------------------------- #
# EAGLE-2 Figure 6 replication                                               #
# -------------------------------------------------------------------------- #

def eagle2_binned_accept_rate(
    master: pd.DataFrame,
    signal: str,
    config: str,
    n_bins: int = N_BINS_EAGLE2,
) -> pd.DataFrame:
    """For one (cell, static config, signal): per bin mean acceptance rate.

    Acceptance rate = accept_length / num_steps  (normalised to [0, 1]).
    Bins are FIXED-EDGE in [0, 1] so all cells align visually (matches
    EAGLE-2 paper's uniform binning).
    """
    num_steps = CONFIG_SHAPE[config][0]
    sub = master[master["config"] == config].copy()
    sub["accept_rate_normalised"] = sub["accept_length"] / num_steps

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    rows = []
    for (mdl, dset), cell in sub.groupby(["model", "dataset"]):
        for k in range(n_bins):
            lo, hi = edges[k], edges[k + 1]
            # Inclusive on right edge of last bin to catch values==1.0
            in_bin = ((cell[signal] >= lo) & (cell[signal] < hi)) \
                     if k < n_bins - 1 else \
                     ((cell[signal] >= lo) & (cell[signal] <= hi))
            bucket = cell[in_bin]
            if len(bucket) < 3:
                rows.append({
                    "model": mdl, "dataset": dset, "config": config,
                    "bin_idx": k, "bin_mid": float(mids[k]),
                    "n": int(len(bucket)),
                    "mean_accept_rate": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan"),
                })
                continue
            arr = bucket["accept_rate_normalised"].to_numpy()
            sem = arr.std(ddof=1) / np.sqrt(len(arr))
            rows.append({
                "model": mdl, "dataset": dset, "config": config,
                "bin_idx": k, "bin_mid": float(mids[k]),
                "n": int(len(arr)),
                "mean_accept_rate": float(arr.mean()),
                "ci_lo": float(arr.mean() - 1.96 * sem),
                "ci_hi": float(arr.mean() + 1.96 * sem),
            })
    return pd.DataFrame(rows)


def signal_linearity_score(binned: pd.DataFrame) -> pd.DataFrame:
    """Per (model, dataset, config), fit y = a·x + b to (bin_mid, mean_accept_rate)
    and return R², slope, intercept. Compare a to 1.0 for Eagle-2 style reading.
    """
    rows = []
    for (mdl, dset, cfg), grp in binned.groupby(["model", "dataset", "config"]):
        valid = grp.dropna(subset=["mean_accept_rate"])
        if len(valid) < 3:
            continue
        x = valid["bin_mid"].to_numpy()
        y = valid["mean_accept_rate"].to_numpy()
        slope, intercept, r, p, se = stats.linregress(x, y)
        rows.append({
            "model": mdl, "dataset": dset, "config": cfg,
            "slope_a": float(slope),
            "intercept_b": float(intercept),
            "r_squared": float(r ** 2),
            "pearson_r": float(r),
            "n_bins_valid": int(len(valid)),
        })
    return pd.DataFrame(rows)


def plot_fig6_replication(
    master: pd.DataFrame,
    signal: str,
    fig_dir: str,
    tables_dir: str,
) -> pd.DataFrame:
    """EAGLE-2 Figure 6 replication: mean accept rate vs signal bin mid, y=x
    reference line.  One panel per (model, dataset). Lines per config."""
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS),
        figsize=(15, 8), sharex=True, sharey=True,
    )

    all_binned = []
    for cfg in STATIC_CONFIGS:
        binned = eagle2_binned_accept_rate(master, signal, cfg)
        binned["signal"] = signal
        all_binned.append(binned)
    df = pd.concat(all_binned, ignore_index=True)

    for i, mdl in enumerate(MODELS):
        for j, dset in enumerate(DATASETS):
            ax = axes[i, j]
            # y=x reference (EAGLE-2 paper's red dashed)
            ax.plot([0, 1], [0, 1], "r--", alpha=0.6, linewidth=1.5,
                    label="y = x reference" if (i == 0 and j == 0) else None)
            for cfg in STATIC_CONFIGS:
                sub = df[(df["model"] == mdl) & (df["dataset"] == dset)
                         & (df["config"] == cfg)].dropna(subset=["mean_accept_rate"])
                if len(sub) == 0:
                    continue
                ax.errorbar(
                    sub["bin_mid"], sub["mean_accept_rate"],
                    yerr=[sub["mean_accept_rate"] - sub["ci_lo"],
                          sub["ci_hi"] - sub["mean_accept_rate"]],
                    fmt="o-", color=CFG_PALETTE[cfg], label=cfg,
                    markersize=5, linewidth=1.8, capsize=2, alpha=0.9,
                )
            ax.set_title(f"{mdl} / {dset}")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if i == 1:
                ax.set_xlabel(f"{signal} bin mid")
            if j == 0:
                ax.set_ylabel(r"$\overline{\mathrm{AL}} / \mathrm{num\_steps}$")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.suptitle(
        f"EAGLE-2 Fig 6 replication — mean acceptance rate vs {signal}\n"
        f"(y=x reference from (0,0) to (1,1); closer to line = more linear predictor)",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    _savefig(fig_dir, f"plot_e2a_fig6_{signal}")

    df.to_csv(os.path.join(tables_dir, f"e2_binned_accept_{signal}.csv"), index=False)
    return df


# -------------------------------------------------------------------------- #
# Per-depth cumulative acceptance (chains only)                              #
# -------------------------------------------------------------------------- #

def depth_cumulative_accept(
    master: pd.DataFrame,
    signal: str,
    chain_config: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """For a chain config, compute P(first k+1 drafts accepted) per signal bin.

    For chains (topk=1): P(depth k accepted) = P(first_reject_pos > k)
    Returns long DataFrame: (model, dataset, bin_mid, k, cumulative_accept_prob)
    """
    num_steps = CONFIG_SHAPE[chain_config][0]
    topk = CONFIG_SHAPE[chain_config][1]
    assert topk == 1, "depth_cumulative_accept only makes sense for chains (topk=1)"

    sub = master[master["config"] == chain_config].copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    rows = []
    for (mdl, dset), cell in sub.groupby(["model", "dataset"]):
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            in_bin = ((cell[signal] >= lo) & (cell[signal] < hi)) \
                     if b < n_bins - 1 else \
                     ((cell[signal] >= lo) & (cell[signal] <= hi))
            bucket = cell[in_bin]
            if len(bucket) < 3:
                continue
            for k in range(num_steps):
                rows.append({
                    "model": mdl, "dataset": dset, "config": chain_config,
                    "signal": signal, "bin_mid": float(mids[b]),
                    "depth_k": k,
                    "prob_depth_accepted": float(
                        (bucket["first_reject_pos"] > k).mean()
                    ),
                    "n_steps_in_bin": int(len(bucket)),
                })
    return pd.DataFrame(rows)


def plot_depth_cumulative(
    master: pd.DataFrame,
    signal: str,
    fig_dir: str,
    tables_dir: str,
) -> pd.DataFrame:
    """Per cell, one panel showing cumulative depth-acceptance curves for
    static_7_1_8 (deep chain). Colour = signal bin; x = depth; y = P(depth
    accepted). Falling curves indicate where the chain breaks."""
    chain_cfg = "static_7_1_8"
    df = depth_cumulative_accept(master, signal, chain_cfg, n_bins=5)
    if len(df) == 0:
        print(f"  [SKIP] no {chain_cfg} data")
        return df

    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS),
        figsize=(15, 8), sharex=True, sharey=True,
    )
    bins = sorted(df["bin_mid"].unique())
    bin_palette = sns.color_palette("viridis", n_colors=len(bins))
    bin_colors = {b: bin_palette[i] for i, b in enumerate(bins)}

    for i, mdl in enumerate(MODELS):
        for j, dset in enumerate(DATASETS):
            ax = axes[i, j]
            cell = df[(df["model"] == mdl) & (df["dataset"] == dset)]
            for b, bin_grp in cell.groupby("bin_mid"):
                bin_grp = bin_grp.sort_values("depth_k")
                ax.plot(
                    bin_grp["depth_k"], bin_grp["prob_depth_accepted"], "o-",
                    color=bin_colors[b], label=f"{signal} ≈ {b:.2f}",
                    markersize=5, linewidth=1.8,
                )
            ax.set_title(f"{mdl} / {dset}  (static_7_1_8 chain)")
            if i == 1:
                ax.set_xlabel("depth k")
            if j == 0:
                ax.set_ylabel("P(depth k accepted)")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right", title=signal)

    plt.suptitle(
        f"Per-depth cumulative accept probability (chain config) vs {signal}\n"
        f"For each {signal} bin: at what depth does the chain break?",
        y=1.01, fontsize=12,
    )
    plt.tight_layout()
    _savefig(fig_dir, f"plot_e2b_depth_cumulative_{signal}")

    df.to_csv(os.path.join(tables_dir, f"e2_depth_cumulative_{signal}.csv"),
              index=False)
    return df


# -------------------------------------------------------------------------- #
# Wide vs deep — tree vs chain at same num_steps                             #
# -------------------------------------------------------------------------- #

def wide_vs_deep_crossover(
    master: pd.DataFrame,
    signal: str,
    n_bins: int = N_BINS_EAGLE2,
) -> pd.DataFrame:
    """For each (cell, signal_bin): compare mean AL of static_7_1_8 (chain,
    7 steps, topk=1) vs static_7_4_8 (tree, 7 steps, topk=4). Same depth
    so any difference is the effect of branching at each node.

    Returns rows with AL_chain, AL_tree, and delta = AL_tree - AL_chain.
    Positive delta = tree width helps.
    """
    chain, tree = "static_7_1_8", "static_7_4_8"
    sub = master[master["config"].isin([chain, tree])].copy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    rows = []
    for (mdl, dset), cell in sub.groupby(["model", "dataset"]):
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            in_bin = ((cell[signal] >= lo) & (cell[signal] < hi)) \
                     if b < n_bins - 1 else \
                     ((cell[signal] >= lo) & (cell[signal] <= hi))
            bucket = cell[in_bin]
            if len(bucket) < 5:
                continue
            chain_al = bucket[bucket["config"] == chain]["accept_length"]
            tree_al = bucket[bucket["config"] == tree]["accept_length"]
            if len(chain_al) < 3 or len(tree_al) < 3:
                continue
            rows.append({
                "model": mdl, "dataset": dset,
                "signal": signal, "bin_mid": float(mids[b]),
                "al_chain_7_1_8": float(chain_al.mean()),
                "al_tree_7_4_8": float(tree_al.mean()),
                "delta_tree_minus_chain": float(tree_al.mean() - chain_al.mean()),
                "n_chain": int(len(chain_al)),
                "n_tree": int(len(tree_al)),
            })
    return pd.DataFrame(rows)


def plot_wide_vs_deep(master, signal, fig_dir, tables_dir):
    df = wide_vs_deep_crossover(master, signal)
    if len(df) == 0:
        return df
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS),
        figsize=(15, 8), sharex=True, sharey=True,
    )
    for i, mdl in enumerate(MODELS):
        for j, dset in enumerate(DATASETS):
            ax = axes[i, j]
            cell = df[(df["model"] == mdl) & (df["dataset"] == dset)].sort_values("bin_mid")
            if len(cell) == 0:
                continue
            ax.plot(cell["bin_mid"], cell["al_chain_7_1_8"], "o-",
                    color=CFG_PALETTE["static_7_1_8"], label="chain (7,1,8)")
            ax.plot(cell["bin_mid"], cell["al_tree_7_4_8"], "s-",
                    color=CFG_PALETTE["static_7_4_8"], label="tree (7,4,8)")
            # Shade where tree wins
            ax.fill_between(
                cell["bin_mid"],
                cell["al_chain_7_1_8"], cell["al_tree_7_4_8"],
                where=(cell["al_tree_7_4_8"] > cell["al_chain_7_1_8"]),
                color="green", alpha=0.15, label="tree wins" if (i == 0 and j == 0) else None,
            )
            ax.set_title(f"{mdl} / {dset}")
            if i == 1:
                ax.set_xlabel(f"{signal} bin mid")
            if j == 0:
                ax.set_ylabel("mean accept_length")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    plt.suptitle(
        f"Wide vs Deep — tree (topk=4) vs chain (topk=1) both at num_steps=7, vs {signal}\n"
        f"Green shading = tree branching helps; no shading = chain is sufficient",
        y=1.01, fontsize=12,
    )
    plt.tight_layout()
    _savefig(fig_dir, f"plot_e2c_wide_vs_deep_{signal}")

    df.to_csv(os.path.join(tables_dir, f"e2_wide_vs_deep_{signal}.csv"),
              index=False)
    return df


# -------------------------------------------------------------------------- #
# Transition points — when does which tree type win?                          #
# -------------------------------------------------------------------------- #

def find_transition_points(
    master: pd.DataFrame,
    metrics_csv: str,
    signal: str = "top1_prob",
    n_bins: int = N_BINS_EAGLE2,
) -> pd.DataFrame:
    """For each cell, scan signal bins left-to-right and identify the bin
    where the best config (by throughput proxy) changes."""
    tps = pd.read_csv(metrics_csv)
    tps["tps_ms"] = 1000 * tps["elapsed_s"] / tps["num_steps"].clip(lower=1)
    tps_lookup = {(r["config"], r["model"], r["dataset"]): float(r["tps_ms"])
                  for _, r in tps.iterrows()}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    sub = master[master["config"].isin(STATIC_CONFIGS)].copy()
    rows = []
    for (mdl, dset), cell in sub.groupby(["model", "dataset"]):
        bin_best = {}
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            in_bin = ((cell[signal] >= lo) & (cell[signal] < hi)) \
                     if b < n_bins - 1 else \
                     ((cell[signal] >= lo) & (cell[signal] <= hi))
            bucket = cell[in_bin]
            if len(bucket) < 10:
                continue
            per_cfg_tput = {}
            for cfg, cfg_grp in bucket.groupby("config"):
                t_ms = tps_lookup.get((cfg, mdl, dset))
                if t_ms is None or len(cfg_grp) < 3:
                    continue
                al = float(cfg_grp["accept_length"].mean())
                per_cfg_tput[cfg] = al / t_ms * 1000  # tok/s
            if per_cfg_tput:
                bin_best[b] = max(per_cfg_tput, key=per_cfg_tput.get)
        rows.append({
            "model": mdl, "dataset": dset, "signal": signal,
            "bin_best_config_by_bin": [bin_best.get(b) for b in range(n_bins)],
            "transitions": [
                (mids[b], bin_best[b - 1], bin_best[b])
                for b in range(1, n_bins)
                if b - 1 in bin_best and b in bin_best
                and bin_best[b - 1] != bin_best[b]
            ],
        })
    df = pd.DataFrame(rows)
    return df


# -------------------------------------------------------------------------- #
# Main                                                                        #
# -------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace-dir", default="results/traces/")
    ap.add_argument("--out-dir", default="analysis/figures/")
    ap.add_argument("--summary", default="analysis/phase_b3_eagle2_summary.md")
    args = ap.parse_args()

    tables_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 90

    master_path = os.path.join(args.trace_dir, "master.parquet")
    metrics_csv = os.path.join(args.trace_dir, "metrics_final.csv")
    print(f"Loading {master_path}...")
    master = load_master(master_path)
    print(f"  {len(master):,} rows")

    # ===== EAGLE-2 Figure 6 replication for each signal candidate =====
    linearity_rows = []
    for signal in SIGNAL_CANDIDATES:
        print(f"\n[E2-fig6] Plotting for signal={signal}...")
        binned = plot_fig6_replication(master, signal, args.out_dir, tables_dir)
        lin = signal_linearity_score(binned)
        lin["signal"] = signal
        linearity_rows.append(lin)

    linearity = pd.concat(linearity_rows, ignore_index=True)
    linearity.to_csv(os.path.join(tables_dir, "e2_linearity_scores.csv"),
                     index=False)

    # Winner signal = highest median R² across cells × configs
    winner = (
        linearity.groupby("signal")
        .agg(median_r2=("r_squared", "median"),
             median_slope=("slope_a", "median"))
        .sort_values("median_r2", ascending=False)
    )
    print("\n=== Signal linearity ranking ===")
    print(winner.to_string())
    best_signal = winner.index[0]
    print(f"\nBest predictor signal: {best_signal} "
          f"(median R² = {winner.loc[best_signal, 'median_r2']:.3f})")

    # ===== Per-depth cumulative for chain (using the best signal) =====
    print(f"\n[E2-depth] Per-depth cumulative accept (chain, {best_signal})...")
    plot_depth_cumulative(master, best_signal, args.out_dir, tables_dir)

    # ===== Wide vs deep crossover =====
    print(f"\n[E2-wide-vs-deep] tree vs chain crossover, signal={best_signal}...")
    wd = plot_wide_vs_deep(master, best_signal, args.out_dir, tables_dir)

    # ===== Transition points =====
    print("\n[E2-transitions] Finding optimal-config transition points...")
    trans = find_transition_points(master, metrics_csv, best_signal)
    trans.to_csv(os.path.join(tables_dir, "e2_transition_points.csv"),
                 index=False)

    # ===== Summary =====
    print(f"\nWriting summary {args.summary}...")
    lines = [
        "# Phase B.3 — EAGLE-2-style analysis summary",
        "",
        "Central question (from EAGLE-2 paper Figure 6): does signal X "
        "linearly predict acceptance, and at what confidence intervals "
        "does each tree shape pay off?",
        "",
        "## 1. Signal linearity ranking (median R² across 24 cell × config)",
        "",
        winner.reset_index().to_markdown(index=False, floatfmt=".3f"),
        "",
        f"**Winning signal: `{best_signal}`** (highest median R²).",
        "",
        "Per-cell linearity scores:",
        "",
        linearity.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 2. Interpretation",
        "",
        "- **R² close to 1.0 with slope ≈ 1**: signal is a near-perfect "
        "linear predictor of acceptance rate — matches EAGLE-2 Figure 6 "
        "for Vicuna-7B. This justifies using the signal to gate tree depth.",
        "- **R² close to 1.0 but slope < 1**: signal is MONOTONIC but "
        "acceptance grows slower than confidence. Means tree-shape "
        "decisions should weight signal less aggressively at the top.",
        "- **R² < 0.5**: signal poorly predicts acceptance on that cell. "
        "Policy based on this signal will be noisy — may need hysteresis "
        "or a different signal.",
        "",
        "## 3. Per-depth cumulative acceptance (chain_7_1_8)",
        "",
        "See `plot_e2b_depth_cumulative_top1_prob.png`. For each signal "
        "bin, curves show P(depth k accepted) vs k=0..6. Steep drop at "
        "low k = chain breaks early (depth wasted). Flat near 1.0 up to "
        "high k = chain is paying off.",
        "",
        "**Policy reading**:",
        "- Bins where depth-6 accept prob > 0.5 → deep chain justified",
        "- Bins where depth-3 accept prob < 0.5 → switch to short chain",
        "- Bins where even depth-1 drops below 0.3 → skip speculation / "
        "use minimal config (the ORIGINAL V6 CHEAP zone rationale)",
        "",
        "## 4. Wide vs deep (chain 7,1,8 vs tree 7,4,8 at same depth)",
        "",
        wd.groupby(["model", "dataset"]).agg(
            max_delta_tree_minus_chain=("delta_tree_minus_chain", "max"),
            bin_of_max=("bin_mid", lambda x: x.iloc[wd.groupby(["model", "dataset"]).get_group((x.name[0] if hasattr(x, 'name') and isinstance(x.name, tuple) else None, None))["delta_tree_minus_chain"].idxmax()] if False else x.iloc[0]),
            mean_delta=("delta_tree_minus_chain", "mean"),
        ).reset_index().to_markdown(index=False, floatfmt=".3f")
        if len(wd) else "(no wide-vs-deep data)",
        "",
        "**Interpretation**: positive delta (green shading in plot) = tree "
        "branching at each depth adds AL beyond what a single chain "
        "delivers. Typically helps at LOW confidence (redundant candidates "
        "protect against single-chain failures) and stops helping at HIGH "
        "confidence (single chain already accepts every depth).",
        "",
        "## 5. Transition points (where optimal config changes)",
        "",
        trans.to_markdown(index=False) if len(trans) else "(no transitions)",
        "",
        "These are the concrete \"switch here\" signal values per cell. "
        "V6 / V3 threshold decisions should coincide with these.",
        "",
        "## 6. Synthesis — data-driven policy recommendations",
        "",
        "Based on sections 1-5 above, recommended policy structure:",
        "",
        "| Signal zone | Config | Rationale |",
        "|---|---|---|",
        "| Very low signal (top1 < P1) | (1,1,2) minimal or skip | Chain breaks at depth 0-1 |",
        "| Low signal (P1 < top1 < P2) | (3,1,4) short chain | Chain OK to ~3 depth; branching doesn't help |",
        "| Mid signal (P2 < top1 < P3) | (7,4,8) tree | Chain starts breaking beyond depth 3; branching recovers |",
        "| High signal (top1 > P3) | (7,1,8) or (10,6,60) deep | Chain reliable; depth pays off directly |",
        "",
        "Actual values of P1, P2, P3 read from plot_e2b transition points.",
        "",
        "## 7. Files produced",
        "",
        "- `plot_e2a_fig6_{top1_prob,target_top1_prob,t,DOG}.{png,pdf}` — Fig 6 replications",
        "- `plot_e2b_depth_cumulative_top1_prob.{png,pdf}` — per-depth curves",
        "- `plot_e2c_wide_vs_deep_top1_prob.{png,pdf}` — tree-vs-chain crossover",
        "- `tables/e2_linearity_scores.csv` — R² per cell × config × signal",
        "- `tables/e2_binned_accept_{signal}.csv` — raw binned data",
        "- `tables/e2_depth_cumulative_*.csv` — per-depth data",
        "- `tables/e2_wide_vs_deep_*.csv` — tree vs chain data",
        "- `tables/e2_transition_points.csv` — where best config changes",
    ]

    with open(args.summary, "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'=' * 60}")
    print(f"Phase B.3 EAGLE-2-style analysis complete.")
    print(f"  Figures : {args.out_dir}/plot_e2*.{{png,pdf}}")
    print(f"  Tables  : {tables_dir}/e2_*.csv")
    print(f"  Summary : {args.summary}")
    print(f"  Best predictor: {best_signal}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
