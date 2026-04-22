#!/usr/bin/env python3
"""
phase_b_run.py — Headless Phase B analysis runner.

Loads results/traces/master.parquet (or .pkl fallback) and generates all
11 figures + analysis/phase_b_summary.md in one go.  No Jupyter, no
browser, no port forwarding.

Usage:
    python3 analysis/phase_b_run.py
    python3 analysis/phase_b_run.py --trace-dir results/traces/ --out-dir analysis/figures/

Produces:
    analysis/figures/plot_1_first_reject_cdf.{png,pdf}
    analysis/figures/plot_2a_DOG_vs_AL.{png,pdf}
    analysis/figures/plot_2b_t_vs_AL.{png,pdf}
    analysis/figures/plot_3_cdf_overlap.{png,pdf}
    analysis/figures/plot_3b_ks_matrices.{png,pdf}
    analysis/figures/plot_4a_buy_DOG.{png,pdf}
    analysis/figures/plot_4b_buy_t.{png,pdf}
    analysis/figures/plot_5a_best_config_per_decile.{png,pdf}
    analysis/phase_b_summary.md
"""

from __future__ import annotations

import argparse
import os
import sys

# headless backend first — must precede pyplot import
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

# local imports
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase_b_analysis import (  # noqa: E402
    load_master,
    STATIC_CONFIGS, DYNAMIC_CONFIGS, CONFIG_SHAPE,
    MODELS, DATASETS, SIGNALS,
    plot1_first_reject_cdf, plot1_wasted_tail_table,
    plot2_signal_deciles, plot2_spearman,
    plot3_ks_matrix, plot3_ks_summary,
    plot4_buy_signal,
    plot5_best_config_per_decile, plot5_derive_thresholds,
    phase_b_summary_md,
)


def _savefig(fig_dir: str, name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(
            os.path.join(fig_dir, f"{name}.{ext}"),
            bbox_inches="tight",
            dpi=150 if ext == "png" else 300,
        )
    plt.close()


# -------------------------------------------------------------------------- #
# Plot 1 — First-reject CDF per static config × cell                         #
# -------------------------------------------------------------------------- #
def run_plot1(master: pd.DataFrame, fig_dir: str) -> pd.DataFrame:
    print("  plot 1 — first-reject CDFs...")
    cdfs = plot1_first_reject_cdf(master)

    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS), figsize=(15, 8),
        sharex=False, sharey=True,
    )
    config_colors = {
        c: col for c, col in zip(STATIC_CONFIGS, sns.color_palette("tab10", 4))
    }

    for i, mdl in enumerate(MODELS):
        for j, dset in enumerate(DATASETS):
            ax = axes[i, j]
            for cfg in STATIC_CONFIGS:
                key = (cfg, mdl, dset)
                if key not in cdfs:
                    continue
                d = cdfs[key]
                ax.step(
                    d["position"], d["cum_fraction"], where="post",
                    label=f"{cfg} (ns={CONFIG_SHAPE[cfg][0]})",
                    color=config_colors[cfg], linewidth=2,
                )
            ax.set_title(f"{mdl} / {dset}")
            ax.set_xlabel("first rejected position")
            if j == 0:
                ax.set_ylabel("cumulative fraction of steps")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=8)

    plt.suptitle(
        "Plot 1 — CDF of first-rejected draft position per static config",
        y=1.02, fontsize=13,
    )
    plt.tight_layout()
    _savefig(fig_dir, "plot_1_first_reject_cdf")

    wasted = plot1_wasted_tail_table(master)
    return wasted


# -------------------------------------------------------------------------- #
# Plot 2 — Signal discriminability (DOG, t)                                  #
# -------------------------------------------------------------------------- #
def _plot_signal_to_al(
    master: pd.DataFrame, signal: str, label: str, fig_dir: str, name: str
) -> None:
    tab = plot2_signal_deciles(master, signal)
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS), figsize=(15, 8),
        sharex=True, sharey=True,
    )
    config_colors = {
        c: col for c, col in zip(STATIC_CONFIGS, sns.color_palette("tab10", 4))
    }

    for i, mdl in enumerate(MODELS):
        for j, dset in enumerate(DATASETS):
            ax = axes[i, j]
            sub = tab[(tab["model"] == mdl) & (tab["dataset"] == dset)]
            for cfg in STATIC_CONFIGS:
                s = sub[sub["config"] == cfg].sort_values("decile")
                if len(s) == 0:
                    continue
                ax.plot(
                    s["signal_mid"], s["mean_al"], "o-",
                    color=config_colors[cfg], label=cfg, markersize=5,
                )
                ax.fill_between(
                    s["signal_mid"], s["ci_lo"], s["ci_hi"],
                    color=config_colors[cfg], alpha=0.15,
                )
            ax.set_title(f"{mdl} / {dset}")
            if i == 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel("mean accept_length")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.suptitle(
        f"Plot 2 — {label} (deciles) vs mean accept_length",
        y=1.02, fontsize=13,
    )
    plt.tight_layout()
    _savefig(fig_dir, name)


def run_plot2(master: pd.DataFrame, fig_dir: str):
    print("  plot 2 — signal discriminability (DOG, t)...")
    _plot_signal_to_al(
        master, "DOG", "DOG = top1 × RAR (V3 signal)",
        fig_dir, "plot_2a_DOG_vs_AL",
    )
    _plot_signal_to_al(
        master, "t", "t = 0.7·top1 + 0.3·target_top1 (V6 signal)",
        fig_dir, "plot_2b_t_vs_AL",
    )

    spearman_dog = plot2_spearman(master, "DOG")
    spearman_t = plot2_spearman(master, "t")
    print(
        f"    DOG Spearman (median): {spearman_dog['spearman_r'].median():.3f}"
    )
    print(
        f"    t   Spearman (median): {spearman_t['spearman_r'].median():.3f}"
    )
    return spearman_dog, spearman_t


# -------------------------------------------------------------------------- #
# Plot 3 — Cross-cell CDF overlap + KS                                       #
# -------------------------------------------------------------------------- #
def run_plot3(master: pd.DataFrame, fig_dir: str) -> pd.DataFrame:
    print("  plot 3 — cross-cell CDFs + KS matrices...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    palette = sns.color_palette("husl", n_colors=6)
    cells = [(m, d) for m in MODELS for d in DATASETS]
    cell_colors = {c: col for c, col in zip(cells, palette)}

    static_sub = master[master["config"].isin(STATIC_CONFIGS)]
    for idx, signal in enumerate(SIGNALS):
        ax = axes.flat[idx]
        for mdl in MODELS:
            for dset in DATASETS:
                vals = static_sub[
                    (static_sub["model"] == mdl)
                    & (static_sub["dataset"] == dset)
                ][signal].to_numpy()
                if len(vals) == 0:
                    continue
                sorted_vals = np.sort(vals)
                cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.plot(
                    sorted_vals, cdf,
                    color=cell_colors[(mdl, dset)],
                    label=f"{mdl}/{dset}", alpha=0.9, linewidth=1.8,
                )
        ax.set_title(signal)
        ax.set_xlabel("signal value")
        ax.set_ylabel("CDF")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    axes.flat[-1].axis("off")
    plt.suptitle(
        "Plot 3 — Cross-cell empirical CDFs per signal (static-only)",
        y=1.00, fontsize=13,
    )
    plt.tight_layout()
    _savefig(fig_dir, "plot_3_cdf_overlap")

    # KS matrices
    ks_mats = {sig: plot3_ks_matrix(master, sig) for sig in SIGNALS}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (signal, mat) in enumerate(ks_mats.items()):
        ax = axes.flat[idx]
        sns.heatmap(
            mat.astype(float), annot=True, fmt=".2f",
            vmin=0, vmax=0.3, cmap="YlOrRd",
            cbar=False, ax=ax, square=True, linewidths=0.5,
        )
        ax.set_title(f"KS({signal})")
    axes.flat[-1].axis("off")
    plt.suptitle(
        "Plot 3b — Pairwise KS divergence per signal (static-only traces)",
        y=1.02, fontsize=13,
    )
    plt.tight_layout()
    _savefig(fig_dir, "plot_3b_ks_matrices")

    ks_summary = plot3_ks_summary(ks_mats)
    print("    KS summary:")
    print(ks_summary.to_string(index=False))
    return ks_summary


# -------------------------------------------------------------------------- #
# Plot 4 — Buy zone                                                          #
# -------------------------------------------------------------------------- #
def _plot_buy(master: pd.DataFrame, signal: str, label: str,
              fig_dir: str, name: str) -> None:
    buy = plot4_buy_signal(master, signal)
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS),
        figsize=(15, 8), sharex=True,
    )
    for i, mdl in enumerate(MODELS):
        for j, dset in enumerate(DATASETS):
            ax = axes[i, j]
            sub = buy[(buy["model"] == mdl) & (buy["dataset"] == dset)]
            agg = (
                sub.groupby("signal_mid", as_index=False)
                .agg(
                    overspend=("overspend", "max"),
                    underspend=("underspend", "max"),
                )
                .sort_values("signal_mid")
            )
            if len(agg) == 0:
                continue
            ax.plot(
                agg["signal_mid"], agg["overspend"], "o-",
                color="#d62728", label="overspend (big tree waste)",
            )
            ax.plot(
                agg["signal_mid"], agg["underspend"], "s-",
                color="#2ca02c", label="underspend (chain ceiling hit)",
            )
            ax.set_title(f"{mdl} / {dset}")
            if i == 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel("mean metric")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    plt.suptitle(f"Plot 4 — Buy-zone analysis by {label}", y=1.02, fontsize=13)
    plt.tight_layout()
    _savefig(fig_dir, name)


def run_plot4(master: pd.DataFrame, fig_dir: str) -> None:
    print("  plot 4 — buy-zone analysis...")
    _plot_buy(master, "DOG", "DOG decile mid", fig_dir, "plot_4a_buy_DOG")
    _plot_buy(master, "t", "t decile mid", fig_dir, "plot_4b_buy_t")


# -------------------------------------------------------------------------- #
# Plot 5 — Threshold recommendations                                          #
# -------------------------------------------------------------------------- #
def run_plot5(master: pd.DataFrame, fig_dir: str):
    print("  plot 5 — best-config-per-decile + threshold derivation...")

    best_dog = plot5_best_config_per_decile(master, "DOG")
    best_t = plot5_best_config_per_decile(master, "t")

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9), sharex=True, sharey=True,
    )
    cfg_palette = {
        c: col for c, col in zip(STATIC_CONFIGS, sns.color_palette("tab10", 4))
    }

    for ax_row, (sig_name, bpd) in enumerate(
        [("DOG", best_dog), ("t", best_t)]
    ):
        for j, dset in enumerate(DATASETS):
            ax = axes[ax_row, j]
            sub = bpd[bpd["dataset"] == dset]
            for mdl in MODELS:
                sub_m = sub[sub["model"] == mdl]
                for _, row in sub_m.iterrows():
                    ax.scatter(
                        row["signal_mid"], mdl,
                        color=cfg_palette[row["best_config"]],
                        s=100, edgecolor="black", linewidth=0.5,
                    )
            ax.set_title(f"{sig_name} — {dset}")
            ax.set_xlabel(f"{sig_name} decile mid")
            if j == 0:
                ax.set_ylabel(f"model ({sig_name})")

    from matplotlib.patches import Patch
    handles = [Patch(color=cfg_palette[c], label=c) for c in STATIC_CONFIGS]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=4, bbox_to_anchor=(0.5, -0.03),
    )
    plt.suptitle(
        "Plot 5 — Best static config per signal decile per cell",
        y=1.00, fontsize=13,
    )
    plt.tight_layout()
    _savefig(fig_dir, "plot_5a_best_config_per_decile")

    thresholds_dog = plot5_derive_thresholds(best_dog)
    thresholds_t = plot5_derive_thresholds(best_t)

    print("    DOG thresholds per cell:")
    print(thresholds_dog.to_string(index=False))
    print("    t thresholds per cell:")
    print(thresholds_t.to_string(index=False))

    # Global thresholds = median per-cell
    def _med(df, col):
        vals = df[col].dropna()
        return float(vals.median()) if len(vals) else float("nan")

    global_dog = {
        "cheap": _med(thresholds_dog, "cheap_threshold"),
        "premium": _med(thresholds_dog, "premium_threshold"),
    }
    global_t = {
        "cheap": _med(thresholds_t, "cheap_threshold"),
        "premium": _med(thresholds_t, "premium_threshold"),
    }
    print(
        f"\n    DOG global (median per-cell):  "
        f"cheap={global_dog['cheap']:.3f}  premium={global_dog['premium']:.3f}"
    )
    print(
        f"    t   global (median per-cell):  "
        f"cheap={global_t['cheap']:.3f}  premium={global_t['premium']:.3f}"
    )
    print("    V6 current constants (reference): cheap=0.30  premium=0.95")

    return thresholds_dog, thresholds_t, global_dog, global_t


# -------------------------------------------------------------------------- #
# Main                                                                        #
# -------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--trace-dir", default="results/traces/",
        help="Where master.parquet lives (default: results/traces/)",
    )
    ap.add_argument(
        "--out-dir", default="analysis/figures/",
        help="Where to write figures (default: analysis/figures/)",
    )
    ap.add_argument(
        "--summary", default="analysis/phase_b_summary.md",
        help="Output markdown path (default: analysis/phase_b_summary.md)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 90

    # -- load --
    master_path = os.path.join(args.trace_dir, "master.parquet")
    print(f"Loading {master_path}...")
    master = load_master(master_path)
    print(f"  {len(master):,} rows × {len(master.columns)} cols")

    # -- generate all plots --
    print("\nGenerating plots...")
    wasted = run_plot1(master, args.out_dir)
    spearman_dog, spearman_t = run_plot2(master, args.out_dir)
    ks_summary = run_plot3(master, args.out_dir)
    run_plot4(master, args.out_dir)
    thresholds_dog, thresholds_t, global_dog, global_t = run_plot5(
        master, args.out_dir
    )

    # -- summary markdown --
    print("\nWriting summary...")
    summary_txt = phase_b_summary_md(
        ks_summary=ks_summary,
        spearman_dog=spearman_dog,
        spearman_t=spearman_t,
        thresholds_t=thresholds_t,
        thresholds_dog=thresholds_dog,
        wasted_tail=wasted,
    )
    with open(args.summary, "w") as f:
        f.write(summary_txt)
    print(f"  wrote {args.summary}")

    # Also save a machine-readable bundle of the key tables
    bundle_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(bundle_dir, exist_ok=True)
    wasted.to_csv(os.path.join(bundle_dir, "wasted_tail.csv"), index=False)
    spearman_dog.to_csv(os.path.join(bundle_dir, "spearman_dog.csv"), index=False)
    spearman_t.to_csv(os.path.join(bundle_dir, "spearman_t.csv"), index=False)
    ks_summary.to_csv(os.path.join(bundle_dir, "ks_summary.csv"), index=False)
    thresholds_dog.to_csv(
        os.path.join(bundle_dir, "thresholds_dog.csv"), index=False
    )
    thresholds_t.to_csv(
        os.path.join(bundle_dir, "thresholds_t.csv"), index=False
    )
    pd.DataFrame([
        {"signal": "DOG", **global_dog},
        {"signal": "t", **global_t},
    ]).to_csv(os.path.join(bundle_dir, "global_thresholds.csv"), index=False)
    print(f"  wrote tables to {bundle_dir}/")

    print(f"\n{'=' * 60}")
    print("Phase B analysis complete.")
    print(f"  Figures : {args.out_dir}/plot_*.{{png,pdf}}")
    print(f"  Tables  : {bundle_dir}/*.csv")
    print(f"  Summary : {args.summary}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
