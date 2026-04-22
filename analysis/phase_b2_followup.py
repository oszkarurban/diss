#!/usr/bin/env python3
"""
phase_b2_followup.py — Phase B.2 detailed follow-up analysis.

Runs sections B.2a through B.2h from the plan:

    B.2a  Fix Plot 5 with proper throughput-proxy ranking
    B.2b  Per-cell views (no cross-cell aggregation)
    B.2c  Fix Plot 4 (per-config lines, dual y-axis) + buy-zone CSVs
    B.2d  Actual vs ideal: what did v3/v6 pick vs what was optimal
    B.2e  Per-model threshold simulation (V6 global vs per-model vs ideal)
    B.2f  Adaptive-quantile feasibility test
    B.2g  V3 capability analysis (is it saturated? is ideal reachable?)
    B.2h  Phase C scope recommendation table

Usage:
    python3 analysis/phase_b2_followup.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase_b_analysis import (  # noqa: E402
    load_master,
    STATIC_CONFIGS, DYNAMIC_CONFIGS, CONFIG_SHAPE,
    MODELS, DATASETS, SIGNALS,
    plot2_spearman,
    plot4_buy_signal,
    # B.2 new
    plot5_best_config_per_decile_v2, plot5_derive_thresholds_v2,
    per_cell_spearman_matrix, per_model_aggregates, per_dataset_aggregates,
    actual_vs_ideal_per_decile,
    simulate_policy_throughput,
    adaptive_quantile_feasibility,
    v3_capability_analysis,
)


CFG_PALETTE = {
    "static_3_1_4":   "#1f77b4",
    "static_7_1_8":   "#ff7f0e",
    "static_7_4_8":   "#2ca02c",
    "static_6_10_60": "#d62728",
}


def _savefig(fig_dir: str, name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(
            os.path.join(fig_dir, f"{name}.{ext}"),
            bbox_inches="tight",
            dpi=150 if ext == "png" else 300,
        )
    plt.close()


# -------------------------------------------------------------------------- #
# B.2a — Fixed Plot 5                                                        #
# -------------------------------------------------------------------------- #
def run_b2a(master, metrics_csv, fig_dir, tables_dir):
    print("\n[B.2a] Fixed Plot 5 — best config per decile by throughput proxy")

    best_t = plot5_best_config_per_decile_v2(master, "t", metrics_csv)
    best_dog = plot5_best_config_per_decile_v2(master, "DOG", metrics_csv)
    best_t.to_csv(os.path.join(tables_dir, "best_per_decile_t_v2.csv"), index=False)
    best_dog.to_csv(os.path.join(tables_dir, "best_per_decile_dog_v2.csv"), index=False)

    thresholds_t = plot5_derive_thresholds_v2(best_t)
    thresholds_dog = plot5_derive_thresholds_v2(best_dog)
    thresholds_t.to_csv(os.path.join(tables_dir, "thresholds_t_v2.csv"), index=False)
    thresholds_dog.to_csv(os.path.join(tables_dir, "thresholds_dog_v2.csv"), index=False)

    print("  Per-cell thresholds (signal=t, v2):")
    print(thresholds_t.to_string(index=False))

    # Plot: 6-panel grid, one line per config's estimated throughput
    for signal_name, best in [("t", best_t), ("DOG", best_dog)]:
        fig, axes = plt.subplots(
            len(MODELS), len(DATASETS),
            figsize=(16, 9), sharex=True, sharey=False,
        )
        for i, mdl in enumerate(MODELS):
            for j, dset in enumerate(DATASETS):
                ax = axes[i, j]
                sub = best[(best["model"] == mdl) & (best["dataset"] == dset)]
                if len(sub) == 0:
                    ax.set_title(f"{mdl} / {dset}  (no data)")
                    continue
                for cfg in STATIC_CONFIGS:
                    col = f"tput_{cfg}"
                    if col not in sub.columns:
                        continue
                    ax.plot(
                        sub["signal_mid"], sub[col], "o-",
                        color=CFG_PALETTE[cfg], label=cfg,
                        markersize=5, linewidth=1.8,
                    )
                # Mark the best per decile with a black star
                for _, r in sub.iterrows():
                    best_col = f"tput_{r['best_config']}"
                    if best_col in r:
                        ax.scatter(
                            r["signal_mid"], r[best_col],
                            marker="*", s=140, color="black",
                            edgecolor="white", linewidth=1.2, zorder=5,
                        )
                ax.set_title(f"{mdl} / {dset}")
                if i == 1:
                    ax.set_xlabel(f"{signal_name} decile mid")
                if j == 0:
                    ax.set_ylabel("est. throughput (tok/s)")
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, loc="lower right")
        plt.suptitle(
            f"Plot 5 v2 — estimated throughput per static config, by {signal_name} decile "
            f"(★ = best per decile)",
            y=1.01, fontsize=13,
        )
        plt.tight_layout()
        _savefig(fig_dir, f"plot_5_v2_best_config_{signal_name}")

    return best_t, best_dog, thresholds_t, thresholds_dog


# -------------------------------------------------------------------------- #
# B.2b — Per-cell detail & model/dataset aggregates                          #
# -------------------------------------------------------------------------- #
def run_b2b(master, fig_dir, tables_dir):
    print("\n[B.2b] Per-cell Spearman r + aggregation heatmaps")

    per_cell_t = per_cell_spearman_matrix(master, "t")
    per_cell_dog = per_cell_spearman_matrix(master, "DOG")
    per_cell_t.to_csv(os.path.join(tables_dir, "per_cell_spearman_t.csv"), index=False)
    per_cell_dog.to_csv(os.path.join(tables_dir, "per_cell_spearman_dog.csv"), index=False)

    per_model_t = per_model_aggregates(per_cell_t)
    per_model_dog = per_model_aggregates(per_cell_dog)
    per_dataset_t = per_dataset_aggregates(per_cell_t)
    per_dataset_dog = per_dataset_aggregates(per_cell_dog)
    per_model_t.to_csv(os.path.join(tables_dir, "per_model_spearman_t.csv"), index=False)
    per_model_dog.to_csv(os.path.join(tables_dir, "per_model_spearman_dog.csv"), index=False)
    per_dataset_t.to_csv(os.path.join(tables_dir, "per_dataset_spearman_t.csv"), index=False)
    per_dataset_dog.to_csv(os.path.join(tables_dir, "per_dataset_spearman_dog.csv"), index=False)

    # Heatmap: 4 configs × 6 cells for each signal
    for signal_name, per_cell in [("t", per_cell_t), ("DOG", per_cell_dog)]:
        pv = per_cell.pivot_table(
            index="config", columns=["model", "dataset"],
            values="spearman_r",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            pv, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, cbar=True, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Per-cell Spearman r (signal={signal_name})")
        plt.tight_layout()
        _savefig(fig_dir, f"plot_7_per_cell_spearman_{signal_name}")

    print("  Per-model medians (t):")
    print(per_model_t.to_string(index=False))
    print("  Per-dataset medians (t):")
    print(per_dataset_t.to_string(index=False))

    return per_cell_t, per_cell_dog, per_model_t, per_dataset_t


# -------------------------------------------------------------------------- #
# B.2c — Fixed Plot 4 + buy-zone opportunity CSV                             #
# -------------------------------------------------------------------------- #
def run_b2c(master, fig_dir, tables_dir, best_t, best_dog):
    print("\n[B.2c] Plot 4 per-config lines + buy-zone opportunities CSV")

    buy_t = plot4_buy_signal(master, "t")
    buy_dog = plot4_buy_signal(master, "DOG")
    buy_t.to_csv(os.path.join(tables_dir, "buy_t.csv"), index=False)
    buy_dog.to_csv(os.path.join(tables_dir, "buy_dog.csv"), index=False)

    for signal_name, buy in [("t", buy_t), ("DOG", buy_dog)]:
        fig, axes = plt.subplots(
            len(MODELS), len(DATASETS),
            figsize=(16, 9), sharex=True, sharey=False,
        )
        for i, mdl in enumerate(MODELS):
            for j, dset in enumerate(DATASETS):
                ax_os = axes[i, j]
                ax_us = ax_os.twinx()  # right-hand axis for underspend
                sub = buy[(buy["model"] == mdl) & (buy["dataset"] == dset)]
                for cfg in STATIC_CONFIGS:
                    sc = sub[sub["config"] == cfg].sort_values("decile")
                    if len(sc) == 0:
                        continue
                    # Overspend (dashed, left axis)
                    ax_os.plot(
                        sc["signal_mid"], sc["overspend"], "--",
                        color=CFG_PALETTE[cfg], alpha=0.85, linewidth=1.8,
                        label=f"{cfg} overspend",
                    )
                    # Underspend (solid, right axis)
                    ax_us.plot(
                        sc["signal_mid"], sc["underspend"], "-",
                        color=CFG_PALETTE[cfg], alpha=0.85, linewidth=1.8,
                        label=f"{cfg} underspend",
                    )
                ax_os.set_title(f"{mdl} / {dset}")
                if i == 1:
                    ax_os.set_xlabel(f"{signal_name} decile mid")
                if j == 0:
                    ax_os.set_ylabel("overspend  (ndt − AL, if ndt≥8)",
                                     color="dimgray")
                if j == len(DATASETS) - 1:
                    ax_us.set_ylabel("underspend  (fraction at ceiling)",
                                     color="dimgray")
                ax_os.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax_os.legend(fontsize=6, loc="upper left", ncol=1)
                    ax_us.legend(fontsize=6, loc="upper right", ncol=1)
        plt.suptitle(
            f"Plot 4 v2 — per-config overspend (dashed, left) vs "
            f"underspend (solid, right) by {signal_name} decile",
            y=1.01, fontsize=13,
        )
        plt.tight_layout()
        _savefig(fig_dir, f"plot_4_v2_per_config_buy_{signal_name}")

    # Build the buy_zone_opportunities table from the v2 best-per-decile data.
    # For each (cell, decile), report: current best config + transition flag
    rows = []
    for _, r in best_t.iterrows():
        rows.append({
            "signal": "t",
            "model": r["model"],
            "dataset": r["dataset"],
            "decile": r["decile"],
            "signal_mid": r["signal_mid"],
            "n_steps_in_decile": r["n_steps_in_decile"],
            "best_config": r["best_config"],
            "best_tput_tok_s": r["best_est_tput_tok_s"],
            "best_al": r["best_al"],
        })
    for _, r in best_dog.iterrows():
        rows.append({
            "signal": "DOG",
            "model": r["model"],
            "dataset": r["dataset"],
            "decile": r["decile"],
            "signal_mid": r["signal_mid"],
            "n_steps_in_decile": r["n_steps_in_decile"],
            "best_config": r["best_config"],
            "best_tput_tok_s": r["best_est_tput_tok_s"],
            "best_al": r["best_al"],
        })
    opp = pd.DataFrame(rows).sort_values(
        ["signal", "model", "dataset", "decile"]
    )
    opp.to_csv(os.path.join(tables_dir, "buy_zone_opportunities.csv"), index=False)
    print(f"  Wrote {len(opp)} buy-zone rows")


# -------------------------------------------------------------------------- #
# B.2d — Actual vs ideal                                                     #
# -------------------------------------------------------------------------- #
def run_b2d(master, best_t, fig_dir, tables_dir):
    print("\n[B.2d] Actual dynamic picks vs ideal best static per decile")

    avi_v3 = actual_vs_ideal_per_decile(master, best_t, "t", "v3_dynamic")
    avi_v6 = actual_vs_ideal_per_decile(master, best_t, "t", "v6_dynamic")
    combined = pd.concat([avi_v3, avi_v6], ignore_index=True)
    combined.to_csv(os.path.join(tables_dir, "actual_vs_ideal.csv"), index=False)

    # Match-rate summary per policy × cell
    if len(combined) > 0:
        match_rate = (
            combined.groupby(["policy", "model", "dataset"], as_index=False)
            .agg(match_rate=("match", "mean"),
                 n_deciles=("match", "count"))
        )
        match_rate.to_csv(
            os.path.join(tables_dir, "actual_vs_ideal_match_rate.csv"),
            index=False,
        )
        print("  Match rate per (policy, cell):")
        print(match_rate.to_string(index=False))

    # Plot: grid of policy × cell; each panel is decile on x, bar = actual vs ideal
    policies = combined["policy"].unique().tolist() if len(combined) else []
    if policies:
        fig, axes = plt.subplots(
            len(policies), len(DATASETS) * len(MODELS),
            figsize=(18, 4 * len(policies)), sharey=True,
        )
        if len(policies) == 1:
            axes = np.array([axes])
        for pi, pol in enumerate(policies):
            col_idx = 0
            for mdl in MODELS:
                for dset in DATASETS:
                    ax = axes[pi, col_idx]
                    sub = combined[
                        (combined["policy"] == pol)
                        & (combined["model"] == mdl)
                        & (combined["dataset"] == dset)
                    ].sort_values("decile")
                    if len(sub):
                        colors = ["#2ca02c" if m else "#d62728"
                                  for m in sub["match"]]
                        ax.bar(sub["decile"], sub["actual_mode_fraction"],
                               color=colors, edgecolor="black", linewidth=0.5)
                        for _, r in sub.iterrows():
                            ax.text(r["decile"], r["actual_mode_fraction"] + 0.02,
                                    r["actual_config"].replace("static_", "")
                                    .replace("custom", "c"),
                                    ha="center", fontsize=6, rotation=75)
                    ax.set_title(f"{pol[:2]} {mdl[:2]}/{dset[:3]}", fontsize=9)
                    ax.set_ylim(0, 1.2)
                    if col_idx == 0:
                        ax.set_ylabel("mode fraction")
                    col_idx += 1
        plt.suptitle(
            "Plot 6 — actual policy pick (mode) per t-decile, green=match ideal",
            y=1.01, fontsize=13,
        )
        plt.tight_layout()
        _savefig(fig_dir, "plot_6_actual_vs_ideal")


# -------------------------------------------------------------------------- #
# B.2e — Threshold simulation                                                #
# -------------------------------------------------------------------------- #
def run_b2e(master, metrics_csv, best_t, thresholds_t, fig_dir, tables_dir):
    print("\n[B.2e] Per-model threshold simulation")

    # 1. Global V6 current: 0.30 / 0.95
    sim_global_v6 = simulate_policy_throughput(
        master, best_t, metrics_csv, "t",
        cheap_threshold=0.30, premium_threshold=0.95,
    ).assign(variant="v6_current_global")

    # 2. Per-model tuned: compute median thresholds per model from thresholds_t
    per_model = thresholds_t.groupby("model").agg(
        cheap=("cheap_threshold", "median"),
        premium=("premium_threshold", "median"),
    ).reset_index()
    per_model_sims = []
    for _, mr in per_model.iterrows():
        only_mdl = master[master["model"] == mr["model"]]
        only_ideal = best_t[best_t["model"] == mr["model"]]
        sim = simulate_policy_throughput(
            only_mdl, only_ideal, metrics_csv, "t",
            cheap_threshold=mr["cheap"], premium_threshold=mr["premium"],
        ).assign(variant=f"per_model_{mr['model']}")
        per_model_sims.append(sim)

    # 3. Ideal switching upper bound
    sim_ideal = simulate_policy_throughput(
        master, best_t, metrics_csv, "t",
        use_ideal_switching=True,
    ).assign(variant="ideal_per_decile")

    all_sims = pd.concat(
        [sim_global_v6, *per_model_sims, sim_ideal], ignore_index=True
    )
    # Add gain vs global
    global_tput = sim_global_v6.set_index(["model", "dataset"])["expected_tput_tok_s"]
    all_sims["gain_vs_global_pct"] = all_sims.apply(
        lambda r: 100 * (r["expected_tput_tok_s"]
                         / global_tput.get((r["model"], r["dataset"]), np.nan) - 1),
        axis=1,
    )
    all_sims.to_csv(
        os.path.join(tables_dir, "threshold_simulation.csv"), index=False,
    )
    print("  Throughput simulation:")
    print(all_sims.to_string(index=False))


# -------------------------------------------------------------------------- #
# B.2f — Adaptive quantile feasibility                                       #
# -------------------------------------------------------------------------- #
def run_b2f(master, thresholds_t, tables_dir):
    print("\n[B.2f] Adaptive-quantile feasibility")
    feas = adaptive_quantile_feasibility(master, thresholds_t, "t")
    feas.to_csv(os.path.join(tables_dir, "adaptive_quantile_feasibility.csv"),
                index=False)
    print("  Per-cell threshold percentiles within each cell's own distribution:")
    print(feas.to_string(index=False))


# -------------------------------------------------------------------------- #
# B.2g — V3 capability analysis                                              #
# -------------------------------------------------------------------------- #
def run_b2g(master, best_t, tables_dir):
    print("\n[B.2g] V3 capability analysis")
    cap = v3_capability_analysis(master, best_t)
    cap.to_csv(os.path.join(tables_dir, "v3_capability.csv"), index=False)
    print(cap.to_string(index=False))


# -------------------------------------------------------------------------- #
# B.2h — Phase C recommendation table                                        #
# -------------------------------------------------------------------------- #
def run_b2h(tables_dir, summary_path):
    print("\n[B.2h] Compose phase_b2_summary.md")

    # Load all tables produced above
    def _load(name):
        p = os.path.join(tables_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    thresholds_t_v2 = _load("thresholds_t_v2.csv")
    per_cell_t = _load("per_cell_spearman_t.csv")
    per_cell_dog = _load("per_cell_spearman_dog.csv")
    per_model_t = _load("per_model_spearman_t.csv")
    per_dataset_t = _load("per_dataset_spearman_t.csv")
    sim = _load("threshold_simulation.csv")
    feas = _load("adaptive_quantile_feasibility.csv")
    cap = _load("v3_capability.csv")
    match_rate = _load("actual_vs_ideal_match_rate.csv")

    lines = ["# Phase B.2 — Follow-up analysis summary", ""]

    # 1. Per-cell Spearman (primary)
    lines += ["## Per-cell Spearman r (signal = t)", ""]
    if per_cell_t is not None:
        lines += [per_cell_t.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += ["## Per-cell Spearman r (signal = DOG)", ""]
    if per_cell_dog is not None:
        lines += [per_cell_dog.to_markdown(index=False, floatfmt=".3f"), ""]

    lines += ["## Per-model medians (supplementary, signal = t)", ""]
    if per_model_t is not None:
        lines += [per_model_t.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += ["## Per-dataset medians (supplementary, signal = t)", ""]
    if per_dataset_t is not None:
        lines += [per_dataset_t.to_markdown(index=False, floatfmt=".3f"), ""]

    # 2. v2 thresholds
    lines += ["## Per-cell thresholds derived from Plot-5 v2 (signal = t)", ""]
    if thresholds_t_v2 is not None:
        lines += [thresholds_t_v2.to_markdown(index=False, floatfmt=".3f"), ""]

    # 3. Simulation
    lines += ["## Policy simulation — expected throughput per cell", ""]
    if sim is not None:
        lines += [sim.to_markdown(index=False, floatfmt=".2f"), ""]
    lines += [
        "Interpretation: compare `per_model_*` rows vs `v6_current_global`. ",
        "Gains > 3% on a cell justify per-model tuning for Phase C. The ",
        "`ideal_per_decile` row is an UPPER BOUND (requires oracle) — the ",
        "gap between per_model and ideal is the remaining unexplained gain.",
        "",
    ]

    # 4. Adaptive feasibility
    lines += ["## Adaptive-quantile feasibility", ""]
    if feas is not None:
        lines += [feas.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += [
        "Interpretation: if `cheap_percentile` is tightly clustered across "
        "the 6 cells (e.g. all in 0.25–0.40), a fixed rolling-quantile "
        "threshold can replace per-model constants. If values are spread "
        "> 0.2 apart, adaptive quantiles won't generalise.",
        "",
    ]

    # 5. V3 capability
    lines += ["## V3 capability", ""]
    if cap is not None:
        lines += [cap.to_markdown(index=False, floatfmt=".3f"), ""]
    lines += [
        "Interpretation: `all_ideal_within_v3_range=False` means V3's ceiling "
        "of (7,4,8) is too low to reach that cell's optimal config. Phase C "
        "should widen V3's max for those cells.",
        "",
    ]

    # 6. Actual-vs-ideal match rate
    lines += ["## V3 / V6 actual pick vs ideal (match rate)", ""]
    if match_rate is not None:
        lines += [match_rate.to_markdown(index=False, floatfmt=".3f"), ""]

    # 7. Recommendation table
    lines += ["## Phase C recommendation (prioritised)", ""]
    lines += [
        "| Action | Effort | Gain signal |",
        "|---|---|---|",
        "| V6 per-model thresholds | ~20 LOC | See §Policy simulation table |",
        "| V6 adaptive quantile thresholds | ~30 LOC | Depends on §Adaptive feasibility |",
        "| V6 higher max bound (→ 10,10,60 on Llama) | 0 LOC | Lift ceiling above best static |",
        "| V3 wider max bound (→ 6,10,60) | 0 LOC | V3 cannot reach best static otherwise |",
        "",
        "Ranking depends on the numerical tables above. Pick the top 2–3 for Phase C runs.",
        "",
    ]

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {summary_path}")


# -------------------------------------------------------------------------- #
# Main                                                                        #
# -------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace-dir", default="results/traces/")
    ap.add_argument("--out-dir", default="analysis/figures/")
    ap.add_argument("--summary", default="analysis/phase_b2_summary.md")
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
    print(f"  {len(master):,} rows × {len(master.columns)} cols")
    assert os.path.exists(metrics_csv), (
        f"{metrics_csv} missing — run compute_trace_metrics.py first."
    )

    best_t, best_dog, thresholds_t, thresholds_dog = run_b2a(
        master, metrics_csv, args.out_dir, tables_dir
    )
    per_cell_t, per_cell_dog, per_model_t, per_dataset_t = run_b2b(
        master, args.out_dir, tables_dir
    )
    run_b2c(master, args.out_dir, tables_dir, best_t, best_dog)
    run_b2d(master, best_t, args.out_dir, tables_dir)
    run_b2e(master, metrics_csv, best_t, thresholds_t, args.out_dir, tables_dir)
    run_b2f(master, thresholds_t, tables_dir)
    run_b2g(master, best_t, tables_dir)
    run_b2h(tables_dir, args.summary)

    print(f"\n{'=' * 60}")
    print("Phase B.2 complete.")
    print(f"  Figures : {args.out_dir}/plot_{{4_v2,5_v2,6,7}}*.{{png,pdf}}")
    print(f"  Tables  : {tables_dir}/*.csv")
    print(f"  Summary : {args.summary}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
