#!/usr/bin/env python3
"""
phase_b_analysis.py — Reusable analysis helpers for the Phase B notebook.

Loaded by ``analysis_traces_crossmodel.ipynb``. Keeping the heavy lifting
here (not inline in the notebook) means the notebook stays short and the
computations are unit-testable from a CLI.

Expected input: ``results/traces/master.parquet`` built by
``phase_b_build_master.py``.

Sections (matches plan §Phase B):

    plot1_first_reject_cdf  — per-position rejection CDFs for static configs
    plot2_signal_to_al      — binned signal → mean AL curves (DOG + t)
    plot3_cdf_overlap_ks    — cross-cell empirical CDFs + KS matrix
    plot4_buy_signal        — overspend / underspend per signal decile
    plot5_threshold_reco    — per-cell + global threshold grid search
    phase_b_summary_md      — compose the decision-gate markdown
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------- #
# Constants — the 4 static configs, ordered by (num_steps, topk, ndt)        #
# -------------------------------------------------------------------------- #

STATIC_CONFIGS: List[str] = [
    "static_3_1_4",     # num_steps=3, topk=1, ndt=4   — small chain
    "static_7_1_8",     # num_steps=7, topk=1, ndt=8   — deep chain
    "static_7_4_8",     # num_steps=7, topk=4, ndt=8   — tree
    "static_6_10_60",   # num_steps=6, topk=10, ndt=60 — wide tree
]
DYNAMIC_CONFIGS: List[str] = ["v3_dynamic", "v6_dynamic"]
ALL_CONFIGS: List[str] = STATIC_CONFIGS + DYNAMIC_CONFIGS

# Config shapes — (num_steps, topk, ndt). Source of truth: runbook + CLAUDE.md.
CONFIG_SHAPE: Dict[str, Tuple[int, int, int]] = {
    "static_3_1_4":    (3, 1, 4),
    "static_7_1_8":    (7, 1, 8),
    "static_7_4_8":    (7, 4, 8),
    "static_6_10_60":  (6, 10, 60),
}

MODELS: List[str] = ["llama", "qwen"]
DATASETS: List[str] = ["mtbench", "math500", "livecodebench"]

# Signals to analyse
SIGNALS: List[str] = ["top1_prob", "target_top1_prob",
                      "rolling_accept_rate", "DOG", "t"]


# -------------------------------------------------------------------------- #
# Data loading                                                                #
# -------------------------------------------------------------------------- #


def load_master(path: str = "results/traces/master.parquet") -> pd.DataFrame:
    """Load the master dataframe, preferring parquet but falling back to
    pickle if pyarrow/fastparquet aren't installed.  Coerces categoricals
    to str for plotting ease."""
    import os

    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
        except ImportError:
            # parquet exists but engine missing — look for .pkl sibling
            pkl = os.path.splitext(path)[0] + ".pkl"
            if os.path.exists(pkl):
                df = pd.read_pickle(pkl)
            else:
                raise
    else:
        pkl = os.path.splitext(path)[0] + ".pkl"
        if os.path.exists(pkl):
            df = pd.read_pickle(pkl)
        else:
            raise FileNotFoundError(
                f"Neither {path} nor {pkl} exists. "
                "Run analysis/phase_b_build_master.py first."
            )

    for col in ("config", "model", "dataset"):
        df[col] = df[col].astype(str)
    return df


def cell_key(row_or_df) -> str:
    """Short 'model/dataset' tag for labels/legends."""
    if isinstance(row_or_df, pd.Series):
        return f"{row_or_df['model']}/{row_or_df['dataset'][:3]}"
    return row_or_df["model"].astype(str) + "/" + \
           row_or_df["dataset"].str[:3]


# -------------------------------------------------------------------------- #
# Plot 1 — Where static makes mistakes: first-reject position CDFs           #
# -------------------------------------------------------------------------- #


def plot1_first_reject_cdf(master: pd.DataFrame) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """Per (static_config, model, dataset), return a DataFrame of
    (position, cumulative_fraction) rows describing the CDF of
    first_reject_pos for that cell.

    The ``num_steps+1`` position in the output represents "all drafts
    accepted in this step" (first_reject_pos == num_steps).
    """
    out = {}
    static = master[master["config"].isin(STATIC_CONFIGS)]
    for (cfg, mdl, dset), sub in static.groupby(["config", "model", "dataset"]):
        max_pos = CONFIG_SHAPE[cfg][0]  # num_steps (chain) or tree depth
        # We clip at num_steps because positions beyond that mean "all accepted".
        vals = sub["first_reject_pos"].clip(upper=max_pos).to_numpy()
        n = len(vals)
        counts = np.bincount(vals, minlength=max_pos + 1)
        cumfrac = np.cumsum(counts) / n
        df = pd.DataFrame({
            "position": np.arange(max_pos + 1),
            "cum_fraction": cumfrac,
            "count": counts,
            "wasted_tail_fraction": [cumfrac[min(p, max_pos // 2)]
                                     for p in range(max_pos + 1)],
        })
        out[(cfg, mdl, dset)] = df
    return out


def plot1_wasted_tail_table(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """For each (static config, model, dataset), report the fraction of steps
    where first_reject_pos <= num_steps/2 — i.e. the tree's back half was
    almost always wasted."""
    static = master[master["config"].isin(STATIC_CONFIGS)]
    rows = []
    for (cfg, mdl, dset), sub in static.groupby(["config", "model", "dataset"]):
        num_steps = CONFIG_SHAPE[cfg][0]
        half = num_steps // 2
        frac = float((sub["first_reject_pos"] <= half).mean())
        rows.append({
            "config": cfg, "model": mdl, "dataset": dset,
            "num_steps": num_steps,
            "wasted_tail_fraction": frac,
            "mean_al": float(sub["accept_length"].mean()),
            "n_steps": len(sub),
        })
    return pd.DataFrame(rows).sort_values(["model", "dataset", "config"])


# -------------------------------------------------------------------------- #
# Plot 2 — Signal → AL discriminability                                       #
# -------------------------------------------------------------------------- #


def plot2_signal_deciles(
    master: pd.DataFrame,
    signal: str,
    n_bins: int = 10,
    configs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return long-form DataFrame: (model, dataset, config, decile, signal_mid,
    mean_al, ci_lo, ci_hi, n)."""
    if configs is None:
        configs = STATIC_CONFIGS
    sub = master[master["config"].isin(configs)].copy()

    rows = []
    for (mdl, dset, cfg), grp in sub.groupby(["model", "dataset", "config"]):
        if len(grp) < n_bins:
            continue
        # Per-cell quantile binning — matches the plan's "deciles per cell"
        try:
            grp["_bin"] = pd.qcut(grp[signal], n_bins, labels=False,
                                  duplicates="drop")
        except ValueError:
            # Signal has too few distinct values for n_bins — fall back to
            # whatever pandas can give us.
            grp["_bin"] = pd.qcut(grp[signal], n_bins, labels=False,
                                  duplicates="drop", retbins=False)
        for b, bin_grp in grp.groupby("_bin"):
            al = bin_grp["accept_length"].astype(float).to_numpy()
            sem = al.std(ddof=1) / np.sqrt(len(al)) if len(al) > 1 else 0.0
            rows.append({
                "model": mdl, "dataset": dset, "config": cfg,
                "decile": int(b),
                "signal_mid": float(bin_grp[signal].mean()),
                "mean_al": float(al.mean()),
                "ci_lo": float(al.mean() - 1.96 * sem),
                "ci_hi": float(al.mean() + 1.96 * sem),
                "n": int(len(bin_grp)),
            })
    return pd.DataFrame(rows)


def plot2_spearman(
    master: pd.DataFrame,
    signal: str,
    configs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Spearman rank correlation between signal and accept_length per cell."""
    from scipy.stats import spearmanr
    if configs is None:
        configs = STATIC_CONFIGS
    sub = master[master["config"].isin(configs)]
    rows = []
    for (mdl, dset, cfg), grp in sub.groupby(["model", "dataset", "config"]):
        if len(grp) < 50:
            continue
        r, p = spearmanr(grp[signal], grp["accept_length"])
        rows.append({
            "signal": signal, "model": mdl, "dataset": dset,
            "config": cfg, "n": len(grp),
            "spearman_r": r, "p_value": p,
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- #
# Plot 3 — Cross-cell CDFs + KS divergence                                    #
# -------------------------------------------------------------------------- #


def plot3_ks_matrix(
    master: pd.DataFrame,
    signal: str,
    configs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return 6x6 pairwise KS matrix across (model, dataset) cells for one
    signal.  Uses static-only runs by default (unbiased by policy)."""
    from scipy.stats import ks_2samp
    if configs is None:
        configs = STATIC_CONFIGS

    sub = master[master["config"].isin(configs)]
    cells = sorted(
        {(r["model"], r["dataset"]) for _, r in
         sub[["model", "dataset"]].drop_duplicates().iterrows()}
    )
    values = {c: sub[(sub["model"] == c[0]) & (sub["dataset"] == c[1])][signal]
              .to_numpy() for c in cells}

    mat = pd.DataFrame(
        index=[f"{m}/{d}" for m, d in cells],
        columns=[f"{m}/{d}" for m, d in cells],
        dtype=float,
    )
    for i, c_i in enumerate(cells):
        for j, c_j in enumerate(cells):
            if i == j:
                mat.iloc[i, j] = 0.0
            elif i < j:
                ks = ks_2samp(values[c_i], values[c_j]).statistic
                mat.iloc[i, j] = ks
                mat.iloc[j, i] = ks
    return mat


def plot3_ks_summary(
    ks_matrices: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarise KS matrices: per-signal max off-diag, plus within/across
    model & dataset splits."""
    rows = []
    for signal, mat in ks_matrices.items():
        vals = mat.values
        off_diag = vals[~np.eye(vals.shape[0], dtype=bool)]
        max_ks = float(off_diag.max()) if len(off_diag) else 0.0
        mean_ks = float(off_diag.mean()) if len(off_diag) else 0.0

        # Within-model (both cells share model)
        within_mdl_vals, cross_mdl_vals = [], []
        labels = list(mat.columns)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i >= j:
                    continue
                model_a = a.split("/")[0]
                model_b = b.split("/")[0]
                if model_a == model_b:
                    within_mdl_vals.append(mat.iloc[i, j])
                else:
                    cross_mdl_vals.append(mat.iloc[i, j])

        rows.append({
            "signal": signal,
            "max_ks": max_ks,
            "mean_ks": mean_ks,
            "max_ks_within_model": float(np.max(within_mdl_vals))
                                   if within_mdl_vals else 0.0,
            "max_ks_across_model": float(np.max(cross_mdl_vals))
                                   if cross_mdl_vals else 0.0,
            "decision_gate": ("per-model thresholds" if max_ks >= 0.20
                              else "adaptive quantiles" if max_ks >= 0.10
                              else "global threshold OK"),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- #
# Plot 4 — Where dynamic can "buy" throughput                                 #
# -------------------------------------------------------------------------- #


def plot4_buy_signal(
    master: pd.DataFrame,
    signal: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per (model, dataset, config, decile), compute:

        overspend = mean((ndt - accept_length) * [ndt >= 8])
            — how much compute the bigger configs are wasting
        underspend = mean([accept_length == num_steps - 1])
            — fraction of steps where the small chain hit its ceiling

    Both on static runs only.  The "buy zone" is the signal decile where
    overspend and underspend are both notable — dynamic switching helps.
    """
    sub = master[master["config"].isin(STATIC_CONFIGS)].copy()

    rows = []
    for (mdl, dset), grp in sub.groupby(["model", "dataset"]):
        # Per-cell binning by signal
        try:
            grp["_bin"] = pd.qcut(grp[signal], n_bins, labels=False,
                                  duplicates="drop")
        except ValueError:
            continue

        for b, bin_grp in grp.groupby("_bin"):
            for cfg in STATIC_CONFIGS:
                cfg_bin = bin_grp[bin_grp["config"] == cfg]
                if len(cfg_bin) == 0:
                    continue
                ns, _tk, ndt_cfg = CONFIG_SHAPE[cfg]
                overspend = float(
                    ((ndt_cfg - cfg_bin["accept_length"]).clip(lower=0)
                     * (ndt_cfg >= 8)).mean()
                )
                underspend = float(
                    (cfg_bin["accept_length"] >= ns - 1).mean()
                )
                rows.append({
                    "model": mdl, "dataset": dset, "config": cfg,
                    "decile": int(b),
                    "signal_mid": float(bin_grp[signal].mean()),
                    "overspend": overspend,
                    "underspend": underspend,
                    "mean_al": float(cfg_bin["accept_length"].mean()),
                    "n": int(len(cfg_bin)),
                })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- #
# Plot 5 — Threshold recommendations                                          #
# -------------------------------------------------------------------------- #


def plot5_best_config_per_decile(
    master: pd.DataFrame,
    signal: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """For each (model, dataset, decile), find the static config with the
    highest "throughput per compute" (accept_length / draft_tree_size)."""
    sub = master[master["config"].isin(STATIC_CONFIGS)].copy()

    rows = []
    for (mdl, dset), grp in sub.groupby(["model", "dataset"]):
        try:
            grp["_bin"] = pd.qcut(grp[signal], n_bins, labels=False,
                                  duplicates="drop")
        except ValueError:
            continue

        for b, bin_grp in grp.groupby("_bin"):
            decile_rows = {}
            for cfg, cfg_grp in bin_grp.groupby("config"):
                if len(cfg_grp) < 5:
                    continue
                decile_rows[cfg] = {
                    "mean_al": float(cfg_grp["accept_length"].mean()),
                    "yield_per_compute": float(cfg_grp["yield_per_compute"].mean()),
                    "n": len(cfg_grp),
                }
            if not decile_rows:
                continue
            best_cfg = max(decile_rows,
                           key=lambda c: decile_rows[c]["yield_per_compute"])
            rows.append({
                "model": mdl, "dataset": dset,
                "decile": int(b),
                "signal_mid": float(bin_grp[signal].mean()),
                "best_config": best_cfg,
                "best_yield_per_compute":
                    decile_rows[best_cfg]["yield_per_compute"],
                "best_al": decile_rows[best_cfg]["mean_al"],
            })
    return pd.DataFrame(rows)


def plot5_derive_thresholds(
    best_per_decile: pd.DataFrame,
) -> pd.DataFrame:
    """From the best-config-per-decile table, derive per-cell CHEAP/PREMIUM
    thresholds.  CHEAP = where start config becomes best; PREMIUM = where
    max config becomes best."""
    out = []
    for (mdl, dset), grp in best_per_decile.groupby(["model", "dataset"]):
        grp = grp.sort_values("signal_mid")
        # CHEAP = highest signal where start config still wins from below
        cheap_rows = grp[grp["best_config"] == "static_3_1_4"]
        premium_rows = grp[grp["best_config"].isin(
            ["static_6_10_60", "static_7_4_8", "static_7_1_8"]
        )]

        cheap_thr = float(cheap_rows["signal_mid"].max()) \
                    if len(cheap_rows) else float("nan")
        premium_thr = float(premium_rows["signal_mid"].min()) \
                      if len(premium_rows) else float("nan")

        out.append({
            "model": mdl, "dataset": dset,
            "cheap_threshold": cheap_thr,
            "premium_threshold": premium_thr,
            "distinct_best_configs": list(grp["best_config"].unique()),
        })
    return pd.DataFrame(out)


# -------------------------------------------------------------------------- #
# Summary generator                                                           #
# -------------------------------------------------------------------------- #


def phase_b_summary_md(
    ks_summary: pd.DataFrame,
    spearman_dog: pd.DataFrame,
    spearman_t: pd.DataFrame,
    thresholds_t: pd.DataFrame,
    thresholds_dog: pd.DataFrame,
    wasted_tail: pd.DataFrame,
) -> str:
    """Produce the phase_b_summary.md text."""
    lines = [
        "# Phase B — Analysis summary",
        "",
        "Source: 33 Phase A trace cells (Llama full 18/18, Qwen 15/18 — "
        "v3_dynamic missing for Qwen).",
        "",
        "## 1. Cross-cell signal stationarity (KS divergence)",
        "",
        ks_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]

    best_signal_by_median = {
        "DOG": float(spearman_dog["spearman_r"].median()),
        "t":   float(spearman_t["spearman_r"].median()),
    }
    winner = max(best_signal_by_median, key=best_signal_by_median.get)
    lines += [
        "## 2. Signal vs AL discriminability (Spearman rank correlation)",
        "",
        f"- Median Spearman r across cells × configs:",
        f"  - **DOG** = {best_signal_by_median['DOG']:.3f}",
        f"  - **t**   = {best_signal_by_median['t']:.3f}",
        f"- **Winning signal: `{winner}`**",
        "",
        "### Per-cell Spearman r (DOG)",
        "",
        spearman_dog.to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Per-cell Spearman r (t)",
        "",
        spearman_t.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 3. Threshold recommendations per cell",
        "",
        "### Using DOG (V3 signal)",
        "",
        thresholds_dog.to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Using t (V6 signal)",
        "",
        thresholds_t.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 4. Where static makes mistakes (wasted-tail fractions)",
        "",
        wasted_tail.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 5. Phase C scope",
        "",
    ]

    # Heuristic for Phase C
    max_ks = float(ks_summary["max_ks"].max())
    if max_ks < 0.10:
        lines.append(
            "- KS < 0.10 cross-cell → **global thresholds work**. "
            "Phase C can focus on fine-tuning a single set."
        )
    elif max_ks < 0.20:
        lines.append(
            f"- KS = {max_ks:.3f} (borderline) → **adaptive rolling-quantile "
            "thresholds recommended**. V6 already has the window infrastructure; "
            "Phase C could wire policy into it."
        )
    else:
        lines.append(
            f"- KS = {max_ks:.3f} → **per-model thresholds required**. "
            "Phase C should split the V6 constants by model."
        )
    lines.append("")

    return "\n".join(lines)


# ======================================================================== #
# Phase B.2 — follow-up detailed analysis helpers                            #
# ======================================================================== #


def load_time_per_step(metrics_csv_path: str) -> pd.DataFrame:
    """Return per-cell-per-config time-per-decode-step in seconds.

    Reads results/traces/metrics_final.csv (produced by
    analysis/compute_trace_metrics.py) and computes
        time_per_step = elapsed_s / n_steps
    for every (config, model, dataset) row.
    """
    df = pd.read_csv(metrics_csv_path)
    df["time_per_step"] = df["elapsed_s"] / df["num_steps"].clip(lower=1)
    return df[["config", "model", "dataset", "num_steps", "elapsed_s",
               "throughput", "accept_length", "time_per_step"]]


def plot5_best_config_per_decile_v2(
    master: pd.DataFrame,
    signal: str,
    metrics_csv_path: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Fixed Plot 5 ranking.

    For each (cell, decile, config), estimate throughput as
        est_tput = decile_mean_AL / time_per_step_for_config
    (config-specific time_per_step from metrics_final.csv).  Pick the
    config with highest est_tput per decile — that's the "ideal" config
    for that signal bucket on that cell.

    Returns long-form DataFrame with columns:
        model, dataset, decile, signal_mid, best_config,
        best_est_tput, best_al, per_config_est_tput (dict of {cfg: tput})
    """
    sub = master[master["config"].isin(STATIC_CONFIGS)].copy()
    tps = load_time_per_step(metrics_csv_path)
    tps_lookup = {
        (r["config"], r["model"], r["dataset"]): float(r["time_per_step"])
        for _, r in tps.iterrows()
    }

    rows = []
    for (mdl, dset), grp in sub.groupby(["model", "dataset"]):
        try:
            grp = grp.assign(
                _bin=pd.qcut(grp[signal], n_bins, labels=False, duplicates="drop")
            )
        except ValueError:
            continue

        for b, bin_grp in grp.groupby("_bin"):
            per_cfg = {}
            for cfg, cfg_grp in bin_grp.groupby("config"):
                if len(cfg_grp) < 3:
                    continue
                t_per_step = tps_lookup.get((cfg, mdl, dset))
                if t_per_step is None or t_per_step <= 0:
                    continue
                al = float(cfg_grp["accept_length"].mean())
                per_cfg[cfg] = {
                    "mean_al": al,
                    "time_per_step_ms": t_per_step * 1000,
                    "est_tput_tok_s": al / t_per_step,
                    "n": int(len(cfg_grp)),
                }
            if not per_cfg:
                continue
            best = max(per_cfg, key=lambda c: per_cfg[c]["est_tput_tok_s"])
            rows.append({
                "model": mdl, "dataset": dset,
                "decile": int(b),
                "signal_mid": float(bin_grp[signal].mean()),
                "n_steps_in_decile": int(len(bin_grp)),
                "best_config": best,
                "best_est_tput_tok_s": per_cfg[best]["est_tput_tok_s"],
                "best_al": per_cfg[best]["mean_al"],
                # Explode each config's tput as its own column for easy plotting
                **{
                    f"tput_{cfg}": per_cfg.get(cfg, {}).get("est_tput_tok_s",
                                                             float("nan"))
                    for cfg in STATIC_CONFIGS
                },
                **{
                    f"al_{cfg}": per_cfg.get(cfg, {}).get("mean_al",
                                                          float("nan"))
                    for cfg in STATIC_CONFIGS
                },
            })
    return pd.DataFrame(rows)


def plot5_derive_thresholds_v2(
    best_per_decile_v2: pd.DataFrame,
) -> pd.DataFrame:
    """From v2's best-per-decile table, derive CHEAP and PREMIUM thresholds.

    CHEAP = max signal_mid where best_config is static_3_1_4 (the start config)
    PREMIUM = min signal_mid where best_config is a LARGE config (7,4,8 or 6,10,60)
    """
    out = []
    large = {"static_7_4_8", "static_6_10_60"}
    for (mdl, dset), grp in best_per_decile_v2.groupby(["model", "dataset"]):
        grp = grp.sort_values("decile")
        cheap_rows = grp[grp["best_config"] == "static_3_1_4"]
        premium_rows = grp[grp["best_config"].isin(large)]

        out.append({
            "model": mdl, "dataset": dset,
            "cheap_threshold": float(cheap_rows["signal_mid"].max())
                               if len(cheap_rows) else float("nan"),
            "premium_threshold": float(premium_rows["signal_mid"].min())
                                 if len(premium_rows) else float("nan"),
            "distinct_best_configs": sorted(grp["best_config"].unique()),
            "n_cheap_deciles": int(len(cheap_rows)),
            "n_premium_deciles": int(len(premium_rows)),
        })
    return pd.DataFrame(out)


def per_cell_spearman_matrix(
    master: pd.DataFrame,
    signal: str,
) -> pd.DataFrame:
    """24-row pivoted table: one row per (model, dataset, config), spearman r."""
    from scipy.stats import spearmanr
    sub = master[master["config"].isin(STATIC_CONFIGS)]
    rows = []
    for (mdl, dset, cfg), grp in sub.groupby(["model", "dataset", "config"]):
        if len(grp) < 50:
            continue
        r, _ = spearmanr(grp[signal], grp["accept_length"])
        rows.append({
            "model": mdl, "dataset": dset, "config": cfg,
            "spearman_r": float(r), "n": int(len(grp)),
        })
    return pd.DataFrame(rows)


def per_model_aggregates(per_cell: pd.DataFrame, metric: str = "spearman_r") -> pd.DataFrame:
    """Roll the per-cell table to per-model summaries."""
    return (
        per_cell.groupby("model", as_index=False)
        .agg(median=(metric, "median"),
             min=(metric, "min"), max=(metric, "max"),
             n_cells=(metric, "count"))
    )


def per_dataset_aggregates(per_cell: pd.DataFrame, metric: str = "spearman_r") -> pd.DataFrame:
    """Roll the per-cell table to per-dataset summaries (both models combined)."""
    return (
        per_cell.groupby("dataset", as_index=False)
        .agg(median=(metric, "median"),
             min=(metric, "min"), max=(metric, "max"),
             n_cells=(metric, "count"))
    )


def actual_vs_ideal_per_decile(
    master: pd.DataFrame,
    ideal_per_decile: pd.DataFrame,
    signal: str,
    policy_config_name: str,  # "v3_dynamic" or "v6_dynamic"
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare what dynamic policy actually picked per decile vs ideal.

    For each (model, dataset, decile) where the policy was run:
    1. Binning on the SAME signal for both streams (static → ideal,
       dynamic → actual).
    2. Actual config = mode(chosen_topk, chosen_num_steps, chosen_num_draft_tokens)
       in that decile. Translate to a canonical config name if it matches one
       of the 4 statics; else keep as "custom(ts,ns,ndt)".
    3. Ideal config = ideal_per_decile.best_config for the same (model, dataset,
       decile).
    4. Emit match/mismatch.
    """
    dyn = master[master["config"] == policy_config_name].copy()
    rows = []

    # Canonical config lookup by (ns, tk, ndt)
    inv_shape = {v: k for k, v in CONFIG_SHAPE.items()}  # (ns, tk, ndt) -> name

    for (mdl, dset), grp in dyn.groupby(["model", "dataset"]):
        try:
            grp = grp.assign(
                _bin=pd.qcut(grp[signal], n_bins, labels=False, duplicates="drop")
            )
        except ValueError:
            continue
        for b, bin_grp in grp.groupby("_bin"):
            # Mode of chosen_* tuples
            tuples = list(zip(
                bin_grp["chosen_num_steps"].astype(int),
                bin_grp["chosen_topk"].astype(int),
                bin_grp["chosen_num_draft_tokens"].astype(int),
            ))
            if not tuples:
                continue
            # Most common tuple
            from collections import Counter
            mode_tuple = Counter(tuples).most_common(1)[0][0]
            actual_name = inv_shape.get(mode_tuple,
                                        f"custom{mode_tuple}")
            mode_fraction = Counter(tuples).most_common(1)[0][1] / len(tuples)

            ideal_row = ideal_per_decile[
                (ideal_per_decile["model"] == mdl)
                & (ideal_per_decile["dataset"] == dset)
                & (ideal_per_decile["decile"] == int(b))
            ]
            ideal_name = (ideal_row["best_config"].iloc[0]
                          if len(ideal_row) else None)

            rows.append({
                "policy": policy_config_name,
                "model": mdl, "dataset": dset,
                "decile": int(b),
                "signal_mid": float(bin_grp[signal].mean()),
                "n_steps_in_decile": int(len(bin_grp)),
                "actual_config": actual_name,
                "actual_mode_fraction": float(mode_fraction),
                "ideal_config": ideal_name,
                "match": bool(ideal_name is not None
                              and actual_name == ideal_name),
            })
    return pd.DataFrame(rows)


def simulate_policy_throughput(
    master: pd.DataFrame,
    ideal_per_decile: pd.DataFrame,
    metrics_csv_path: str,
    signal: str,
    n_bins: int = 10,
    cheap_threshold: Optional[float] = None,
    premium_threshold: Optional[float] = None,
    use_ideal_switching: bool = False,
) -> pd.DataFrame:
    """Simulate expected throughput of a given threshold policy per cell.

    Policy behaviour per decile:
      if signal_mid < cheap_threshold:
          fire static_3_1_4
      elif signal_mid >= premium_threshold:
          fire static_6_10_60
      else:
          fire static_7_4_8 (interpolation midpoint; practical workhorse)
    If use_ideal_switching=True, override with the decile's best_config.

    For each (model, dataset, decile), expected tokens contributed =
    n_steps_in_decile × mean_AL_of(fired_config, cell, decile).
    Expected time contributed = n_steps_in_decile × time_per_step_for(fired_config, cell).
    Cell throughput = total_tokens / total_time.
    """
    tps = load_time_per_step(metrics_csv_path)
    tps_lookup = {
        (r["config"], r["model"], r["dataset"]): float(r["time_per_step"])
        for _, r in tps.iterrows()
    }

    # Build a (cell, decile, config) → mean_AL lookup from master via binning
    static = master[master["config"].isin(STATIC_CONFIGS)].copy()
    al_map: Dict[Tuple[str, str, int, str], float] = {}
    n_map: Dict[Tuple[str, str, int], int] = {}
    sig_mid_map: Dict[Tuple[str, str, int], float] = {}

    for (mdl, dset), grp in static.groupby(["model", "dataset"]):
        try:
            grp = grp.assign(
                _bin=pd.qcut(grp[signal], n_bins, labels=False, duplicates="drop")
            )
        except ValueError:
            continue
        for b, bin_grp in grp.groupby("_bin"):
            bi = int(b)
            sig_mid_map[(mdl, dset, bi)] = float(bin_grp[signal].mean())
            n_map[(mdl, dset, bi)] = int(len(bin_grp["config"].unique()))
            for cfg, cfg_grp in bin_grp.groupby("config"):
                al_map[(mdl, dset, bi, cfg)] = float(
                    cfg_grp["accept_length"].mean()
                )

    # For each cell and decile, count total trace steps (in master) that land
    # in that decile.  Use the static runs' count as a proxy for dynamic step
    # distribution since we're simulating what COULD have happened.
    count_map: Dict[Tuple[str, str, int], int] = {}
    for (mdl, dset), grp in static.groupby(["model", "dataset"]):
        try:
            grp = grp.assign(
                _bin=pd.qcut(grp[signal], n_bins, labels=False, duplicates="drop")
            )
        except ValueError:
            continue
        for b, bin_grp in grp.groupby("_bin"):
            count_map[(mdl, dset, int(b))] = int(len(bin_grp)
                                                 / len(STATIC_CONFIGS))

    rows = []
    for (mdl, dset), _ in static.groupby(["model", "dataset"]):
        total_tokens = 0.0
        total_time = 0.0
        cell_ideal = ideal_per_decile[
            (ideal_per_decile["model"] == mdl)
            & (ideal_per_decile["dataset"] == dset)
        ]
        for _, ir in cell_ideal.iterrows():
            bi = int(ir["decile"])
            sig_mid = float(ir["signal_mid"])
            n_steps = int(ir["n_steps_in_decile"]) if "n_steps_in_decile" in ir else 0
            if n_steps == 0:
                n_steps = count_map.get((mdl, dset, bi), 0)

            if use_ideal_switching:
                fired = ir["best_config"]
            else:
                if (cheap_threshold is not None
                        and sig_mid < cheap_threshold):
                    fired = "static_3_1_4"
                elif (premium_threshold is not None
                      and sig_mid >= premium_threshold):
                    fired = "static_6_10_60"
                else:
                    fired = "static_7_4_8"

            al = al_map.get((mdl, dset, bi, fired))
            tps_ = tps_lookup.get((fired, mdl, dset))
            if al is None or tps_ is None:
                continue
            total_tokens += n_steps * al
            total_time += n_steps * tps_

        if total_time <= 0:
            continue
        rows.append({
            "model": mdl, "dataset": dset,
            "policy_cheap_threshold": cheap_threshold
                                      if not use_ideal_switching else None,
            "policy_premium_threshold": premium_threshold
                                        if not use_ideal_switching else None,
            "use_ideal_switching": use_ideal_switching,
            "expected_tput_tok_s": total_tokens / total_time,
        })
    return pd.DataFrame(rows)


def adaptive_quantile_feasibility(
    master: pd.DataFrame,
    thresholds_v2: pd.DataFrame,
    signal: str,
) -> pd.DataFrame:
    """Test: is each cell's derived CHEAP/PREMIUM threshold consistent with a
    fixed PERCENTILE of the running signal distribution?"""
    static = master[master["config"].isin(STATIC_CONFIGS)]
    rows = []
    for _, tr in thresholds_v2.iterrows():
        mdl, dset = tr["model"], tr["dataset"]
        cell_vals = static[
            (static["model"] == mdl) & (static["dataset"] == dset)
        ][signal].to_numpy()
        if len(cell_vals) == 0:
            continue
        cheap = tr.get("cheap_threshold")
        premium = tr.get("premium_threshold")
        # Percentile of the threshold within this cell's signal distribution
        def _pct(x):
            import math
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return float("nan")
            return float((cell_vals < x).mean())
        rows.append({
            "model": mdl, "dataset": dset,
            "cheap_threshold": cheap,
            "cheap_percentile": _pct(cheap),
            "premium_threshold": premium,
            "premium_percentile": _pct(premium),
        })
    return pd.DataFrame(rows)


def v3_capability_analysis(
    master: pd.DataFrame,
    ideal_per_decile: pd.DataFrame,
) -> pd.DataFrame:
    """For each v3_dynamic cell, report:
    - Fraction of steps where v3 fired at its max (7, 4, 8)
    - Whether the best static config per this cell is within v3's reach"""
    v3 = master[master["config"] == "v3_dynamic"]
    rows = []
    max_tuple = (7, 4, 8)
    for (mdl, dset), grp in v3.groupby(["model", "dataset"]):
        at_max = ((grp["chosen_num_steps"] == max_tuple[0])
                  & (grp["chosen_topk"] == max_tuple[1])
                  & (grp["chosen_num_draft_tokens"] == max_tuple[2]))
        saturation = float(at_max.mean())
        cell_ideal = ideal_per_decile[
            (ideal_per_decile["model"] == mdl)
            & (ideal_per_decile["dataset"] == dset)
        ]
        ideals = set(cell_ideal["best_config"].unique())
        v3_reach = {"static_3_1_4", "static_7_1_8", "static_7_4_8"}
        reachable = ideals.issubset(v3_reach)
        rows.append({
            "model": mdl, "dataset": dset,
            "n_steps": int(len(grp)),
            "frac_at_max": saturation,
            "ideal_configs_observed": sorted(ideals),
            "all_ideal_within_v3_range": reachable,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test: load master, run plot1 helpers
    master = load_master()
    print(f"Master df: {len(master):,} rows")
    print("\nWasted tail fractions:")
    print(plot1_wasted_tail_table(master).to_string(index=False))
