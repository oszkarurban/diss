"""
Phase 0 calibration. Re-bins the master.parquet trace data by top1_prob
(instead of DOG or t) and finds the per-workload threshold structure for
the single-signal policy in sglang/.../speculative/dynamic_spec.py.

Writes two tables (alongside the existing best_per_decile_{dog,t}_v2.csv):
  analysis/figures/tables/best_per_decile_top1_v2.csv
  analysis/figures/tables/thresholds_top1_v2.csv

Then prints the median cheap_threshold and premium_threshold across the 6
(model, dataset) cells — these are the seed values for TOP1_UNCONFIDENT
and TOP1_DIVISOR in the policy module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "results/traces/master.parquet"
METRICS_CSV = REPO_ROOT / "results/traces/metrics_final.csv"
OUT_DIR = REPO_ROOT / "analysis/figures/tables"
BEST_PER_DECILE_FILE = OUT_DIR / "best_per_decile_top1_v2.csv"
THRESHOLDS_FILE = OUT_DIR / "thresholds_top1_v2.csv"
SIGNAL = "top1_prob"
SMALL_CFG = "static_3_1_4"


def _load_step_rates() -> pd.DataFrame:
    """Per-(model, dataset, config) mean step rate = num_steps / elapsed_s.

    Multiplied with per-step accept_length this yields an estimated per-step
    throughput contribution matching how best_per_decile_{dog,t}_v2.csv scores
    configs.
    """
    m = pd.read_csv(METRICS_CSV)
    m = m[m.config.str.startswith("static_")].copy()
    m["mean_step_rate"] = m["num_steps"] / m["elapsed_s"]
    return m[["config", "model", "dataset", "mean_step_rate"]]


def best_per_decile(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset, decile), pick the static config with highest
    mean estimated per-step throughput on steps in that decile.

    Throughput proxy: ``accept_length * step_rate`` where step_rate is the
    config-level mean (num_steps / elapsed_s). Aggregated across the steps
    in a decile, this is the per-second token yield if you ran that config
    on those steps — same shape as the DOG/t tables.
    """
    static = df[df.config.str.startswith("static_")].copy()
    rates = _load_step_rates()
    static = static.merge(rates, on=["config", "model", "dataset"], how="left")
    static["step_tput"] = static["accept_length"] * static["mean_step_rate"]

    rows = []
    for (model, dataset), g_md in static.groupby(["model", "dataset"], observed=True):
        try:
            _, edges = pd.qcut(g_md[SIGNAL], 10, retbins=True, duplicates="drop")
        except Exception:
            continue
        for di in range(len(edges) - 1):
            lo, hi = edges[di], edges[di + 1]
            mid = 0.5 * (lo + hi)
            mask = (g_md[SIGNAL] >= lo) & (
                g_md[SIGNAL] <= hi if di == len(edges) - 2 else g_md[SIGNAL] < hi
            )
            bin_ = g_md[mask]
            if bin_.empty:
                continue
            per_cfg = bin_.groupby("config", observed=True).agg(
                mean_tput=("step_tput", "mean"),
                mean_al=("accept_length", "mean"),
                n=("accept_length", "size"),
            )
            if per_cfg.empty:
                continue
            best_cfg = per_cfg["mean_tput"].idxmax()
            row = {
                "model": model,
                "dataset": dataset,
                "decile": di,
                "signal_mid": float(mid),
                "n_steps_in_decile": int(bin_.shape[0]),
                "best_config": best_cfg,
                "best_est_tput_tok_s": float(per_cfg.loc[best_cfg, "mean_tput"]),
                "best_al": float(per_cfg.loc[best_cfg, "mean_al"]),
            }
            for cfg in ("static_3_1_4", "static_7_1_8", "static_7_4_8", "static_6_10_60"):
                row[f"tput_{cfg}"] = (
                    float(per_cfg.loc[cfg, "mean_tput"]) if cfg in per_cfg.index else None
                )
                row[f"al_{cfg}"] = (
                    float(per_cfg.loc[cfg, "mean_al"]) if cfg in per_cfg.index else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def thresholds_from_bpd(bpd: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset):
      cheap_threshold   — signal_mid of the highest *contiguous-from-zero*
                          decile where ``static_3_1_4`` wins.  Captures the
                          low-top1 tail regime only, ignoring any
                          high-top1 saturation regime (where a small tree
                          may also win because it hits accept_length ceiling
                          quickly).  Used to seed ``TOP1_UNCONFIDENT``.
      premium_threshold — lowest signal_mid where best_config is a chain
                          (static_3_1_4 / static_7_1_8).  Used to seed
                          ``TOP1_DIVISOR``.
    """
    rows = []
    for (model, dataset), g in bpd.groupby(["model", "dataset"], observed=True):
        g = g.sort_values("decile").reset_index(drop=True)
        cheap_end = 0
        for _, row in g.iterrows():
            if row["best_config"] != SMALL_CFG:
                break
            cheap_end += 1
        cheap_threshold = (
            float(g.iloc[cheap_end - 1]["signal_mid"]) if cheap_end > 0 else None
        )
        premium = g[g.best_config.isin({"static_7_1_8", "static_3_1_4"})]
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "cheap_threshold": cheap_threshold,
                "premium_threshold": (
                    float(premium["signal_mid"].min()) if not premium.empty else None
                ),
                "distinct_best_configs": sorted(g.best_config.unique().tolist()),
                "n_cheap_deciles_contiguous": cheap_end,
                "n_premium_deciles": int(len(premium)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bpd = best_per_decile(df)
    bpd.to_csv(BEST_PER_DECILE_FILE, index=False)
    print(f"wrote {BEST_PER_DECILE_FILE} ({len(bpd)} rows)")

    thr = thresholds_from_bpd(bpd)
    thr.to_csv(THRESHOLDS_FILE, index=False)
    print(f"wrote {THRESHOLDS_FILE} ({len(thr)} rows)\n")

    print("Per-workload top1_prob thresholds:")
    print(thr.to_string(index=False))
    print()

    cheap_vals = thr["cheap_threshold"].dropna()
    premium_vals = thr["premium_threshold"].dropna()
    n_cells_with_cheap = len(cheap_vals)
    n_cells_total = len(thr)

    if n_cells_with_cheap >= 3:
        top1_unconfident = float(cheap_vals.median())
        uncertain_note = (
            f"cross-workload median of contiguous cheap_threshold "
            f"({n_cells_with_cheap}/{n_cells_total} cells carry one)"
        )
    else:
        top1_unconfident = 0.10
        uncertain_note = (
            f"only {n_cells_with_cheap}/{n_cells_total} cells show a low-top1 "
            f"small-chain regime; falling back to 0.10 as a conservative seed"
        )

    if premium_vals.empty:
        top1_divisor = 0.5
        premium_note = "no premium rows; using V3 default 0.5"
    else:
        top1_divisor = min(0.5, float(premium_vals.median()))
        premium_note = (
            f"median={premium_vals.median():.4f}, "
            f"range=[{premium_vals.min():.4f}, {premium_vals.max():.4f}]"
        )

    print(
        "\nSeed constants for sglang/.../dynamic_spec.py:\n"
        f"  TOP1_UNCONFIDENT = {top1_unconfident:.4f}   ({uncertain_note})\n"
        f"  TOP1_DIVISOR     = {top1_divisor:.4f}   ({premium_note})"
    )


if __name__ == "__main__":
    main()
