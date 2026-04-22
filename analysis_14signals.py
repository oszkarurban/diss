#!/usr/bin/env python3
"""
14-Signal Dynamic Speculative Decoding Analysis
================================================
Compares 14-signal dynamic spec runs with 7-signal and vanilla baselines
for both Llama-8B and DeepSeek-8B+LlamaDraft model pairs.

Output: results/analysis_14signals.md
"""

import json
import re
import os
import numpy as np
from collections import Counter

BASE = "/rds/user/ou222/hpc-work/diss"

# =============================================================================
# Configuration
# =============================================================================

SIGNAL_14_KEYS = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "hidden_cosine_sim", "hidden_var", "hidden_max",
    "target_entropy", "target_top1_gap", "target_varentropy",
    "joint_entropy_gate", "entropy_gap_pos", "entropy_gap_neg",
    "rolling_accept_rate",
]

OLD_7_SIGNALS = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "target_entropy", "entropy_gap", "rolling_accept_rate",
]

NEW_SIGNALS = [
    "hidden_cosine_sim", "hidden_var", "hidden_max",
    "target_top1_gap", "target_varentropy",
    "joint_entropy_gate", "entropy_gap_pos", "entropy_gap_neg",
]

CONFIG_KEYS = ["chosen_topk", "chosen_num_steps", "chosen_num_draft_tokens"]

# Data files
DATA_14SIG_LLAMA = os.path.join(BASE, "signal_data_14signals.json")
DATA_14SIG_DEEPSEEK = os.path.join(BASE, "signal_data_14signals_deepseek8b_llamadraft.json")
DATA_7SIG_LLAMA = os.path.join(BASE, "signal_data_dynamic_llama8b.json")
DATA_7SIG_DEEPSEEK = os.path.join(BASE, "signal_data_dynamic_deepseek8b_wrongdraft.json")

# Server logs
LOG_14SIG_LLAMA = os.path.join(BASE, "logs/server_14signals.log")
LOG_14SIG_DEEPSEEK = os.path.join(BASE, "logs/server_14signals_deepseek8b_llamadraft.log")
LOG_7SIG_LLAMA = os.path.join(BASE, "logs/server_dynamic_llama8b.log")
LOG_7SIG_DEEPSEEK = os.path.join(BASE, "logs/server_dynamic_deepseek8b_wrongdraft.log")
LOG_VANILLA_LLAMA = os.path.join(BASE, "logs/server_vanilla_llama8b.log")
LOG_VANILLA_DEEPSEEK = os.path.join(BASE, "logs/server_vanilla_deepseek8b_wrongdraft.log")


# =============================================================================
# Utility functions
# =============================================================================

def load_signal_data(path):
    """Load JSON signal data, flatten all turns into a list of step dicts."""
    with open(path) as f:
        data = json.load(f)
    steps = []
    for turn in data["per_turn_logs"]:
        steps.extend(turn)
    meta = {k: v for k, v in data.items() if k != "per_turn_logs"}
    return steps, meta


def parse_server_log(path):
    """Extract accept_len and throughput from server decode-batch log lines."""
    accept_lens = []
    throughputs = []
    pattern = re.compile(
        r"accept len:\s*([\d.]+).*?gen throughput \(token/s\):\s*([\d.]+)"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                accept_lens.append(float(m.group(1)))
                throughputs.append(float(m.group(2)))
    return np.array(accept_lens), np.array(throughputs)


def pearson_corr(x, y):
    """Pearson correlation coefficient between two arrays."""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # Remove NaN/inf pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((dx * dy).sum() / denom)


def stats_row(arr):
    """Return (N, min, max, mean, std, median) for an array."""
    a = np.array(arr, dtype=np.float64)
    return len(a), a.min(), a.max(), a.mean(), a.std(), np.median(a)


def fmt(v, decimals=4):
    """Format a number."""
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1e6):
        return str(int(v))
    return f"{v:.{decimals}f}"


# =============================================================================
# Main analysis
# =============================================================================

def main():
    out_lines = []

    def w(line=""):
        out_lines.append(line)

    # =========================================================================
    # Load all data
    # =========================================================================
    steps_14_llama, meta_14_llama = load_signal_data(DATA_14SIG_LLAMA)
    steps_14_ds, meta_14_ds = load_signal_data(DATA_14SIG_DEEPSEEK)

    # Server logs
    al_14_llama, tp_14_llama = parse_server_log(LOG_14SIG_LLAMA)
    al_14_ds, tp_14_ds = parse_server_log(LOG_14SIG_DEEPSEEK)
    al_7_llama, tp_7_llama = parse_server_log(LOG_7SIG_LLAMA)
    al_7_ds, tp_7_ds = parse_server_log(LOG_7SIG_DEEPSEEK)
    al_v_llama, tp_v_llama = parse_server_log(LOG_VANILLA_LLAMA)
    al_v_ds, tp_v_ds = parse_server_log(LOG_VANILLA_DEEPSEEK)

    # Steady-state: exclude first batch per request (cold start)
    # For 5 questions x 2 turns = 10 turns, first batch of each has ~1-2 tok/s
    # We detect cold-start as throughput < 10 tok/s
    def steady(tp, al):
        mask = tp > 10.0
        return al[mask], tp[mask]

    al_14_llama_ss, tp_14_llama_ss = steady(tp_14_llama, al_14_llama)
    al_14_ds_ss, tp_14_ds_ss = steady(tp_14_ds, al_14_ds)
    al_7_llama_ss, tp_7_llama_ss = steady(tp_7_llama, al_7_llama)
    al_7_ds_ss, tp_7_ds_ss = steady(tp_7_ds, al_7_ds)
    al_v_llama_ss, tp_v_llama_ss = steady(tp_v_llama, al_v_llama)
    al_v_ds_ss, tp_v_ds_ss = steady(tp_v_ds, al_v_ds)

    # =========================================================================
    # Section 1: Server-level metrics comparison
    # =========================================================================
    w("# 14-Signal Dynamic Speculative Decoding: Comprehensive Analysis")
    w()
    w("**Setup**: 5 MT-Bench questions, 2 turns each (10 turns total), bs=1, temperature=0, A100 GPU")
    w()
    w("**14-signal runs** add 7 new signals on top of the original 7: hidden_cosine_sim, hidden_var, hidden_max, target_top1_gap, target_varentropy, joint_entropy_gate, entropy_gap_pos, entropy_gap_neg (entropy_gap split into positive/negative components).")
    w()

    w("## Section 1: Accept Length and Throughput Comparison")
    w()

    w("### 1.1 Accept Length (all batches)")
    w()
    w("| Config | N batches | Mean | Std | Min | Max | Median |")
    w("|--------|-----------|------|-----|-----|-----|--------|")
    for label, al in [
        ("Vanilla Llama-8B", al_v_llama),
        ("7-sig Dynamic Llama-8B", al_7_llama),
        ("14-sig Dynamic Llama-8B", al_14_llama),
        ("Vanilla DeepSeek-8B", al_v_ds),
        ("7-sig Dynamic DeepSeek-8B", al_7_ds),
        ("14-sig Dynamic DeepSeek-8B", al_14_ds),
    ]:
        n, mn, mx, mean, std, med = stats_row(al)
        w(f"| {label} | {n} | {mean:.4f} | {std:.4f} | {mn:.4f} | {mx:.4f} | {med:.4f} |")
    w()

    w("### 1.2 Accept Length (steady-state, excluding cold-start batches)")
    w()
    w("| Config | N batches | Mean | Std | Median |")
    w("|--------|-----------|------|-----|--------|")
    for label, al in [
        ("Vanilla Llama-8B", al_v_llama_ss),
        ("7-sig Dynamic Llama-8B", al_7_llama_ss),
        ("14-sig Dynamic Llama-8B", al_14_llama_ss),
        ("Vanilla DeepSeek-8B", al_v_ds_ss),
        ("7-sig Dynamic DeepSeek-8B", al_7_ds_ss),
        ("14-sig Dynamic DeepSeek-8B", al_14_ds_ss),
    ]:
        n, mn, mx, mean, std, med = stats_row(al)
        w(f"| {label} | {n} | {mean:.4f} | {std:.4f} | {med:.4f} |")
    w()

    w("### 1.3 Throughput (steady-state tok/s)")
    w()
    w("| Config | N batches | Mean | Std | Median | Max |")
    w("|--------|-----------|------|-----|--------|-----|")
    for label, tp in [
        ("Vanilla Llama-8B", tp_v_llama_ss),
        ("7-sig Dynamic Llama-8B", tp_7_llama_ss),
        ("14-sig Dynamic Llama-8B", tp_14_llama_ss),
        ("Vanilla DeepSeek-8B", tp_v_ds_ss),
        ("7-sig Dynamic DeepSeek-8B", tp_7_ds_ss),
        ("14-sig Dynamic DeepSeek-8B", tp_14_ds_ss),
    ]:
        n, mn, mx, mean, std, med = stats_row(tp)
        w(f"| {label} | {n} | {mean:.2f} | {std:.2f} | {med:.2f} | {mx:.2f} |")
    w()

    w("### 1.4 Delta Summary")
    w()

    def delta_str(base, test, label):
        pct = (test - base) / base * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    w("**Llama-8B (matched draft model)**:")
    w()
    w("| Metric | Vanilla | 7-sig Dynamic | 14-sig Dynamic | 7sig vs Vanilla | 14sig vs Vanilla | 14sig vs 7sig |")
    w("|--------|---------|---------------|----------------|-----------------|------------------|---------------|")
    for metric_name, v_arr, s7_arr, s14_arr in [
        ("Accept len (ss)", al_v_llama_ss, al_7_llama_ss, al_14_llama_ss),
        ("Throughput (ss)", tp_v_llama_ss, tp_7_llama_ss, tp_14_llama_ss),
    ]:
        v, s7, s14 = v_arr.mean(), s7_arr.mean(), s14_arr.mean()
        w(f"| {metric_name} | {v:.4f} | {s7:.4f} | {s14:.4f} | {delta_str(v, s7, '')} | {delta_str(v, s14, '')} | {delta_str(s7, s14, '')} |")
    w(f"| Decode steps | {len(al_v_llama)} | {len(al_7_llama)} | {len(al_14_llama)} | — | — | — |")
    w(f"| Signal steps | 0 | {meta_14_llama.get('num_steps_total', 'N/A')} (7sig: from prev analysis) | {meta_14_llama['num_steps_total']} | — | — | — |")
    w()

    w("**DeepSeek-8B + LlamaDraft (mismatched draft model)**:")
    w()
    w("| Metric | Vanilla | 7-sig Dynamic | 14-sig Dynamic | 7sig vs Vanilla | 14sig vs Vanilla | 14sig vs 7sig |")
    w("|--------|---------|---------------|----------------|-----------------|------------------|---------------|")
    for metric_name, v_arr, s7_arr, s14_arr in [
        ("Accept len (ss)", al_v_ds_ss, al_7_ds_ss, al_14_ds_ss),
        ("Throughput (ss)", tp_v_ds_ss, tp_7_ds_ss, tp_14_ds_ss),
    ]:
        v, s7, s14 = v_arr.mean(), s7_arr.mean(), s14_arr.mean()
        w(f"| {metric_name} | {v:.4f} | {s7:.4f} | {s14:.4f} | {delta_str(v, s7, '')} | {delta_str(v, s14, '')} | {delta_str(s7, s14, '')} |")
    w(f"| Decode steps | {len(al_v_ds)} | {len(al_7_ds)} | {len(al_14_ds)} | — | — | — |")
    w()

    # =========================================================================
    # Section 2: Per-signal correlation rankings
    # =========================================================================
    w("## Section 2: Per-Signal Correlation Rankings")
    w()

    for ds_label, steps, sig_keys in [
        ("14-sig Llama-8B", steps_14_llama, SIGNAL_14_KEYS),
        ("14-sig DeepSeek-8B+LlamaDraft", steps_14_ds, SIGNAL_14_KEYS),
    ]:
        w(f"### 2.1 {ds_label} — Signal Statistics ({len(steps)} steps)")
        w()
        w("| Signal | N | Min | Max | Mean | Std | Median |")
        w("|--------|---|-----|-----|------|-----|--------|")
        for sig in sig_keys:
            vals = [s[sig] for s in steps if sig in s]
            if not vals:
                w(f"| {sig} | 0 | — | — | — | — | — |")
                continue
            n, mn, mx, mean, std, med = stats_row(vals)
            w(f"| {sig} | {n} | {mn:.4f} | {mx:.4f} | {mean:.4f} | {std:.4f} | {med:.4f} |")
        w()

        # Correlation with rolling_accept_rate
        rar_vals = np.array([s["rolling_accept_rate"] for s in steps], dtype=np.float64)
        conf_vals = np.array([s["confidence"] for s in steps], dtype=np.float64)

        corr_rar = []
        corr_conf = []
        for sig in sig_keys:
            if sig == "rolling_accept_rate":
                corr_rar.append((sig, 1.0))  # trivially 1
                corr_conf.append((sig, pearson_corr([s["rolling_accept_rate"] for s in steps], conf_vals)))
                continue
            vals = np.array([s.get(sig, float("nan")) for s in steps], dtype=np.float64)
            corr_rar.append((sig, pearson_corr(vals, rar_vals)))
            corr_conf.append((sig, pearson_corr(vals, conf_vals)))

        # Sort by |corr with RAR|
        corr_rar_sorted = sorted(corr_rar, key=lambda x: abs(x[1]), reverse=True)

        w(f"### 2.2 {ds_label} — Correlation with rolling_accept_rate (ranked by |r|)")
        w()
        w("| Rank | Signal | r(signal, RAR) | |r| | Signal type |")
        w("|------|--------|----------------|-----|-------------|")
        for i, (sig, r) in enumerate(corr_rar_sorted, 1):
            sig_type = "NEW" if sig in NEW_SIGNALS else ("OLD" if sig in OLD_7_SIGNALS else "—")
            if sig == "rolling_accept_rate":
                sig_type = "SELF"
            w(f"| {i} | {sig} | {r:+.4f} | {abs(r):.4f} | {sig_type} |")
        w()

        # Sort by |corr with confidence|
        corr_conf_sorted = sorted(corr_conf, key=lambda x: abs(x[1]), reverse=True)

        w(f"### 2.3 {ds_label} — Correlation with confidence (ranked by |r|)")
        w()
        w("| Rank | Signal | r(signal, confidence) | |r| | Signal type |")
        w("|------|--------|----------------------|-----|-------------|")
        for i, (sig, r) in enumerate(corr_conf_sorted, 1):
            sig_type = "NEW" if sig in NEW_SIGNALS else ("OLD" if sig in OLD_7_SIGNALS else "—")
            if sig == "rolling_accept_rate":
                sig_type = "OLD"
            w(f"| {i} | {sig} | {r:+.4f} | {abs(r):.4f} | {sig_type} |")
        w()

        # Config distribution
        w(f"### 2.4 {ds_label} — Config Distribution")
        w()
        for ck in CONFIG_KEYS:
            vals = [s[ck] for s in steps]
            ctr = Counter(vals)
            w(f"**{ck}**:")
            w("| Value | Count | Pct |")
            w("|-------|-------|-----|")
            for v in sorted(ctr.keys()):
                w(f"| {v} | {ctr[v]} | {ctr[v]/len(vals)*100:.1f}% |")
            w()

        # Config tuple distribution
        tuples = [(s["chosen_topk"], s["chosen_num_steps"], s["chosen_num_draft_tokens"]) for s in steps]
        tuple_ctr = Counter(tuples)
        w(f"**Config tuple (topk, steps, ndt) distribution**:")
        w("| (topk, steps, ndt) | Count | Pct |")
        w("|--------------------|-------|-----|")
        for t, c in tuple_ctr.most_common():
            w(f"| {t} | {c} | {c/len(tuples)*100:.1f}% |")
        w()

        # Confidence distribution
        w(f"### 2.5 {ds_label} — Confidence Distribution")
        w()
        w("| Bucket | Count | Pct |")
        w("|--------|-------|-----|")
        for lo in np.arange(0, 1.0, 0.1):
            hi = lo + 0.1
            count = sum(1 for s in steps if lo <= s["confidence"] < hi)
            w(f"| [{lo:.1f}, {hi:.1f}) | {count} | {count/len(steps)*100:.1f}% |")
        w()

    # =========================================================================
    # Section 3: New signals assessment
    # =========================================================================
    w("## Section 3: New Signals Assessment")
    w()
    w("This section evaluates whether the 7 new signals (hidden_cosine_sim, hidden_var, hidden_max, target_top1_gap, target_varentropy, joint_entropy_gate, entropy_gap_pos, entropy_gap_neg) improve prediction of acceptance rate compared to the original 7 signals.")
    w()

    # Build comparison tables for both datasets
    for ds_label, steps in [
        ("Llama-8B", steps_14_llama),
        ("DeepSeek-8B+LlamaDraft", steps_14_ds),
    ]:
        rar_vals = np.array([s["rolling_accept_rate"] for s in steps], dtype=np.float64)

        # Compute correlations for all non-RAR signals
        all_corrs = {}
        for sig in SIGNAL_14_KEYS:
            if sig == "rolling_accept_rate":
                continue
            vals = np.array([s.get(sig, float("nan")) for s in steps], dtype=np.float64)
            all_corrs[sig] = pearson_corr(vals, rar_vals)

        # Rank
        ranked = sorted(all_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

        w(f"### 3.1 {ds_label} — Signal Ranking by |r(signal, RAR)| (excluding RAR itself)")
        w()
        w("| Rank | Signal | r | |r| | Old/New | Top-3? |")
        w("|------|--------|---|-----|---------|--------|")
        for i, (sig, r) in enumerate(ranked, 1):
            old_new = "NEW" if sig in NEW_SIGNALS else "OLD"
            top3 = "YES" if i <= 3 else ""
            w(f"| {i} | {sig} | {r:+.4f} | {abs(r):.4f} | {old_new} | {top3} |")
        w()

        # Count new signals in top half
        n_total = len(ranked)
        top_half = ranked[:n_total // 2]
        new_in_top_half = sum(1 for sig, _ in top_half if sig in NEW_SIGNALS)
        old_in_top_half = sum(1 for sig, _ in top_half if sig not in NEW_SIGNALS)

        w(f"**New signals in top half**: {new_in_top_half}/{len(NEW_SIGNALS)} new signals rank in top {n_total//2}")
        w(f"**Old signals in top half**: {old_in_top_half}/{len(OLD_7_SIGNALS)-1} old signals rank in top {n_total//2}")
        w()

        # Average |r| for old vs new
        old_avg_r = np.mean([abs(all_corrs[s]) for s in OLD_7_SIGNALS if s != "rolling_accept_rate" and s in all_corrs])
        new_avg_r = np.mean([abs(all_corrs[s]) for s in NEW_SIGNALS if s in all_corrs])
        w(f"**Average |r| for old signals**: {old_avg_r:.4f}")
        w(f"**Average |r| for new signals**: {new_avg_r:.4f}")
        w()

    # Cross-dataset comparison of 7-sig vs 14-sig correlations
    w("### 3.2 Comparison with Previous 7-Signal Analysis")
    w()
    w("From the previous analysis (analysis_4configs.md), the 7-signal correlations with RAR were:")
    w()
    w("| Signal | 7sig Llama r(RAR) | 7sig DeepSeek r(RAR) | 14sig Llama r(RAR) | 14sig DeepSeek r(RAR) |")
    w("|--------|-------------------|----------------------|--------------------|-----------------------|")

    # 7-signal correlations from the previous analysis
    old_corrs_llama = {
        "draft_entropy": -0.1629,
        "top1_prob": 0.1669,
        "top1_minus_top2": 0.1563,
        "hidden_norm": -0.0458,
        "target_entropy": -0.3724,
        "entropy_gap": 0.0496,
    }
    old_corrs_ds = {
        "draft_entropy": -0.1696,
        "top1_prob": 0.2503,
        "top1_minus_top2": 0.2552,
        "hidden_norm": -0.0689,
        "target_entropy": -0.3827,
        "entropy_gap": 0.0100,
    }

    # 14-signal correlations
    rar_llama = np.array([s["rolling_accept_rate"] for s in steps_14_llama], dtype=np.float64)
    rar_ds = np.array([s["rolling_accept_rate"] for s in steps_14_ds], dtype=np.float64)
    new_corrs_llama = {}
    new_corrs_ds = {}
    for sig in SIGNAL_14_KEYS:
        if sig == "rolling_accept_rate":
            continue
        vals_l = np.array([s.get(sig, float("nan")) for s in steps_14_llama], dtype=np.float64)
        vals_d = np.array([s.get(sig, float("nan")) for s in steps_14_ds], dtype=np.float64)
        new_corrs_llama[sig] = pearson_corr(vals_l, rar_llama)
        new_corrs_ds[sig] = pearson_corr(vals_d, rar_ds)

    # Print overlapping old signals
    # Note: entropy_gap was split into entropy_gap_pos and entropy_gap_neg in 14-sig
    for sig in ["draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm", "target_entropy"]:
        old_l = old_corrs_llama.get(sig, "—")
        old_d = old_corrs_ds.get(sig, "—")
        new_l = new_corrs_llama.get(sig, "—")
        new_d = new_corrs_ds.get(sig, "—")
        old_l_str = f"{old_l:+.4f}" if isinstance(old_l, float) else old_l
        old_d_str = f"{old_d:+.4f}" if isinstance(old_d, float) else old_d
        new_l_str = f"{new_l:+.4f}" if isinstance(new_l, float) else new_l
        new_d_str = f"{new_d:+.4f}" if isinstance(new_d, float) else new_d
        w(f"| {sig} | {old_l_str} | {old_d_str} | {new_l_str} | {new_d_str} |")
    # entropy_gap -> entropy_gap_pos + entropy_gap_neg
    w(f"| entropy_gap (old) | {old_corrs_llama['entropy_gap']:+.4f} | {old_corrs_ds['entropy_gap']:+.4f} | split below | split below |")
    w(f"| entropy_gap_pos (new) | — | — | {new_corrs_llama.get('entropy_gap_pos', float('nan')):+.4f} | {new_corrs_ds.get('entropy_gap_pos', float('nan')):+.4f} |")
    w(f"| entropy_gap_neg (new) | — | — | {new_corrs_llama.get('entropy_gap_neg', float('nan')):+.4f} | {new_corrs_ds.get('entropy_gap_neg', float('nan')):+.4f} |")
    w()

    w("### 3.3 Key Findings on New Signals")
    w()

    # Analyze which new signals are most useful
    for ds_label, corrs in [("Llama-8B", new_corrs_llama), ("DeepSeek-8B", new_corrs_ds)]:
        ranked_new = sorted(
            [(s, corrs[s]) for s in NEW_SIGNALS if s in corrs],
            key=lambda x: abs(x[1]), reverse=True,
        )
        ranked_all = sorted(
            [(s, corrs[s]) for s in corrs],
            key=lambda x: abs(x[1]), reverse=True,
        )
        # Find rank of each new signal in overall ranking
        rank_map = {s: i+1 for i, (s, _) in enumerate(ranked_all)}

        w(f"**{ds_label}**: New signals ranked by |r(signal, RAR)|:")
        w()
        for sig, r in ranked_new:
            w(f"- {sig}: r={r:+.4f} (overall rank {rank_map[sig]}/{len(ranked_all)})")
        w()

    # =========================================================================
    # Section 4: Recommended signal weights
    # =========================================================================
    w("## Section 4: Recommended Signal Weights Based on Correlation Strength")
    w()
    w("The current policy uses equal weights (1/N for N signals). Below we propose weights proportional to |r(signal, RAR)|, normalized to sum to 1.")
    w()

    for ds_label, corrs in [
        ("Llama-8B", new_corrs_llama),
        ("DeepSeek-8B+LlamaDraft", new_corrs_ds),
    ]:
        # Exclude RAR from weights (it's what we're predicting)
        sigs = [s for s in SIGNAL_14_KEYS if s != "rolling_accept_rate" and s in corrs]
        abs_corrs = {s: abs(corrs[s]) for s in sigs}
        total = sum(abs_corrs.values())
        if total == 0:
            total = 1.0

        w(f"### 4.1 {ds_label} — Proposed Weights")
        w()
        w("| Signal | |r(RAR)| | Equal weight (1/13) | Proposed weight | Ratio vs equal |")
        w("|--------|---------|---------------------|-----------------|----------------|")
        equal_w = 1.0 / len(sigs)
        for sig in sorted(sigs, key=lambda s: abs_corrs[s], reverse=True):
            prop_w = abs_corrs[sig] / total
            ratio = prop_w / equal_w
            w(f"| {sig} | {abs_corrs[sig]:.4f} | {equal_w:.4f} | {prop_w:.4f} | {ratio:.2f}x |")
        w()

        # Tier analysis
        high = [(s, abs_corrs[s]) for s in sigs if abs_corrs[s] >= 0.2]
        medium = [(s, abs_corrs[s]) for s in sigs if 0.1 <= abs_corrs[s] < 0.2]
        low = [(s, abs_corrs[s]) for s in sigs if abs_corrs[s] < 0.1]

        w(f"**Tier breakdown for {ds_label}**:")
        w(f"- HIGH (|r| >= 0.2): {', '.join(s for s, _ in high) if high else 'none'}")
        w(f"- MEDIUM (0.1 <= |r| < 0.2): {', '.join(s for s, _ in medium) if medium else 'none'}")
        w(f"- LOW (|r| < 0.1): {', '.join(s for s, _ in low) if low else 'none'}")
        w()

    # Cross-model consensus
    w("### 4.2 Cross-Model Consensus Weights")
    w()
    w("For a model-agnostic policy, we average |r| across both datasets:")
    w()
    sigs_common = [s for s in SIGNAL_14_KEYS if s != "rolling_accept_rate" and s in new_corrs_llama and s in new_corrs_ds]
    avg_abs = {s: (abs(new_corrs_llama[s]) + abs(new_corrs_ds[s])) / 2 for s in sigs_common}
    total_avg = sum(avg_abs.values())
    if total_avg == 0:
        total_avg = 1.0

    w("| Signal | |r| Llama | |r| DeepSeek | Avg |r| | Consensus weight | Old/New |")
    w("|--------|-----------|--------------|---------|------------------|---------|")
    for sig in sorted(sigs_common, key=lambda s: avg_abs[s], reverse=True):
        cw = avg_abs[sig] / total_avg
        old_new = "NEW" if sig in NEW_SIGNALS else "OLD"
        w(f"| {sig} | {abs(new_corrs_llama[sig]):.4f} | {abs(new_corrs_ds[sig]):.4f} | {avg_abs[sig]:.4f} | {cw:.4f} | {old_new} |")
    w()

    # =========================================================================
    # Section 5: Signal dilution analysis
    # =========================================================================
    w("## Section 5: Signal Dilution Analysis")
    w()
    w("This section investigates why the 14-signal policy may achieve slightly different accept_len than the 7-signal policy.")
    w()

    # 5.1 Compare config distributions
    w("### 5.1 Config Distribution Comparison (Llama-8B)")
    w()
    w("| Config Param | 7-sig Mean (from prev analysis) | 14-sig Mean |")
    w("|-------------|--------------------------------|-------------|")
    # 7-sig means from previous analysis
    prev_7sig_llama = {"chosen_topk": 2.0587, "chosen_num_steps": 3.6368, "chosen_num_draft_tokens": 5.3189}
    for ck in CONFIG_KEYS:
        vals_14 = np.array([s[ck] for s in steps_14_llama], dtype=np.float64)
        prev = prev_7sig_llama.get(ck, "—")
        w(f"| {ck} | {prev:.4f} | {vals_14.mean():.4f} |")
    w()

    w("### 5.2 Config Distribution Comparison (DeepSeek-8B)")
    w()
    w("| Config Param | 7-sig Mean (from prev analysis) | 14-sig Mean |")
    w("|-------------|--------------------------------|-------------|")
    prev_7sig_ds = {"chosen_topk": 1.1514, "chosen_num_steps": 2.7800, "chosen_num_draft_tokens": 3.8738}
    for ck in CONFIG_KEYS:
        vals_14 = np.array([s[ck] for s in steps_14_ds], dtype=np.float64)
        prev = prev_7sig_ds.get(ck, "—")
        w(f"| {ck} | {prev:.4f} | {vals_14.mean():.4f} |")
    w()

    # 5.3 Confidence comparison
    w("### 5.3 Confidence Distribution Comparison")
    w()
    conf_14_llama = np.array([s["confidence"] for s in steps_14_llama], dtype=np.float64)
    conf_14_ds = np.array([s["confidence"] for s in steps_14_ds], dtype=np.float64)

    w("| Metric | 7-sig Llama | 14-sig Llama | 7-sig DeepSeek | 14-sig DeepSeek |")
    w("|--------|-------------|-------------|----------------|-----------------|")
    # 7-sig values from previous analysis
    w(f"| Confidence mean | 0.6671 | {conf_14_llama.mean():.4f} | 0.4452 | {conf_14_ds.mean():.4f} |")
    w(f"| Confidence std | 0.1239 | {conf_14_llama.std():.4f} | 0.1218 | {conf_14_ds.std():.4f} |")
    w(f"| Confidence median | 0.6889 | {np.median(conf_14_llama):.4f} | 0.4317 | {np.median(conf_14_ds):.4f} |")
    w()

    # 5.4 Signal dilution mechanism
    w("### 5.4 Signal Dilution Mechanism")
    w()
    w("When expanding from 7 to 14 signals with equal weights, each signal's influence on the")
    w("confidence score is reduced from 1/7 (0.143) to 1/14 (0.071) — a 50% reduction per signal.")
    w("This has several consequences:")
    w()
    w("1. **Reduced sensitivity to strong predictors**: target_entropy (the strongest predictor of RAR")
    w("   in the 7-signal analysis) now has half the weight. If new signals are weaker predictors,")
    w("   the overall confidence becomes a noisier estimate of true acceptance probability.")
    w()
    w("2. **Regression to the mean**: Adding weak signals (those with |r| < 0.1) pulls confidence")
    w("   toward the center of its range, reducing the system's ability to discriminate between")
    w("   easy (high-acceptance) and hard (low-acceptance) regions.")
    w()

    # Quantify: what fraction of total |r| comes from new signals?
    for ds_label, corrs in [("Llama-8B", new_corrs_llama), ("DeepSeek-8B", new_corrs_ds)]:
        sigs_no_rar = [s for s in SIGNAL_14_KEYS if s != "rolling_accept_rate"]
        old_sigs = [s for s in sigs_no_rar if s not in NEW_SIGNALS]
        new_sigs = [s for s in sigs_no_rar if s in NEW_SIGNALS]
        old_total_r = sum(abs(corrs.get(s, 0)) for s in old_sigs)
        new_total_r = sum(abs(corrs.get(s, 0)) for s in new_sigs)
        all_total_r = old_total_r + new_total_r

        w(f"**{ds_label}**:")
        w(f"- Old signals (5 excl RAR, entropy_gap): total |r| = {old_total_r:.4f} ({old_total_r/all_total_r*100:.1f}% of total)")
        w(f"- New signals (8): total |r| = {new_total_r:.4f} ({new_total_r/all_total_r*100:.1f}% of total)")
        w(f"- But equal weighting gives new signals {len(new_sigs)}/{len(sigs_no_rar)} = {len(new_sigs)/len(sigs_no_rar)*100:.0f}% of total weight")
        w(f"- Information-weight mismatch: new signals get {len(new_sigs)/len(sigs_no_rar)*100:.0f}% of weight but contribute only {new_total_r/all_total_r*100:.1f}% of predictive power")
        w()

    w("### 5.5 Recommendations to Fix Signal Dilution")
    w()
    w("1. **Use correlation-weighted signals**: Replace equal 1/N weights with weights proportional")
    w("   to |r(signal, RAR)| as shown in Section 4. This preserves the influence of strong")
    w("   predictors while allowing new signals to contribute where informative.")
    w()
    w("2. **Prune uninformative signals**: Remove signals with |r| < 0.05 across both datasets.")
    w("   Candidates for removal:")

    # Find signals weak in both
    for sig in SIGNAL_14_KEYS:
        if sig == "rolling_accept_rate":
            continue
        r_l = abs(new_corrs_llama.get(sig, 0))
        r_d = abs(new_corrs_ds.get(sig, 0))
        if r_l < 0.05 and r_d < 0.05:
            w(f"   - {sig} (|r|={r_l:.4f} Llama, |r|={r_d:.4f} DeepSeek)")
    w()
    w("3. **Use a learned weighting**: Train a small linear model (or logistic regression) to predict")
    w("   RAR from the 14 signals. The learned coefficients directly give optimal weights. This can")
    w("   be done offline on collected signal data without modifying the serving infrastructure.")
    w()
    w("4. **Consider feature selection**: Use forward stepwise selection starting from the strongest")
    w("   signal (target_entropy) and adding signals only if they improve cross-validated R-squared.")
    w("   This would likely yield a 4-6 signal subset that outperforms both the 7 and 14 signal sets.")
    w()

    # =========================================================================
    # Section 6: Inter-signal correlation matrix (compact)
    # =========================================================================
    w("## Section 6: Inter-Signal Correlation Highlights")
    w()
    w("Highly correlated signals are redundant. Below are signal pairs with |r| > 0.7:")
    w()

    for ds_label, steps in [
        ("Llama-8B", steps_14_llama),
        ("DeepSeek-8B+LlamaDraft", steps_14_ds),
    ]:
        sigs = [s for s in SIGNAL_14_KEYS if s != "rolling_accept_rate"]
        sig_arrays = {}
        for sig in sigs:
            sig_arrays[sig] = np.array([s.get(sig, float("nan")) for s in steps], dtype=np.float64)

        w(f"**{ds_label}** (|r| > 0.7):")
        w()
        w("| Signal A | Signal B | r |")
        w("|----------|----------|---|")
        pairs_found = False
        for i, s1 in enumerate(sigs):
            for s2 in sigs[i+1:]:
                r = pearson_corr(sig_arrays[s1], sig_arrays[s2])
                if abs(r) > 0.7:
                    w(f"| {s1} | {s2} | {r:+.4f} |")
                    pairs_found = True
        if not pairs_found:
            w("| (none) | — | — |")
        w()

    # =========================================================================
    # Summary
    # =========================================================================
    w("## Summary")
    w()
    w("### Key Findings")
    w()

    # Determine if any new signal is in top 3 for either dataset
    for ds_label, corrs in [("Llama-8B", new_corrs_llama), ("DeepSeek-8B", new_corrs_ds)]:
        ranked = sorted(
            [(s, abs(corrs[s])) for s in corrs],
            key=lambda x: x[1], reverse=True,
        )
        top3 = [s for s, _ in ranked[:3]]
        new_in_top3 = [s for s in top3 if s in NEW_SIGNALS]
        w(f"- **{ds_label}** top-3 predictors of RAR: {', '.join(top3)}")
        if new_in_top3:
            w(f"  - NEW signal(s) in top 3: {', '.join(new_in_top3)}")
        else:
            w(f"  - No new signals in top 3")

    w()
    w("### Accept Length Summary")
    w()
    w("| Model Pair | Vanilla | 7-sig Dynamic | 14-sig Dynamic |")
    w("|------------|---------|---------------|----------------|")
    w(f"| Llama-8B (ss) | {al_v_llama_ss.mean():.4f} | {al_7_llama_ss.mean():.4f} | {al_14_llama_ss.mean():.4f} |")
    w(f"| DeepSeek-8B (ss) | {al_v_ds_ss.mean():.4f} | {al_7_ds_ss.mean():.4f} | {al_14_ds_ss.mean():.4f} |")
    w()

    w("### Verdict on 14-Signal Expansion")
    w()
    w("The expansion from 7 to 14 signals introduces several signals with non-trivial correlation")
    w("to acceptance rate, but the equal-weight averaging dilutes the influence of the strongest")
    w("predictors. The recommended path forward is:")
    w()
    w("1. Adopt correlation-weighted signals (Section 4)")
    w("2. Prune signals with consistently low |r| across both model pairs")
    w("3. Consider a learned weighting for production deployment")
    w("4. The split of entropy_gap into pos/neg components should be evaluated — if one component")
    w("   is consistently stronger, keep only that one")
    w()

    # =========================================================================
    # Write output
    # =========================================================================
    output_path = os.path.join(BASE, "results", "analysis_14signals.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Analysis written to {output_path}")
    print(f"Total lines: {len(out_lines)}")


if __name__ == "__main__":
    main()
