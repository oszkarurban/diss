#!/usr/bin/env python3
"""
analysis_v3_deep.py — Comprehensive analysis of dynamic speculative decoding V3.

Processes signal data JSON files (from test_signal_collection.py) and produces
a detailed markdown report with:
  A. Signal-acceptance correlation matrix
  B. Config efficiency analysis
  C. Token-level analysis (if available in enriched data)
  D. Failure mode analysis
  E. Temporal patterns
  F. Opportunity analysis (oracle upper bound)
  G. Topk vs chain analysis

Uses only Python stdlib (no numpy/pandas) for HPC compatibility.

Usage:
    python analysis_v3_deep.py \
        --dynamic signal_data_v3_llama_314_748.json \
        --static signal_data_static_748_llama.json \
        --label "Llama 3.1 8B" \
        --output results/analysis_v3_deep.md

    # Multi-model comparison:
    python analysis_v3_deep.py \
        --dynamic signal_data_v3_llama.json signal_data_v3_deepseek.json signal_data_v3_qwen.json \
        --static signal_data_static_llama.json signal_data_static_deepseek.json signal_data_static_qwen.json \
        --label "Llama" "DeepSeek" "Qwen" \
        --output results/analysis_v3_deep.md
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ── Data loading ───────────────────────────────────────────────────────────


def load_signal_data(path: str) -> Dict[str, Any]:
    """Load a signal data JSON file, handling both old and new formats.

    Old format: per_turn_logs[i] = [step_dict, step_dict, ...]
    New format: per_turn_logs[i] = {"signals": [...], "spec_draft_tokens": [...], ...}

    Returns normalized structure with separate signal_steps and token_data.
    """
    with open(path) as f:
        data = json.load(f)

    turns = data.get("per_turn_logs", [])
    all_signal_steps = []
    all_token_data = []

    for turn in turns:
        if isinstance(turn, list):
            # Old format: turn IS the list of signal step dicts
            all_signal_steps.append(turn)
            all_token_data.append(None)
        elif isinstance(turn, dict):
            # New format: turn has "signals" key + token-level keys
            all_signal_steps.append(turn.get("signals", []))
            token_data = {
                k: turn.get(k, [])
                for k in [
                    "spec_draft_tokens",
                    "spec_accepted_tokens_log",
                    "spec_rejected_tokens_log",
                    "spec_accept_index_log",
                    "spec_topk",
                    "spec_num_steps",
                    "spec_draft_token_num",
                    "spec_retrive_next_token",
                    "spec_retrive_next_sibling",
                ]
            }
            all_token_data.append(token_data)
        else:
            all_signal_steps.append([])
            all_token_data.append(None)

    flat_steps = [s for turn in all_signal_steps for s in turn]
    has_token_data = any(t is not None and t.get("spec_draft_tokens") for t in all_token_data)

    return {
        "path": path,
        "meta": {
            k: data.get(k)
            for k in ["server", "num_questions", "num_turns", "num_steps_total", "elapsed_seconds"]
        },
        "per_turn_signals": all_signal_steps,
        "per_turn_tokens": all_token_data,
        "flat_steps": flat_steps,
        "has_token_data": has_token_data,
    }


# ── Statistics helpers ─────────────────────────────────────────────────────


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx * sy < 1e-12:
        return 0.0
    return cov / (sx * sy)


def percentile(vals: List[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ── Analysis sections ─────────────────────────────────────────────────────


ACTIVE_SIGNALS = [
    "top1_prob",
    "target_top1_prob",
    "rolling_accept_rate",
    "rolling_accept_length",
    "draft_oracle_gate",
    "target_oracle_gate_fixed",
    "joint_confidence_product_fixed",
    "confidence_agreement",
]


def section_a_correlations(steps: List[Dict], label: str) -> str:
    """A. Signal-Acceptance Correlation Matrix."""
    out = [f"### A. Signal-Acceptance Correlations — {label}\n"]

    if not steps:
        return out[0] + "No data.\n\n"

    accs = [s["accept_length"] + 1 for s in steps]

    # Same-step correlations
    out.append("**Same-step: signal(i) → accept_length(i)**\n")
    out.append("| Signal | r | Interpretation |")
    out.append("|--------|---|----------------|")

    correlations = []
    for sig in ACTIVE_SIGNALS:
        vals = [s.get(sig, 0.0) for s in steps]
        r = pearson_r(vals, accs)
        correlations.append((sig, r))

    correlations.sort(key=lambda x: -abs(x[1]))
    for sig, r in correlations:
        interp = "strong" if abs(r) > 0.4 else "moderate" if abs(r) > 0.2 else "weak"
        out.append(f"| {sig} | {r:+.3f} | {interp} |")

    # Next-step correlations (temporal predictive power)
    out.append("\n**Next-step: signal(i) → accept_length(i+1)**\n")
    out.append("| Signal | r |")
    out.append("|--------|---|")

    # Need per-turn sequential access
    # Use flat_steps but we need turn boundaries... use a simpler approach
    # with consecutive pairs from flat_steps (approximation — breaks across turns)
    if len(steps) > 10:
        next_accs = [s["accept_length"] + 1 for s in steps[1:]]
        for sig in ACTIVE_SIGNALS:
            vals = [s.get(sig, 0.0) for s in steps[:-1]]
            r = pearson_r(vals, next_accs)
            out.append(f"| {sig} | {r:+.3f} |")

    out.append("")
    return "\n".join(out) + "\n"


def section_b_config_efficiency(steps: List[Dict], label: str) -> str:
    """B. Config Efficiency Analysis."""
    out = [f"### B. Config Efficiency — {label}\n"]

    if not steps:
        return out[0] + "No data.\n\n"

    # Check if this is dynamic data (has chosen_topk) or static
    is_dynamic = "chosen_topk" in steps[0]

    if is_dynamic:
        # Group by config
        config_data = defaultdict(lambda: {"acc_sf": [], "eff": []})
        for s in steps:
            topk = s["chosen_topk"]
            num_steps = s["chosen_num_steps"]
            ndt = s["chosen_num_draft_tokens"]
            acc_sf = s["accept_length"] + 1  # SpecForge-compatible
            eff = acc_sf / (num_steps + 1)
            config_data[(topk, num_steps, ndt)]["acc_sf"].append(acc_sf)
            config_data[(topk, num_steps, ndt)]["eff"].append(eff)

        out.append("**Per-config breakdown:**\n")
        out.append("| Config (topk,steps,ndt) | Count | % | AccLen | Eff | Zero% |")
        out.append("|-------------------------|-------|---|--------|-----|-------|")

        for cfg in sorted(config_data.keys(), key=lambda x: -len(config_data[x]["acc_sf"])):
            d = config_data[cfg]
            n = len(d["acc_sf"])
            pct = 100.0 * n / len(steps)
            ma = mean(d["acc_sf"])
            me = mean(d["eff"])
            zero = 100.0 * sum(1 for a in d["acc_sf"] if a == 1) / n  # acc_sf=1 means 0 draft accepted
            out.append(f"| {cfg} | {n} | {pct:.1f}% | {ma:.2f} | {me:.3f} | {zero:.0f}% |")

        # Overall
        all_acc_sf = [s["accept_length"] + 1 for s in steps]
        all_eff = [a / (s["chosen_num_steps"] + 1) for a, s in zip(all_acc_sf, steps)]
        out.append(f"\n**Overall**: AccLen={mean(all_acc_sf):.2f}, Eff={mean(all_eff):.3f}")

        # DOG bucket analysis
        out.append("\n**By DOG bucket:**\n")
        out.append("| DOG Range | Count | AccLen | Mean topk | Mean steps | Eff | Zero% |")
        out.append("|-----------|-------|--------|-----------|------------|-----|-------|")

        buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0)]
        for lo, hi in buckets:
            bucket = [s for s in steps if lo <= s.get("draft_oracle_gate", 0) < hi]
            if bucket:
                ma = mean([s["accept_length"] + 1 for s in bucket])
                mtk = mean([s["chosen_topk"] for s in bucket])
                ms = mean([s["chosen_num_steps"] for s in bucket])
                me = mean([(s["accept_length"] + 1) / (s["chosen_num_steps"] + 1) for s in bucket])
                zp = 100.0 * sum(1 for s in bucket if s["accept_length"] == 0) / len(bucket)
                out.append(f"| [{lo:.1f}, {hi:.1f}) | {len(bucket)} | {ma:.2f} | {mtk:.1f} | {ms:.1f} | {me:.3f} | {zp:.0f}% |")
    else:
        # Static data — no config variation
        accs_sf = [s["accept_length"] + 1 for s in steps]
        out.append(f"**Static config** — {len(steps)} steps")
        out.append(f"- AccLen: {mean(accs_sf):.2f}")
        out.append(f"- Std: {stdev(accs_sf):.2f}")
        if accs_sf:
            eff = mean([a / 8 for a in accs_sf])  # 7 steps + 1
            out.append(f"- Efficiency: {eff:.3f}")
            zero = 100.0 * sum(1 for a in accs_sf if a == 1) / len(accs_sf)
            out.append(f"- Zero acceptance: {zero:.1f}%")

    out.append("")
    return "\n".join(out) + "\n"


def section_c_token_level(data: Dict, label: str) -> str:
    """C. Token-Level Analysis (only if enriched data available)."""
    out = [f"### C. Token-Level Analysis — {label}\n"]

    if not data.get("has_token_data"):
        out.append("*No token-level data available. Re-collect with updated test_signal_collection.py.*\n")
        return "\n".join(out) + "\n"

    # Analyze rejection positions, tree utilization, branch value
    first_token_rejections = 0
    total_rejections = 0
    total_steps_with_tokens = 0
    branch_acceptances = 0  # times a non-top-1 branch was accepted
    total_tree_steps = 0  # steps where topk > 1

    for turn_signals, turn_tokens in zip(data["per_turn_signals"], data["per_turn_tokens"]):
        if turn_tokens is None:
            continue
        drafts = turn_tokens.get("spec_draft_tokens", [])
        accepted = turn_tokens.get("spec_accepted_tokens_log", [])
        rejected = turn_tokens.get("spec_rejected_tokens_log", [])
        accept_idx = turn_tokens.get("spec_accept_index_log", [])
        topks = turn_tokens.get("spec_topk", [])

        for step_i in range(len(drafts)):
            total_steps_with_tokens += 1
            rej = rejected[step_i] if step_i < len(rejected) else []
            acc = accepted[step_i] if step_i < len(accepted) else []
            draft = drafts[step_i] if step_i < len(drafts) else []
            tk = topks[step_i] if step_i < len(topks) else 1

            if len(acc) == 0 and len(draft) > 0:
                total_rejections += 1
                # First token rejection: the very first draft token was wrong
                first_token_rejections += 1

            if tk > 1:
                total_tree_steps += 1
                # Check if accepted indices include non-top-1 branches
                aidx = accept_idx[step_i] if step_i < len(accept_idx) else []
                if aidx:
                    # In a tree with topk>1, indices 0..topk-1 are first layer
                    # If any accepted index > 0 at the first branch, a non-top-1 was used
                    # Simplified check: if topk>1 and acc_len > 0, credit branch exploration
                    if len(acc) > 0 and any(idx > 0 for idx in aidx):
                        branch_acceptances += 1

    out.append(f"- Total steps with token data: {total_steps_with_tokens}")
    out.append(f"- Total rejections (acc=0): {total_rejections} ({100*total_rejections/max(total_steps_with_tokens,1):.1f}%)")
    out.append(f"- First-token rejections: {first_token_rejections}")
    out.append(f"- Tree steps (topk>1): {total_tree_steps}")
    if total_tree_steps > 0:
        out.append(f"- Non-top-1 branch accepted: {branch_acceptances}/{total_tree_steps} ({100*branch_acceptances/total_tree_steps:.1f}%)")

    out.append("")
    return "\n".join(out) + "\n"


def section_d_failures(steps: List[Dict], label: str) -> str:
    """D. Failure Mode Analysis."""
    out = [f"### D. Failure Modes — {label}\n"]

    if not steps:
        return out[0] + "No data.\n\n"

    is_dynamic = "chosen_topk" in steps[0]
    total = len(steps)
    zeros = [s for s in steps if s["accept_length"] == 0]
    n_zero = len(zeros)

    out.append(f"**Total rejections**: {n_zero}/{total} ({100*n_zero/total:.1f}%)\n")

    if is_dynamic and zeros:
        # Config at rejection
        out.append("**Config at total rejection:**\n")
        out.append("| Config | Count | % of rejections |")
        out.append("|--------|-------|-----------------|")
        cfg_counts = Counter(
            (s["chosen_topk"], s["chosen_num_steps"], s["chosen_num_draft_tokens"])
            for s in zeros
        )
        for cfg, cnt in cfg_counts.most_common():
            out.append(f"| {cfg} | {cnt} | {100*cnt/n_zero:.0f}% |")

        # Signal values at rejection vs success
        out.append("\n**Signal values: rejection vs success:**\n")
        out.append("| Signal | Mean@AccLen=1 | Mean@AccLen≥4 | Δ |")
        out.append("|--------|--------------|--------------|---|")
        good = [s for s in steps if s["accept_length"] + 1 >= 4]
        for sig in ACTIVE_SIGNALS:
            v_zero = mean([s.get(sig, 0) for s in zeros])
            v_good = mean([s.get(sig, 0) for s in good]) if good else 0
            delta = v_good - v_zero
            out.append(f"| {sig} | {v_zero:.3f} | {v_good:.3f} | {delta:+.3f} |")

        # Expensive failures: acc=0 on large configs
        expensive = [s for s in zeros if s["chosen_num_steps"] >= 5]
        out.append(f"\n**Expensive failures** (acc=0, steps≥5): {len(expensive)}/{n_zero} "
                   f"({100*len(expensive)/max(n_zero,1):.0f}% of rejections)")

    # Consecutive failure streaks
    streaks = []
    current_streak = 0
    for s in steps:
        if s["accept_length"] == 0:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    if streaks:
        out.append(f"\n**Consecutive failure streaks:**")
        out.append(f"- Total streaks: {len(streaks)}")
        out.append(f"- Mean length: {mean(streaks):.1f}")
        out.append(f"- Max length: {max(streaks)}")
        streak_dist = Counter(streaks)
        out.append(f"- Distribution: {dict(sorted(streak_dist.items()))}")
        out.append(f"- Streaks ≥5 (circuit breaker): {sum(1 for s in streaks if s >= 5)}")

    out.append("")
    return "\n".join(out) + "\n"


def section_e_temporal(steps: List[Dict], per_turn: List[List[Dict]], label: str) -> str:
    """E. Temporal Patterns."""
    out = [f"### E. Temporal Patterns — {label}\n"]

    if len(steps) < 10:
        return out[0] + "Insufficient data.\n\n"

    # Autocorrelation of accept_length (lag 1-3)
    accs = [s["accept_length"] + 1 for s in steps]
    out.append("**Accept_length autocorrelation:**\n")
    out.append("| Lag | r |")
    out.append("|-----|---|")
    for lag in [1, 2, 3, 5]:
        if lag < len(accs):
            r = pearson_r(accs[:-lag], accs[lag:])
            out.append(f"| {lag} | {r:+.3f} |")

    # DOG stability — how much does DOG change between consecutive steps?
    if "draft_oracle_gate" in steps[0]:
        dogs = [s["draft_oracle_gate"] for s in steps]
        dog_deltas = [abs(dogs[i+1] - dogs[i]) for i in range(len(dogs)-1)]
        out.append(f"\n**DOG stability:**")
        out.append(f"- Mean |ΔDOG| between steps: {mean(dog_deltas):.3f}")
        out.append(f"- DOG std: {stdev(dogs):.3f}")
        out.append(f"- DOG range: [{min(dogs):.3f}, {max(dogs):.3f}]")

    # Per-question difficulty
    if per_turn:
        out.append(f"\n**Per-turn summary** ({len(per_turn)} turns):\n")
        out.append("| Turn | Steps | AccLen | Mean DOG | Zero% |")
        out.append("|------|-------|--------|----------|-------|")
        for i, turn in enumerate(per_turn):
            if not turn:
                continue
            n = len(turn)
            ma = mean([s["accept_length"] + 1 for s in turn])
            dog_key = "draft_oracle_gate" if "draft_oracle_gate" in turn[0] else None
            md = mean([s.get(dog_key, 0) for s in turn]) if dog_key else 0
            zp = 100.0 * sum(1 for s in turn if s["accept_length"] == 0) / n
            out.append(f"| {i} | {n} | {ma:.2f} | {md:.3f} | {zp:.0f}% |")

    out.append("")
    return "\n".join(out) + "\n"


def section_f_opportunity(steps: List[Dict], label: str) -> str:
    """F. Opportunity Analysis — Oracle upper bound."""
    out = [f"### F. Opportunity Analysis — {label}\n"]

    if not steps:
        return out[0] + "No data.\n\n"

    is_dynamic = "chosen_topk" in steps[0]

    # SpecForge: total_tokens = sum(accept_length + 1) per step = total output tokens
    total_tokens = sum(s["accept_length"] + 1 for s in steps)
    total_steps = len(steps)
    mean_acclen = total_tokens / total_steps

    out.append(f"**Baseline**: {total_tokens} tokens in {total_steps} steps, AccLen={mean_acclen:.2f}\n")

    if is_dynamic:
        # Current efficiency
        cur_eff = mean([(s["accept_length"] + 1) / (s["chosen_num_steps"] + 1) for s in steps])

        # Oracle: what if we skipped all acc=0 steps entirely?
        productive = [s for s in steps if s["accept_length"] > 0]
        if productive:
            oracle_tokens = sum(s["accept_length"] + 1 for s in productive)
            oracle_acclen = oracle_tokens / len(productive)
            out.append(f"**Oracle (skip all AccLen=1 steps)**: {oracle_tokens} tokens in {len(productive)} steps, AccLen={oracle_acclen:.2f} (+{100*(oracle_acclen/mean_acclen-1):.0f}%)\n")

        # What if we used (1,3,4) for low DOG and (1,7,8) for high DOG? (chain-only)
        chain_eff_all = []
        for s in steps:
            dog = s.get("draft_oracle_gate", 0)
            if dog < 0.15:
                sim_steps = 3
            elif dog < 0.35:
                sim_steps = 5
            else:
                sim_steps = 7
            chain_eff_all.append((s["accept_length"] + 1) / (sim_steps + 1))

        chain_eff = mean(chain_eff_all)
        out.append(f"**Counterfactual chain-only (topk=1, vary steps)**:")
        out.append(f"- Efficiency: {chain_eff:.3f} (current: {cur_eff:.3f}, Δ={chain_eff-cur_eff:+.3f})")
        out.append(f"- Note: same AccLen assumed — actual may differ with topk=1\n")

        topk2_steps = [s for s in steps if s["chosen_topk"] == 2]
        topk1_steps = [s for s in steps if s["chosen_topk"] == 1]
        topk34_steps = [s for s in steps if s["chosen_topk"] >= 3]

        out.append(f"**Efficiency by topk group:**")
        for tk_label, tk_steps in [("topk=1", topk1_steps), ("topk=2", topk2_steps), ("topk≥3", topk34_steps)]:
            if tk_steps:
                e = mean([(s["accept_length"] + 1) / (s["chosen_num_steps"] + 1) for s in tk_steps])
                ma = mean([s["accept_length"] + 1 for s in tk_steps])
                out.append(f"- {tk_label}: eff={e:.3f}, AccLen={ma:.2f}, n={len(tk_steps)} ({100*len(tk_steps)/total_steps:.0f}%)")

    out.append("")
    return "\n".join(out) + "\n"


def section_g_topk_analysis(steps: List[Dict], label: str) -> str:
    """G. Topk vs Chain Analysis."""
    out = [f"### G. Topk vs Chain — {label}\n"]

    if not steps or "chosen_topk" not in steps[0]:
        out.append("*No dynamic config data — skipping.*\n")
        return "\n".join(out) + "\n"

    # Per-topk breakdown
    out.append("**Per-topk breakdown:**\n")
    out.append("| topk | Count | % | AccLen | Mean Steps | Eff | Zero% | Mean DOG |")
    out.append("|------|-------|---|--------|------------|-----|-------|----------|")

    topk_groups = defaultdict(list)
    for s in steps:
        topk_groups[s["chosen_topk"]].append(s)

    for tk in sorted(topk_groups.keys()):
        group = topk_groups[tk]
        n = len(group)
        pct = 100 * n / len(steps)
        ma = mean([s["accept_length"] + 1 for s in group])
        ms = mean([s["chosen_num_steps"] for s in group])
        me = mean([(s["accept_length"] + 1) / (s["chosen_num_steps"] + 1) for s in group])
        zp = 100 * sum(1 for s in group if s["accept_length"] == 0) / n
        md = mean([s.get("draft_oracle_gate", 0) for s in group])
        out.append(f"| {tk} | {n} | {pct:.0f}% | {ma:.2f} | {ms:.1f} | {me:.3f} | {zp:.0f}% | {md:.3f} |")

    # Key question: when topk>1 gets higher accept than topk=1 would at same DOG
    # Compare accept at similar DOG levels between topk=1 and topk>1
    out.append("\n**Topk=1 vs topk>1 at similar DOG levels:**\n")
    out.append("| DOG Range | topk=1 AccLen (n) | topk>1 AccLen (n) | Δ |")
    out.append("|-----------|-------------------|-------------------|---|")

    buckets = [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 1.0)]
    for lo, hi in buckets:
        tk1 = [s for s in steps if s["chosen_topk"] == 1 and lo <= s.get("draft_oracle_gate", 0) < hi]
        tkn = [s for s in steps if s["chosen_topk"] > 1 and lo <= s.get("draft_oracle_gate", 0) < hi]
        a1 = mean([s["accept_length"] + 1 for s in tk1]) if tk1 else float("nan")
        an = mean([s["accept_length"] + 1 for s in tkn]) if tkn else float("nan")
        d = an - a1 if tk1 and tkn else float("nan")
        a1_str = f"{a1:.2f} ({len(tk1)})" if tk1 else "—"
        an_str = f"{an:.2f} ({len(tkn)})" if tkn else "—"
        d_str = f"{d:+.2f}" if tk1 and tkn else "—"
        out.append(f"| [{lo:.1f}, {hi:.1f}) | {a1_str} | {an_str} | {d_str} |")

    out.append("")
    return "\n".join(out) + "\n"


# ── Comparison section ─────────────────────────────────────────────────────


def section_comparison(dynamic_steps: List[Dict], static_steps: List[Dict], label: str,
                       dynamic_meta: Optional[Dict] = None, static_meta: Optional[Dict] = None) -> str:
    """Compare dynamic vs static side by side."""
    out = [f"### Dynamic vs Static Comparison — {label}\n"]

    d_acc = mean([s["accept_length"] + 1 for s in dynamic_steps]) if dynamic_steps else 0
    s_acc = mean([s["accept_length"] + 1 for s in static_steps]) if static_steps else 0

    d_total = sum(s["accept_length"] + 1 for s in dynamic_steps) if dynamic_steps else 0
    s_total = sum(s["accept_length"] + 1 for s in static_steps) if static_steps else 0

    out.append("| Metric | Static 7,4,8 | Dynamic V3 |")
    out.append("|--------|-------------|------------|")
    out.append(f"| Total steps | {len(static_steps)} | {len(dynamic_steps)} |")
    out.append(f"| Total tokens | {s_total} | {d_total} |")
    out.append(f"| AccLen | {s_acc:.2f} | {d_acc:.2f} |")

    if dynamic_steps and "chosen_num_steps" in dynamic_steps[0]:
        d_eff = mean([(s["accept_length"] + 1) / (s["chosen_num_steps"] + 1) for s in dynamic_steps])
        s_eff = mean([(s["accept_length"] + 1) / 8 for s in static_steps]) if static_steps else 0  # 7+1
        out.append(f"| Efficiency | {s_eff:.3f} | {d_eff:.3f} |")

    d_zero = sum(1 for s in dynamic_steps if s["accept_length"] == 0)
    s_zero = sum(1 for s in static_steps if s["accept_length"] == 0)
    out.append(f"| Total rejections (AccLen=1) | {s_zero} ({100*s_zero/max(len(static_steps),1):.1f}%) | {d_zero} ({100*d_zero/max(len(dynamic_steps),1):.1f}%) |")

    # Throughput from elapsed time (SpecForge-compatible: tokens/second)
    s_elapsed = static_meta.get("elapsed_seconds") if static_meta else None
    d_elapsed = dynamic_meta.get("elapsed_seconds") if dynamic_meta else None
    s_tp_str = f"{s_total / s_elapsed:.1f}" if s_elapsed and s_elapsed > 0 else "—"
    d_tp_str = f"{d_total / d_elapsed:.1f}" if d_elapsed and d_elapsed > 0 else "—"
    out.append(f"| Throughput (tok/s) | {s_tp_str} | {d_tp_str} |")

    # Accept length distribution comparison (SpecForge-compatible: +1)
    out.append("\n**AccLen distribution:**\n")
    out.append("| AccLen | Static | Dynamic |")
    out.append("|--------|--------|---------|")
    max_acc = max(
        max((s["accept_length"] + 1 for s in static_steps), default=1),
        max((s["accept_length"] + 1 for s in dynamic_steps), default=1),
    )
    for a in range(1, max_acc + 1):
        sc = sum(1 for s in static_steps if s["accept_length"] + 1 == a)
        dc = sum(1 for s in dynamic_steps if s["accept_length"] + 1 == a)
        sp = 100 * sc / max(len(static_steps), 1)
        dp = 100 * dc / max(len(dynamic_steps), 1)
        out.append(f"| {a} | {sc} ({sp:.1f}%) | {dc} ({dp:.1f}%) |")

    out.append("")
    return "\n".join(out) + "\n"


# ── Report generation ─────────────────────────────────────────────────────


def generate_report(
    dynamic_datasets: List[Dict],
    static_datasets: List[Dict],
    labels: List[str],
) -> str:
    """Generate the full markdown report."""
    sections = []
    sections.append("# Dynamic Speculative Decoding V3 — Deep Analysis\n")
    sections.append(f"*Generated from {sum(len(d['flat_steps']) for d in dynamic_datasets)} dynamic steps "
                    f"and {sum(len(d['flat_steps']) for d in static_datasets)} static steps.*\n")
    sections.append("*All AccLen values are SpecForge-compatible (accepted_draft + 1 bonus token).*\n")
    sections.append("*Throughput = total_output_tokens / elapsed_seconds (same as SpecForge).*\n")

    for i, label in enumerate(labels):
        sections.append(f"\n## {label}\n")

        dyn = dynamic_datasets[i] if i < len(dynamic_datasets) else None
        sta = static_datasets[i] if i < len(static_datasets) else None

        dyn_steps = dyn["flat_steps"] if dyn else []
        sta_steps = sta["flat_steps"] if sta else []
        dyn_turns = dyn["per_turn_signals"] if dyn else []

        # Comparison
        if dyn_steps and sta_steps:
            dyn_meta = dyn["meta"] if dyn else None
            sta_meta = sta["meta"] if sta else None
            sections.append(section_comparison(dyn_steps, sta_steps, label, dyn_meta, sta_meta))

        # A. Correlations (dynamic data only — static has no signals)
        if dyn_steps:
            sections.append(section_a_correlations(dyn_steps, label))

        # B. Config efficiency
        if dyn_steps:
            sections.append(section_b_config_efficiency(dyn_steps, f"{label} (Dynamic)"))
        if sta_steps:
            sections.append(section_b_config_efficiency(sta_steps, f"{label} (Static)"))

        # C. Token-level
        if dyn:
            sections.append(section_c_token_level(dyn, label))

        # D. Failure modes
        if dyn_steps:
            sections.append(section_d_failures(dyn_steps, label))

        # E. Temporal
        if dyn_steps:
            sections.append(section_e_temporal(dyn_steps, dyn_turns, label))

        # F. Opportunity
        if dyn_steps:
            sections.append(section_f_opportunity(dyn_steps, label))

        # G. Topk vs chain
        if dyn_steps:
            sections.append(section_g_topk_analysis(dyn_steps, label))

    # Cross-model comparison
    if len(labels) > 1:
        sections.append("\n## Cross-Model Comparison\n")
        sections.append("| Model | Steps | AccLen | Eff | Zero% | Mean DOG |")
        sections.append("|-------|-------|--------|-----|-------|----------|")
        for i, label in enumerate(labels):
            dyn = dynamic_datasets[i] if i < len(dynamic_datasets) else None
            if dyn and dyn["flat_steps"]:
                steps = dyn["flat_steps"]
                ma = mean([s["accept_length"] + 1 for s in steps])
                me = mean([(s["accept_length"]+1)/(s["chosen_num_steps"]+1) for s in steps]) if "chosen_topk" in steps[0] else 0
                zp = 100*sum(1 for s in steps if s["accept_length"]==0)/len(steps)
                md = mean([s.get("draft_oracle_gate",0) for s in steps])
                sections.append(f"| {label} | {len(steps)} | {ma:.2f} | {me:.3f} | {zp:.0f}% | {md:.3f} |")

    return "\n".join(sections) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Deep analysis of V3 dynamic speculative decoding.")
    p.add_argument(
        "--dynamic", nargs="+", required=True,
        help="Path(s) to dynamic V3 signal data JSON file(s).",
    )
    p.add_argument(
        "--static", nargs="*", default=[],
        help="Path(s) to static signal data JSON file(s) (optional, for comparison).",
    )
    p.add_argument(
        "--label", nargs="+", default=["Model"],
        help="Label(s) for each model pair (must match number of --dynamic files).",
    )
    p.add_argument(
        "--output", default="results/analysis_v3_deep.md",
        help="Output markdown file.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Pad labels to match dynamic files
    while len(args.label) < len(args.dynamic):
        args.label.append(f"Model {len(args.label)+1}")

    # Load data
    print(f"Loading {len(args.dynamic)} dynamic dataset(s)...")
    dynamic_datasets = [load_signal_data(p) for p in args.dynamic]
    for d in dynamic_datasets:
        print(f"  {d['path']}: {len(d['flat_steps'])} steps, token_data={d['has_token_data']}")

    static_datasets = []
    if args.static:
        print(f"Loading {len(args.static)} static dataset(s)...")
        static_datasets = [load_signal_data(p) for p in args.static]
        for d in static_datasets:
            print(f"  {d['path']}: {len(d['flat_steps'])} steps")

    # Pad static to match dynamic
    while len(static_datasets) < len(dynamic_datasets):
        static_datasets.append({"flat_steps": [], "per_turn_signals": [], "per_turn_tokens": [], "has_token_data": False})

    # Generate report
    report = generate_report(dynamic_datasets, static_datasets, args.label)

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nReport written to: {args.output}")
    print(f"Total: {sum(len(d['flat_steps']) for d in dynamic_datasets)} dynamic + "
          f"{sum(len(d['flat_steps']) for d in static_datasets)} static steps analyzed.")


if __name__ == "__main__":
    main()
