#!/usr/bin/env python3
"""
26-Signal Deep Analysis for Dynamic Speculative Decoding
=========================================================
Comprehensive analysis of 26 signals across 2 model pairs.
Correlates signals with accept_length (instantaneous), rolling_accept_rate
(lagging EMA), and 3-step windowed accept_length (sustained).

Output: results/analysis_26signals.md

Uses only Python stdlib (no numpy/scipy).
"""

import json
import math
import os
import statistics as stats
from collections import Counter, defaultdict

BASE = "/rds/user/ou222/hpc-work/diss"

DATASETS = [
    ("Llama-8B", f"{BASE}/signal_data_26signals_llama8b.json"),
    ("DS+LlamaDraft", f"{BASE}/signal_data_26signals_deepseek8b_llamadraft.json"),
]

SIGNAL_KEYS = [
    # Draft-side (5)
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm", "hidden_cosine_sim",
    # Draft hidden directional (2)
    "hidden_projection_score", "hidden_entropy",
    # Target-side (4)
    "target_entropy", "target_top1_prob", "target_top1_gap", "target_varentropy",
    # Target hidden directional (4)
    "target_hidden_norm", "target_hidden_cosine_sim", "target_projection_score", "target_hidden_entropy",
    # Distribution divergence (2)
    "kl_approx_target_draft", "target_draft_agree",
    # Joint — original (4)
    "joint_entropy_gate", "draft_oracle_gate", "target_oracle_gate", "joint_confidence_product",
    # Joint — fixed with target_top1_prob (4)
    "draft_oracle_gate_fixed", "target_oracle_gate_fixed", "joint_confidence_product_fixed",
    "confidence_agreement",
    # Historical (1)
    "rolling_accept_rate",
]

# Signal metadata: (category, phase, formula_short, what_high_means, polarity_for_confidence)
SIGNAL_META = {
    "draft_entropy":            ("Draft", "Draft phase", "-sum(p log p) of draft softmax", "Draft model uncertain about prediction", "inverted"),
    "top1_prob":                ("Draft", "Extend phase", "topk_p[:, 0].mean()", "Draft strongly favors one token", "direct"),
    "top1_minus_top2":          ("Draft", "Extend phase", "(topk_p[:,0] - topk_p[:,1]).mean()", "Large gap between top-2 draft predictions", "direct"),
    "hidden_norm":              ("Draft hidden", "Extend phase", "torch.norm(hidden_states, dim=-1).mean()", "High activation magnitude in draft representation", "unclear"),
    "hidden_cosine_sim":        ("Draft hidden", "Draft phase", "cosine_sim(h[step_i], h[step_i-1])", "Draft representation stable between consecutive steps", "unclear"),
    "hidden_projection_score":  ("Draft hidden", "Draft phase", "h · W_lm[predicted_token] / ||W||", "Draft hidden strongly commits to its top prediction", "direct"),
    "hidden_entropy":           ("Draft hidden", "Draft phase", "entropy(h²/sum(h²))", "Draft activation energy spread across many dimensions", "inverted"),
    "target_entropy":           ("Target", "Verify phase", "-sum(p log p) of target softmax", "Target model uncertain — harder to match", "inverted"),
    "target_top1_gap":          ("Target", "Verify phase", "target top-1 minus top-2 probability", "Target has one clear winner — easy to match", "direct"),
    "target_varentropy":        ("Target", "Verify phase", "Var(-log p) of target distribution", "Target uncertainty is spread (uniform-ish) — harder", "inverted"),
    "target_hidden_cosine_sim": ("Target hidden", "Verify phase", "cosine_sim between consecutive target positions", "Target representation stable across draft positions", "unclear"),
    "target_projection_score":  ("Target hidden", "Verify phase", "h · W_target_lm[predicted] / ||W||", "Target hidden commits to its prediction", "direct"),
    "target_hidden_entropy":    ("Target hidden", "Verify phase", "entropy(h²/sum(h²)) of target hidden", "Target activation energy spread", "inverted"),
    "kl_approx_target_draft":   ("Divergence", "Verify phase", "F.cross_entropy(target_logits, draft_tokens)", "Target assigns low probability to draft's choices", "inverted"),
    "target_draft_agree":       ("Agreement", "Verify phase", "fraction argmax(target)==draft_token", "Target and draft agree on token choices", "direct"),
    "joint_entropy_gate":       ("Joint", "Derived", "(1/(1+draft_ent)) * (1/(1+target_ent))", "Both models are certain", "direct"),
    "draft_oracle_gate":        ("Joint", "Derived", "top1_prob * rolling_accept_rate", "Draft confident AND target has been accepting", "direct"),
    "target_oracle_gate":       ("Joint", "Derived", "target_top1_gap * rolling_accept_rate", "Target confident AND has been accepting", "direct"),
    "joint_confidence_product": ("Joint", "Derived", "top1_prob * target_top1_gap", "Both models confident in their top prediction", "direct"),
    "target_top1_prob":         ("Target", "Verify phase", "topk(target_probs, 1).values.mean()", "Target strongly favors one token", "direct"),
    "target_hidden_norm":       ("Target hidden", "Verify phase", "torch.norm(target_hs, dim=-1).mean()", "High activation magnitude in target representation", "unclear"),
    "draft_oracle_gate_fixed":  ("Joint (fixed)", "Derived", "top1_prob * rolling_accept_rate", "Draft confident AND historically accepted (same as original)", "direct"),
    "target_oracle_gate_fixed": ("Joint (fixed)", "Derived", "target_top1_prob * rolling_accept_rate", "Target confident (raw prob) AND historically accepted", "direct"),
    "joint_confidence_product_fixed": ("Joint (fixed)", "Derived", "top1_prob * target_top1_prob", "Both models confident (symmetric raw probs)", "direct"),
    "confidence_agreement":     ("Joint (2x2)", "Derived", "1 - abs(top1_prob - target_top1_prob)", "Both models at same confidence level (both certain OR both uncertain)", "direct"),
    "rolling_accept_rate":      ("Historical", "Verify EMA", "alpha*step_rate + (1-alpha)*prev, alpha=0.3", "Recent acceptance has been high", "direct"),
}


# ── Helpers ──────────────────────────────────────────────────────────────

def pearson(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / (n - 1)) if n > 1 else 1.0
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / (n - 1)) if n > 1 else 1.0
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / ((n - 1) * sx * sy)


def cohens_d(group1, group2):
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1 = stats.mean(group1)
    m2 = stats.mean(group2)
    s1 = stats.stdev(group1)
    s2 = stats.stdev(group2)
    pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2) if (s1 + s2) > 0 else 1.0
    return (m2 - m1) / pooled if pooled > 1e-10 else 0.0


def safe_stdev(vals):
    return stats.stdev(vals) if len(vals) > 1 else 0.0


def load_steps(path):
    with open(path) as f:
        data = json.load(f)
    all_steps = []
    for turn in data["per_turn_logs"]:
        turn_steps = [s for s in turn if "accept_length" in s]
        all_steps.append(turn_steps)
    flat = [s for turn in all_steps for s in turn]
    return flat, all_steps


def compute_windowed_al(turns, window=3):
    """Compute 3-step centered windowed accept_length within each turn."""
    windowed = []
    for turn_steps in turns:
        als = [s["accept_length"] for s in turn_steps]
        for i in range(len(als)):
            lo = max(0, i - window // 2)
            hi = min(len(als), i + window // 2 + 1)
            windowed.append(stats.mean(als[lo:hi]))
    return windowed


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    out_path = f"{BASE}/results/analysis_26signals.md"
    lines = []
    W = lines.append

    W("# 26-Signal Deep Analysis for Dynamic Speculative Decoding\n")
    W(f"**Date**: Generated from analysis_20signals.py")
    W(f"**Datasets**: {', '.join(name for name, _ in DATASETS)}")
    W("")

    # Load all data
    all_data = {}
    for name, path in DATASETS:
        print(f"Loading {name}...")
        flat, turns = load_steps(path)
        windowed_al = compute_windowed_al(turns, window=3)
        all_data[name] = {
            "flat": flat,
            "turns": turns,
            "windowed_al": windowed_al,
        }
        print(f"  {len(flat)} steps, {len(turns)} turns")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: Signal Encyclopedia
    # ═══════════════════════════════════════════════════════════════════
    print("Section 1: Signal Encyclopedia...")
    W("---\n")
    W("## Section 1: Signal Encyclopedia\n")

    for sig_idx, sig in enumerate(SIGNAL_KEYS, 1):
        meta = SIGNAL_META.get(sig, ("Unknown", "Unknown", "N/A", "N/A", "unclear"))
        category, phase, formula, what_high, polarity = meta

        W(f"### 1.{sig_idx} `{sig}` — {category}\n")
        W(f"- **Formula**: `{formula}`")
        W(f"- **Collected from**: {phase}")
        W(f"- **High value means**: {what_high}")
        W(f"- **Polarity**: {polarity} (high = {'more' if polarity == 'direct' else 'less' if polarity == 'inverted' else 'unclear'} confident)\n")

        # Stats table
        W(f"| Dataset | N | Mean | Std | Min | Max |")
        W(f"|---------|---|------|-----|-----|-----|")
        for name in all_data:
            vals = [s[sig] for s in all_data[name]["flat"]]
            n = len(vals)
            W(f"| {name} | {n} | {stats.mean(vals):.4f} | {safe_stdev(vals):.4f} | {min(vals):.4f} | {max(vals):.4f} |")
        W("")

        # Correlation table
        W(f"| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |")
        W(f"|---------|------------------|--------|------------------|---------------|")
        for name in all_data:
            flat = all_data[name]["flat"]
            vals = [s[sig] for s in flat]
            al = [s["accept_length"] for s in flat]
            rar = [s["rolling_accept_rate"] for s in flat]
            wal = all_data[name]["windowed_al"][:len(vals)]
            conf = [s["confidence"] for s in flat]
            r_al = pearson(vals, al)
            r_rar = pearson(vals, rar)
            r_wal = pearson(vals, wal) if len(wal) == len(vals) else 0.0
            r_conf = pearson(vals, conf)
            W(f"| {name} | {r_al:+.4f} | {r_rar:+.4f} | {r_wal:+.4f} | {r_conf:+.4f} |")
        W("")

        # Mean by accept_length bucket
        W(f"| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |")
        W(f"|---------|------|------|------|------|-------|")
        for name in all_data:
            flat = all_data[name]["flat"]
            buckets = defaultdict(list)
            for s in flat:
                al_b = min(s["accept_length"], 4)
                buckets[al_b].append(s[sig])
            row = f"| {name} "
            for b in [0, 1, 2, 3, 4]:
                if buckets[b]:
                    row += f"| {stats.mean(buckets[b]):.4f} "
                else:
                    row += "| N/A "
            row += "|"
            W(row)
        W("")

        # Cohen's d (separation: accept=0 vs accept>=3)
        W("| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |")
        W("|---------|--------------------------|---------|")
        for name in all_data:
            flat = all_data[name]["flat"]
            fail = [s[sig] for s in flat if s["accept_length"] == 0]
            success = [s[sig] for s in flat if s["accept_length"] >= 3]
            d = cohens_d(fail, success)
            # Verdict
            ad = abs(d)
            if ad > 0.8:
                verdict = "**Strong separator**"
            elif ad > 0.5:
                verdict = "Moderate separator"
            elif ad > 0.2:
                verdict = "Weak separator"
            else:
                verdict = "Not useful"
            W(f"| {name} | {d:+.3f} | {verdict} |")
        W("\n---\n")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: Correlation Rankings
    # ═══════════════════════════════════════════════════════════════════
    print("Section 2: Correlation Rankings...")
    W("## Section 2: Correlation Rankings\n")

    for metric_name, get_metric in [
        ("accept_length", lambda flat, wal: [s["accept_length"] for s in flat]),
        ("rolling_accept_rate", lambda flat, wal: [s["rolling_accept_rate"] for s in flat]),
        ("windowed_AL_3", lambda flat, wal: wal[:len(flat)]),
    ]:
        W(f"### Ranked by `|r(signal, {metric_name})|`\n")
        W(f"| Rank | Signal | r ({list(all_data.keys())[0]}) | r ({list(all_data.keys())[1]}) | Avg `\|r\|` | Consistent? |")
        W(f"|------|--------|------|------|---------|-------------|")

        sig_scores = []
        for sig in SIGNAL_KEYS:
            rs = []
            for name in all_data:
                flat = all_data[name]["flat"]
                vals = [s[sig] for s in flat]
                metric = get_metric(flat, all_data[name]["windowed_al"])
                if len(metric) != len(vals):
                    metric = metric[:len(vals)]
                r = pearson(vals, metric)
                rs.append(r)
            avg_abs = sum(abs(r) for r in rs) / len(rs)
            consistent = "YES" if all(r > 0 for r in rs) or all(r < 0 for r in rs) else "NO"
            sig_scores.append((avg_abs, sig, rs, consistent))

        sig_scores.sort(reverse=True)
        for rank, (avg, sig, rs, cons) in enumerate(sig_scores, 1):
            W(f"| {rank} | {sig} | {rs[0]:+.4f} | {rs[1]:+.4f} | {avg:.4f} | {cons} |")
        W("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: Failure Pattern Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("Section 3: Failure Pattern Analysis...")
    W("## Section 3: Failure Pattern Analysis\n")

    for name in all_data:
        flat = all_data[name]["flat"]
        W(f"### {name} ({len(flat)} steps)\n")

        # Accept length distribution
        al_dist = Counter(s["accept_length"] for s in flat)
        W(f"**Accept length distribution:**\n")
        W(f"| accept_length | Count | Pct |")
        W(f"|---------------|-------|-----|")
        for al in sorted(al_dist.keys()):
            W(f"| {al} | {al_dist[al]} | {al_dist[al]/len(flat)*100:.1f}% |")
        W("")

        # Wrong aggressive
        wrong_agg = [s for s in flat if s["accept_length"] == 0 and s["confidence"] > 0.55]
        W(f"**Wrong aggressive** (conf>0.55, accept=0): {len(wrong_agg)} steps ({len(wrong_agg)/len(flat)*100:.1f}%)\n")
        if wrong_agg:
            W(f"Mean signals in wrong-aggressive steps vs overall:\n")
            W(f"| Signal | Wrong-Agg Mean | Overall Mean | Delta |")
            W(f"|--------|---------------|-------------|-------|")
            for sig in ["draft_oracle_gate", "joint_confidence_product", "top1_prob",
                        "target_draft_agree", "target_entropy", "kl_approx_target_draft",
                        "rolling_accept_rate", "draft_entropy"]:
                wa_mean = stats.mean([s[sig] for s in wrong_agg])
                all_mean = stats.mean([s[sig] for s in flat])
                W(f"| {sig} | {wa_mean:.4f} | {all_mean:.4f} | {wa_mean - all_mean:+.4f} |")
            W("")

        # Wrong conservative
        wrong_con = [s for s in flat if s["accept_length"] >= 3 and s["confidence"] < 0.4]
        W(f"**Wrong conservative** (conf<0.4, accept≥3): {len(wrong_con)} steps ({len(wrong_con)/len(flat)*100:.1f}%)\n")
        if wrong_con:
            W(f"| Signal | Wrong-Con Mean | Overall Mean | Delta |")
            W(f"|--------|---------------|-------------|-------|")
            for sig in ["draft_oracle_gate", "joint_confidence_product", "top1_prob",
                        "target_draft_agree", "target_entropy", "rolling_accept_rate"]:
                wc_mean = stats.mean([s[sig] for s in wrong_con])
                all_mean = stats.mean([s[sig] for s in flat])
                W(f"| {sig} | {wc_mean:.4f} | {all_mean:.4f} | {wc_mean - all_mean:+.4f} |")
            W("")

        # Signature comparison: accept=0 vs accept=4
        fail_steps = [s for s in flat if s["accept_length"] == 0]
        perfect_steps = [s for s in flat if s["accept_length"] >= 4]
        if fail_steps and perfect_steps:
            W(f"**Signature comparison: total failure (al=0) vs perfect (al≥4)**\n")
            W(f"| Signal | Fail Mean | Perfect Mean | Gap | Cohen's d |")
            W(f"|--------|-----------|-------------|-----|-----------|")
            sigs_sorted = []
            for sig in SIGNAL_KEYS:
                fv = [s[sig] for s in fail_steps]
                pv = [s[sig] for s in perfect_steps]
                d = cohens_d(fv, pv)
                sigs_sorted.append((abs(d), sig, stats.mean(fv), stats.mean(pv), d))
            sigs_sorted.sort(reverse=True)
            for _, sig, fm, pm, d in sigs_sorted:
                W(f"| {sig} | {fm:.4f} | {pm:.4f} | {pm-fm:+.4f} | {d:+.3f} |")
            W("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: Signal Interaction Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("Section 4: Signal Interaction Analysis...")
    W("## Section 4: Signal Interaction Analysis\n")

    for name in all_data:
        flat = all_data[name]["flat"]
        W(f"### {name}: Highly correlated pairs (|r| > 0.7)\n")
        W(f"| Signal A | Signal B | r |")
        W(f"|----------|----------|---|")
        pairs_found = []
        for i, sig_a in enumerate(SIGNAL_KEYS):
            for j, sig_b in enumerate(SIGNAL_KEYS):
                if j <= i:
                    continue
                va = [s[sig_a] for s in flat]
                vb = [s[sig_b] for s in flat]
                r = pearson(va, vb)
                if abs(r) > 0.7:
                    pairs_found.append((abs(r), sig_a, sig_b, r))
        pairs_found.sort(reverse=True)
        for _, sa, sb, r in pairs_found:
            W(f"| {sa} | {sb} | {r:+.3f} |")
        W("")

        # Independent signals (max |r| with any other signal < 0.3)
        W(f"### {name}: Most independent signals (max |r| with others < 0.3)\n")
        for sig in SIGNAL_KEYS:
            va = [s[sig] for s in flat]
            max_r = 0
            for other in SIGNAL_KEYS:
                if other == sig:
                    continue
                vb = [s[other] for s in flat]
                r = abs(pearson(va, vb))
                if r > max_r:
                    max_r = r
            if max_r < 0.3:
                W(f"- `{sig}` (max |r| with others = {max_r:.3f})")
        W("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: Top-5 Signal Selection
    # ═══════════════════════════════════════════════════════════════════
    print("Section 5: Top-5 Signal Selection...")
    W("## Section 5: Top-5 Signal Selection for Cross-Validation\n")

    # Compute consensus |r| with accept_length
    consensus = {}
    for sig in SIGNAL_KEYS:
        rs = []
        for name in all_data:
            flat = all_data[name]["flat"]
            vals = [s[sig] for s in flat]
            al = [s["accept_length"] for s in flat]
            rs.append(abs(pearson(vals, al)))
        consensus[sig] = sum(rs) / len(rs)

    # Greedy selection: pick top by consensus |r|, but skip if |r| > 0.7 with already selected
    selected = []
    remaining = sorted(SIGNAL_KEYS, key=lambda s: consensus[s], reverse=True)

    # Use first dataset for inter-signal correlation
    ref_flat = all_data[list(all_data.keys())[0]]["flat"]

    for candidate in remaining:
        if len(selected) >= 5:
            break
        # Check correlation with already selected
        too_correlated = False
        vc = [s[candidate] for s in ref_flat]
        for sel in selected:
            vs = [s[sel] for s in ref_flat]
            r = abs(pearson(vc, vs))
            if r > 0.7:
                too_correlated = True
                break
        if not too_correlated:
            selected.append(candidate)

    W(f"**Selected 5 signals** (greedy: highest |r(accept_length)| with `|inter-r|` < 0.7):\n")
    for i, sig in enumerate(selected, 1):
        meta = SIGNAL_META.get(sig, ("?", "?", "?", "?", "?"))
        W(f"{i}. **`{sig}`** — {meta[0]}, consensus |r|={consensus[sig]:.4f}")
    W("")

    # 5x5 inter-correlation matrix
    W(f"### Inter-correlation matrix of selected 5:\n")
    header = "| Signal |"
    for sig in selected:
        short = sig[:12]
        header += f" {short} |"
    W(header)
    W("|--------|" + "------|" * len(selected))
    for sig_a in selected:
        va = [s[sig_a] for s in ref_flat]
        row = f"| {sig_a[:12]} |"
        for sig_b in selected:
            vb = [s[sig_b] for s in ref_flat]
            r = pearson(va, vb)
            row += f" {r:+.3f} |"
        W(row)
    W("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: Recommendations
    # ═══════════════════════════════════════════════════════════════════
    print("Section 6: Recommendations...")
    W("## Section 6: Recommendations\n")

    # Final ranking
    W("### Final signal ranking by consensus |r(accept_length)|\n")
    W(f"| Rank | Signal | Consensus `\|r\|` | Tier |")
    W(f"|------|--------|----------------|------|")
    ranked = sorted(SIGNAL_KEYS, key=lambda s: consensus[s], reverse=True)
    for rank, sig in enumerate(ranked, 1):
        c = consensus[sig]
        if c > 0.4:
            tier = "**KEEP (high)**"
        elif c > 0.25:
            tier = "KEEP (medium)"
        elif c > 0.1:
            tier = "Maybe"
        else:
            tier = "DROP"
        W(f"| {rank} | {sig} | {c:.4f} | {tier} |")
    W("")

    # Summary
    keep_high = [s for s in ranked if consensus[s] > 0.4]
    keep_med = [s for s in ranked if 0.25 < consensus[s] <= 0.4]
    maybe = [s for s in ranked if 0.1 < consensus[s] <= 0.25]
    drop = [s for s in ranked if consensus[s] <= 0.1]

    W(f"### Summary\n")
    W(f"- **KEEP (high, |r|>0.4)**: {', '.join(f'`{s}`' for s in keep_high)}")
    W(f"- **KEEP (medium, |r|>0.25)**: {', '.join(f'`{s}`' for s in keep_med)}")
    W(f"- **Maybe (|r|>0.1)**: {', '.join(f'`{s}`' for s in maybe)}")
    W(f"- **DROP (|r|≤0.1)**: {', '.join(f'`{s}`' for s in drop)}")
    W("")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 7: Per-Turn Trajectory Analysis — Spikes and Falls
    # ═══════════════════════════════════════════════════════════════════
    print("Section 7: Per-Turn Trajectory Analysis...")
    W("## Section 7: Per-Turn Trajectory Analysis — Where Accept Length Spikes and Falls\n")
    W("This section examines individual turns to find moments where acceptance")
    W("dramatically changes, and what signals look like at those transitions.\n")

    # Key signals to track at transitions
    track_sigs = [
        "draft_oracle_gate", "joint_confidence_product", "joint_confidence_product_fixed",
        "confidence_agreement", "target_draft_agree", "target_top1_prob",
        "top1_prob", "target_entropy", "rolling_accept_rate",
    ]

    for ds_name in all_data:
        turns = all_data[ds_name]["turns"]
        flat = all_data[ds_name]["flat"]
        W(f"### {ds_name}\n")

        # Find the 3 most interesting turns (longest, to have enough data)
        turn_lens = [(len(t), ti) for ti, t in enumerate(turns) if len(t) > 20]
        turn_lens.sort(reverse=True)
        interesting_turns = [ti for _, ti in turn_lens[:3]]

        for turn_idx in interesting_turns:
            turn_steps = turns[turn_idx]
            if not turn_steps:
                continue
            als = [s["accept_length"] for s in turn_steps]
            n_steps = len(als)

            # Find spikes (accept_length jumps up by ≥2) and falls (drops by ≥2)
            spikes = []
            falls = []
            for i in range(1, n_steps):
                delta = als[i] - als[i - 1]
                if delta >= 2:
                    spikes.append(i)
                elif delta <= -2:
                    falls.append(i)

            W(f"**Turn {turn_idx}** ({n_steps} steps, mean accept_len={stats.mean(als):.2f}):\n")

            # Show accept_length trajectory as compact text
            al_str = " ".join(str(a) for a in als[:60])
            if n_steps > 60:
                al_str += f" ... ({n_steps - 60} more)"
            W(f"Accept length trajectory: `{al_str}`\n")

            # Spikes
            if spikes:
                W(f"**Spikes** (accept_length jumps up by ≥2): {len(spikes)} transitions\n")
                for sp in spikes[:3]:  # show up to 3
                    s_before = turn_steps[sp - 1]
                    s_after = turn_steps[sp]
                    W(f"Step {sp-1}→{sp}: accept_length {als[sp-1]}→{als[sp]}\n")
                    W(f"| Signal | Before (al={als[sp-1]}) | After (al={als[sp]}) | Delta |")
                    W(f"|--------|--------|-------|-------|")
                    for sig in track_sigs:
                        v_b = s_before[sig]
                        v_a = s_after[sig]
                        W(f"| {sig} | {v_b:.4f} | {v_a:.4f} | {v_a - v_b:+.4f} |")
                    W("")

            # Falls
            if falls:
                W(f"**Falls** (accept_length drops by ≥2): {len(falls)} transitions\n")
                for fl in falls[:3]:
                    s_before = turn_steps[fl - 1]
                    s_after = turn_steps[fl]
                    W(f"Step {fl-1}→{fl}: accept_length {als[fl-1]}→{als[fl]}\n")
                    W(f"| Signal | Before (al={als[fl-1]}) | After (al={als[fl]}) | Delta |")
                    W(f"|--------|--------|-------|-------|")
                    for sig in track_sigs:
                        v_b = s_before[sig]
                        v_a = s_after[sig]
                        W(f"| {sig} | {v_b:.4f} | {v_a:.4f} | {v_a - v_b:+.4f} |")
                    W("")

            # Sustained success streaks (3+ consecutive steps with accept_length ≥ 3)
            streak_start = None
            streaks = []
            for i in range(n_steps):
                if als[i] >= 3:
                    if streak_start is None:
                        streak_start = i
                else:
                    if streak_start is not None and (i - streak_start) >= 3:
                        streaks.append((streak_start, i - 1))
                    streak_start = None
            if streak_start is not None and (n_steps - streak_start) >= 3:
                streaks.append((streak_start, n_steps - 1))

            if streaks:
                W(f"**Sustained success streaks** (≥3 consecutive steps with accept_length ≥ 3): {len(streaks)}\n")
                for si, (start, end) in enumerate(streaks[:2]):
                    streak_steps = turn_steps[start:end + 1]
                    W(f"Streak {si+1}: steps {start}-{end} ({end-start+1} steps), "
                      f"mean accept_len={stats.mean([s['accept_length'] for s in streak_steps]):.2f}\n")
                    W(f"| Signal | Streak Mean | Turn Mean | Delta |")
                    W(f"|--------|------------|-----------|-------|")
                    for sig in track_sigs:
                        streak_mean = stats.mean([s[sig] for s in streak_steps])
                        turn_mean = stats.mean([s[sig] for s in turn_steps])
                        W(f"| {sig} | {streak_mean:.4f} | {turn_mean:.4f} | {streak_mean - turn_mean:+.4f} |")
                    W("")

            # Sustained failure streaks (3+ consecutive steps with accept_length = 0)
            streak_start = None
            fail_streaks = []
            for i in range(n_steps):
                if als[i] == 0:
                    if streak_start is None:
                        streak_start = i
                else:
                    if streak_start is not None and (i - streak_start) >= 3:
                        fail_streaks.append((streak_start, i - 1))
                    streak_start = None
            if streak_start is not None and (n_steps - streak_start) >= 3:
                fail_streaks.append((streak_start, n_steps - 1))

            if fail_streaks:
                W(f"**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): {len(fail_streaks)}\n")
                for si, (start, end) in enumerate(fail_streaks[:2]):
                    streak_steps = turn_steps[start:end + 1]
                    W(f"Failure streak {si+1}: steps {start}-{end} ({end-start+1} steps)\n")
                    W(f"| Signal | Failure Mean | Turn Mean | Delta |")
                    W(f"|--------|-------------|-----------|-------|")
                    for sig in track_sigs:
                        streak_mean = stats.mean([s[sig] for s in streak_steps])
                        turn_mean = stats.mean([s[sig] for s in turn_steps])
                        W(f"| {sig} | {streak_mean:.4f} | {turn_mean:.4f} | {streak_mean - turn_mean:+.4f} |")
                    W("")

            W("---\n")

    # Write output
    print(f"Writing output to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Done. Output file: {out_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
