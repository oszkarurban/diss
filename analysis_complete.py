#!/usr/bin/env python3
"""
Comprehensive signal analysis for dynamic speculative decoding experiments.
Processes all server logs and signal JSONs, produces signal_analysis_complete.md.

Uses only Python stdlib: json, re, os, statistics, collections, math.
"""

import json
import re
import os
import math
import statistics
from collections import OrderedDict

BASE = "/rds/user/ou222/hpc-work/diss"
OUT_DIR = os.path.join(BASE, "results")
OUT_FILE = os.path.join(OUT_DIR, "signal_analysis_complete.md")

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = OrderedDict([
    ("vanilla_llama8b", {
        "log": "logs/server_vanilla_llama8b.log",
        "signal_json": None,
        "target": "Llama-3.1-8B-Instruct",
        "draft": "EAGLE3-LLaMA3.1-8B",
        "signal_version": "vanilla",
        "num_questions": 5,
        "label": "Vanilla Llama (5q)",
    }),
    ("vanilla_deepseek8b", {
        "log": "logs/server_vanilla_deepseek8b.log",
        "signal_json": None,
        "target": "DeepSeek-R1-Distill-Llama-8B",
        "draft": "EAGLE3-DeepSeek-R1-8B",
        "signal_version": "vanilla",
        "num_questions": 5,
        "label": "Vanilla DS (5q)",
    }),
    ("vanilla_deepseek8b_wrongdraft", {
        "log": "logs/server_vanilla_deepseek8b_wrongdraft.log",
        "signal_json": None,
        "target": "DeepSeek-R1-Distill-Llama-8B",
        "draft": "EAGLE3-LLaMA3.1-8B (mismatched)",
        "signal_version": "vanilla",
        "num_questions": 5,
        "label": "Vanilla DS+Llama (5q)",
    }),
    ("dynamic_llama8b", {
        "log": "logs/server_dynamic_llama8b.log",
        "signal_json": "signal_data_dynamic_llama8b.json",
        "target": "Llama-3.1-8B-Instruct",
        "draft": "EAGLE3-LLaMA3.1-8B",
        "signal_version": "7sig",
        "num_questions": 5,
        "label": "7sig Llama (5q)",
    }),
    ("dynamic_deepseek8b", {
        "log": "logs/server_dynamic_deepseek8b.log",
        "signal_json": "signal_data_dynamic_deepseek8b.json",
        "target": "DeepSeek-R1-Distill-Llama-8B",
        "draft": "EAGLE3-DeepSeek-R1-8B",
        "signal_version": "7sig",
        "num_questions": 5,
        "label": "7sig DS (5q)",
    }),
    ("dynamic_deepseek8b_wrongdraft", {
        "log": "logs/server_dynamic_deepseek8b_wrongdraft.log",
        "signal_json": "signal_data_dynamic_deepseek8b_wrongdraft.json",
        "target": "DeepSeek-R1-Distill-Llama-8B",
        "draft": "EAGLE3-LLaMA3.1-8B (mismatched)",
        "signal_version": "7sig",
        "num_questions": 5,
        "label": "7sig DS+Llama (5q)",
    }),
    ("14signals", {
        "log": "logs/server_14signals.log",
        "signal_json": "signal_data_14signals.json",
        "target": "Llama-3.1-8B-Instruct",
        "draft": "EAGLE3-LLaMA3.1-8B",
        "signal_version": "14sig",
        "num_questions": 5,
        "label": "14sig Llama (5q)",
    }),
    ("14signals_deepseek8b_llamadraft", {
        "log": "logs/server_14signals_deepseek8b_llamadraft.log",
        "signal_json": "signal_data_14signals_deepseek8b_llamadraft.json",
        "target": "DeepSeek-R1-Distill-Llama-8B",
        "draft": "EAGLE3-LLaMA3.1-8B (mismatched)",
        "signal_version": "14sig",
        "num_questions": 5,
        "label": "14sig DS+Llama (5q)",
    }),
    ("11signals_llama8b", {
        "log": "logs/server_11signals_llama8b.log",
        "signal_json": "signal_data_11signals_llama8b.json",
        "target": "Llama-3.1-8B-Instruct",
        "draft": "EAGLE3-LLaMA3.1-8B",
        "signal_version": "11sig",
        "num_questions": 10,
        "label": "11sig Llama (10q)",
    }),
    ("11signals_deepseek8b_llamadraft", {
        "log": "logs/server_11signals_deepseek8b_llamadraft.log",
        "signal_json": "signal_data_11signals_deepseek8b_llamadraft.json",
        "target": "DeepSeek-R1-Distill-Llama-8B",
        "draft": "EAGLE3-LLaMA3.1-8B (mismatched)",
        "signal_version": "11sig",
        "num_questions": 10,
        "label": "11sig DS+Llama (10q)",
    }),
])

# Signal sets per version
SIGNALS_7 = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "target_entropy", "entropy_gap", "rolling_accept_rate",
]

SIGNALS_14 = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "hidden_cosine_sim", "hidden_var", "hidden_max",
    "target_entropy", "target_top1_gap", "target_varentropy",
    "joint_entropy_gate", "entropy_gap_pos", "entropy_gap_neg",
    "rolling_accept_rate",
]

SIGNALS_11 = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "hidden_cosine_sim", "target_entropy", "target_top1_gap",
    "target_varentropy", "joint_entropy_gate", "draft_oracle_gate",
    "rolling_accept_rate",
]

# All 16 unique signals across all versions
ALL_SIGNALS = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "hidden_cosine_sim", "hidden_var", "hidden_max",
    "target_entropy", "target_top1_gap", "target_varentropy",
    "joint_entropy_gate", "entropy_gap", "entropy_gap_pos", "entropy_gap_neg",
    "draft_oracle_gate", "rolling_accept_rate",
]

# Signal experiments mapping (which signal JSONs have which signals)
SIGNAL_EXPERIMENTS = OrderedDict([
    ("dynamic_llama8b", {"version": "7sig", "signals": SIGNALS_7, "label": "7sig Llama (5q)"}),
    ("dynamic_deepseek8b", {"version": "7sig", "signals": SIGNALS_7, "label": "7sig DS (5q)"}),
    ("dynamic_deepseek8b_wrongdraft", {"version": "7sig", "signals": SIGNALS_7, "label": "7sig DS+Llama (5q)"}),
    ("14signals", {"version": "14sig", "signals": SIGNALS_14, "label": "14sig Llama (5q)"}),
    ("14signals_deepseek8b_llamadraft", {"version": "14sig", "signals": SIGNALS_14, "label": "14sig DS+Llama (5q)"}),
    ("11signals_llama8b", {"version": "11sig", "signals": SIGNALS_11, "label": "11sig Llama (10q)"}),
    ("11signals_deepseek8b_llamadraft", {"version": "11sig", "signals": SIGNALS_11, "label": "11sig DS+Llama (10q)"}),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Parse a server log for accept_len and gen_throughput values."""
    full = os.path.join(BASE, log_path)
    accept_lens = []
    throughputs = []
    with open(full, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="ignore")
            m_al = re.search(r"accept len:\s*([\d.]+)", line)
            m_tp = re.search(r"gen throughput \(token/s\):\s*([\d.]+)", line)
            if m_al:
                accept_lens.append(float(m_al.group(1)))
            if m_tp:
                throughputs.append(float(m_tp.group(1)))
    return accept_lens, throughputs


def steady_state(values, throughputs_for_filter=None):
    """Return values excluding initial warmup batches (throughput < 10)."""
    if throughputs_for_filter is not None and len(throughputs_for_filter) == len(values):
        return [v for v, t in zip(values, throughputs_for_filter) if t >= 10.0]
    # If no throughput filter, just skip the first value
    return values[1:] if len(values) > 1 else values


def safe_mean(vals):
    return statistics.mean(vals) if vals else float("nan")


def safe_stdev(vals):
    return statistics.stdev(vals) if len(vals) >= 2 else float("nan")


def safe_median(vals):
    return statistics.median(vals) if vals else float("nan")


def pearson_r(xs, ys):
    """Compute Pearson correlation coefficient."""
    n = min(len(xs), len(ys))
    if n < 3:
        return float("nan")
    xs, ys = xs[:n], ys[:n]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def load_signal_data(json_name):
    """Load signal JSON and flatten per_turn_logs into list of step dicts."""
    if json_name is None:
        return [], {}
    full = os.path.join(BASE, json_name)
    if not os.path.exists(full):
        return [], {}
    with open(full, "r") as f:
        data = json.load(f)
    meta = {k: v for k, v in data.items() if k != "per_turn_logs"}
    steps = []
    for turn in data.get("per_turn_logs", []):
        if isinstance(turn, list):
            steps.extend(turn)
        elif isinstance(turn, dict):
            steps.append(turn)
    return steps, meta


def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def fmt2(v):
    return fmt(v, 2)


def fmt3(v):
    return fmt(v, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def w(s=""):
        lines.append(s)

    # -----------------------------------------------------------------------
    # Load all data
    # -----------------------------------------------------------------------
    print("Loading server logs...")
    log_data = {}
    for name, exp in EXPERIMENTS.items():
        al, tp = parse_log(exp["log"])
        ss_al = steady_state(al, tp)
        ss_tp = steady_state(tp, tp)
        log_data[name] = {
            "accept_lens": al,
            "throughputs": tp,
            "ss_accept_lens": ss_al,
            "ss_throughputs": ss_tp,
        }
        print(f"  {name}: {len(al)} decode batches, ss={len(ss_al)}")

    print("\nLoading signal JSONs...")
    signal_data = {}
    for name, info in SIGNAL_EXPERIMENTS.items():
        exp = EXPERIMENTS[name]
        steps, meta = load_signal_data(exp["signal_json"])
        signal_data[name] = {"steps": steps, "meta": meta}
        print(f"  {name}: {len(steps)} steps")

    # -----------------------------------------------------------------------
    # Section 1: Experiment Overview
    # -----------------------------------------------------------------------
    print("\nWriting Section 1: Experiment Overview...")
    w("# Comprehensive Signal Analysis for Dynamic Speculative Decoding")
    w()
    w("## 1. Experiment Overview")
    w()
    w("| Config | Target Model | Draft Model | Signal Ver. | # Questions | Accept Len (mean) | Accept Len (SS) | Throughput SS (tok/s) | # Decode Batches | # Signal Steps |")
    w("|--------|-------------|-------------|-------------|-------------|-------------------|-----------------|----------------------|------------------|----------------|")

    for name, exp in EXPERIMENTS.items():
        ld = log_data[name]
        sd = signal_data.get(name, {"steps": [], "meta": {}})
        n_steps = len(sd["steps"])
        al_mean = fmt2(safe_mean(ld["accept_lens"])) if ld["accept_lens"] else "N/A"
        al_ss = fmt2(safe_mean(ld["ss_accept_lens"])) if ld["ss_accept_lens"] else "N/A"
        tp_ss = fmt2(safe_mean(ld["ss_throughputs"])) if ld["ss_throughputs"] else "N/A"
        w(f"| {exp['label']} | {exp['target'][:25]} | {exp['draft'][:25]} | {exp['signal_version']} | {exp['num_questions']} | {al_mean} | {al_ss} | {tp_ss} | {len(ld['accept_lens'])} | {n_steps} |")

    w()

    # -----------------------------------------------------------------------
    # Section 2: Signal-by-Signal Encyclopedia
    # -----------------------------------------------------------------------
    print("\nWriting Section 2: Signal-by-Signal Encyclopedia...")
    w("## 2. Signal-by-Signal Encyclopedia")
    w()

    # Signal metadata
    SIGNAL_META = {
        "draft_entropy": {
            "category": "Draft Quality",
            "status": "ACTIVE",
            "definition": "Shannon entropy of the draft model's output probability distribution at each draft step. Higher values indicate the draft model is uncertain about its prediction.",
            "hypothesis": "When draft entropy is low, the draft model is confident and more likely to match the target model's greedy pick, leading to higher acceptance.",
            "collection": "Phase: Draft. Computed in `eagle_worker.py:draft_forward()` after `probs = softmax(logits)` as `-sum(probs * log(probs))`. Written to pre-allocated CUDA graph signal buffer (`signal_draft_entropy`). CUDA graph compatible. Low cost (fused with existing softmax).",
        },
        "top1_prob": {
            "category": "Draft Quality",
            "status": "ACTIVE",
            "definition": "The probability assigned to the top-1 draft token by the draft model. Ranges from 0 to 1.",
            "hypothesis": "Higher top-1 probability means the draft model strongly favors one token, which should correlate with acceptance if target agrees.",
            "collection": "Phase: Extend (carried from draft). Read from `spec_info.topk_p[:, 0]` in `_collect_signals()`. CUDA graph compatible (read from graph output). Negligible cost.",
        },
        "top1_minus_top2": {
            "category": "Draft Quality",
            "status": "ACTIVE",
            "definition": "Difference between the top-1 and top-2 draft token probabilities. Measures how decisively the draft model favors its top pick.",
            "hypothesis": "A large gap suggests the draft model is decisive, not hedging between alternatives, which should predict acceptance.",
            "collection": "Phase: Extend. Computed from `topk_p[:, 0] - topk_p[:, 1]` in `_collect_signals()`. Requires topk >= 2. CUDA graph compatible. Negligible cost.",
        },
        "hidden_norm": {
            "category": "Representation",
            "status": "ACTIVE",
            "definition": "L2 norm of the draft model's hidden state vector, averaged across the sequence. Captures activation magnitude.",
            "hypothesis": "Abnormally high or low hidden norms may indicate the model is in an unusual state, potentially leading to draft-target disagreement.",
            "collection": "Phase: Extend. Computed from `torch.norm(hidden_states, dim=-1).mean()` in `_collect_signals()`. CUDA graph compatible. Low cost.",
        },
        "hidden_cosine_sim": {
            "category": "Representation",
            "status": "ACTIVE",
            "definition": "Cosine similarity between consecutive hidden states from the draft model. Measures how much the representation changes between steps.",
            "hypothesis": "High cosine similarity (stable representations) may indicate the model is in a predictable region, correlating with acceptance.",
            "collection": "Phase: Extend. Computed from consecutive hidden states in `_collect_signals()`. CUDA graph compatible. Low cost.",
        },
        "hidden_var": {
            "category": "Representation",
            "status": "REMOVED",
            "definition": "Variance of the draft model's hidden state activations across dimensions. Captures how spread out the activation values are.",
            "hypothesis": "Higher variance might indicate a more informative hidden state, potentially correlating with prediction quality.",
            "collection": "Phase: Extend. Computed from hidden state tensor variance. CUDA graph compatible. Low cost.",
            "removal_reason": "Removed: r=0.99 with hidden_norm (near-perfect redundancy). The variance and norm of a vector are algebraically related, so this signal adds no independent information.",
        },
        "hidden_max": {
            "category": "Representation",
            "status": "REMOVED",
            "definition": "Maximum activation value in the draft model's hidden state vector. Captures peak activation magnitude.",
            "hypothesis": "Extreme activation values might indicate confident internal representations.",
            "collection": "Phase: Extend. Computed from max of hidden state tensor. CUDA graph compatible. Negligible cost.",
            "removal_reason": "Removed: r < 0.05 with acceptance rate (pure noise). The maximum single activation has no predictive relationship with draft quality.",
        },
        "target_entropy": {
            "category": "Target Quality",
            "status": "ACTIVE",
            "definition": "Shannon entropy of the target model's output distribution at the verification step. Measures how uncertain the target model is.",
            "hypothesis": "Low target entropy means the target model strongly favors one token. If the draft picked that token, acceptance is likely. High entropy means any draft token has a reasonable chance.",
            "collection": "Phase: Verify. Computed from `softmax(logits) * log(softmax(logits))` on target logits in `eagle_info.py:verify()`. Eager mode (not in CUDA graph). Moderate cost (full softmax on target vocab).",
        },
        "target_top1_gap": {
            "category": "Target Quality",
            "status": "ACTIVE",
            "definition": "Gap between the target model's top-1 and top-2 token probabilities. Analogous to top1_minus_top2 but on the target side.",
            "hypothesis": "A large target gap means the target strongly prefers one token. If the draft model also picked it, acceptance follows.",
            "collection": "Phase: Verify. Computed from target logits after softmax in `eagle_info.py:verify()`. Eager mode. Moderate cost.",
        },
        "target_varentropy": {
            "category": "Target Quality",
            "status": "ACTIVE",
            "definition": "Variance of the entropy across the target model's token-level distributions. Captures heterogeneity of target confidence across positions.",
            "hypothesis": "Low varentropy indicates uniformly confident/uncertain predictions; high varentropy indicates mixed confidence, which may affect acceptance patterns.",
            "collection": "Phase: Verify. Computed from target logits in `eagle_info.py:verify()`. Eager mode. Moderate cost.",
        },
        "joint_entropy_gate": {
            "category": "Derived / Gate",
            "status": "ACTIVE",
            "definition": "Gating signal combining draft and target entropies. Computed as `sigmoid(target_entropy - draft_entropy) * min(draft_entropy, target_entropy) / max(draft_entropy, target_entropy)`.",
            "hypothesis": "Captures the interaction between draft and target uncertainty. When both agree on confidence level, this should be higher and predict acceptance.",
            "collection": "Phase: Derived. Computed in `_collect_signals()` from draft_entropy and target_entropy. No GPU cost beyond the constituent signals.",
        },
        "entropy_gap": {
            "category": "Derived",
            "status": "REMOVED (replaced)",
            "definition": "Simple difference: `target_entropy - draft_entropy`. Used in 7sig version.",
            "hypothesis": "Negative values (draft more uncertain than target) should predict lower acceptance since the draft model is spreading probability over tokens the target does not consider.",
            "collection": "Phase: Derived. Computed in `_collect_signals()`. Zero additional cost.",
            "removal_reason": "Replaced by entropy_gap_pos/entropy_gap_neg split in 14sig, then both halves removed. The signed gap conflates two different phenomena (draft overconfidence vs underconfidence).",
        },
        "entropy_gap_pos": {
            "category": "Derived",
            "status": "REMOVED",
            "definition": "Positive part of the entropy gap: `max(0, target_entropy - draft_entropy)`. Captures cases where the target is more uncertain than the draft.",
            "hypothesis": "When the target is more uncertain than the draft (gap > 0), the draft's confident pick should often be accepted.",
            "collection": "Phase: Derived. Computed in `_collect_signals()`. Zero cost.",
            "removal_reason": "Removed: too sparse. In practice, the draft model (smaller) is almost always more uncertain than the target, so this signal is zero for most steps.",
        },
        "entropy_gap_neg": {
            "category": "Derived",
            "status": "REMOVED",
            "definition": "Negative part of the entropy gap: `max(0, draft_entropy - target_entropy)`. Captures cases where the draft is more uncertain than the target.",
            "hypothesis": "Larger values (draft much more uncertain) should predict lower acceptance.",
            "collection": "Phase: Derived. Computed in `_collect_signals()`. Zero cost.",
            "removal_reason": "Removed: r=0.96 with draft_entropy. Since draft entropy is almost always higher than target entropy, `max(0, draft_entropy - target_entropy)` approximately equals `draft_entropy - constant`, making it a near-linear transform of draft_entropy.",
        },
        "draft_oracle_gate": {
            "category": "Derived / Gate",
            "status": "ACTIVE",
            "definition": "Gating signal that combines draft confidence with the empirical acceptance rate: `top1_prob * rolling_accept_rate`. Captures when the draft model is confident AND historically accurate.",
            "hypothesis": "Should be a strong predictor because it directly multiplies confidence with track record. High values indicate a confident draft model with a good acceptance history.",
            "collection": "Phase: Derived. Computed in `_collect_signals()` from top1_prob and rolling_accept_rate. Zero cost.",
        },
        "rolling_accept_rate": {
            "category": "Empirical / EMA",
            "status": "ACTIVE",
            "definition": "Exponential moving average of the per-step acceptance rate, updated after each verification: `alpha * step_rate + (1-alpha) * prev_rate`.",
            "hypothesis": "Acts as an adaptive baseline. When acceptance rate is trending high, the policy should be more aggressive (larger trees); when trending low, more conservative.",
            "collection": "Phase: Verify. Updated in `eagle_info.py` per-request loop via EMA. Eager mode. Negligible cost.",
        },
    }

    # Precompute per-signal stats and correlations for each signal experiment
    print("  Computing per-signal statistics...")
    signal_stats = {}  # {exp_name: {signal_name: {mean, std, min, max, median, n}}}
    signal_values = {}  # {exp_name: {signal_name: [values]}}

    for exp_name, exp_info in SIGNAL_EXPERIMENTS.items():
        steps = signal_data[exp_name]["steps"]
        signal_stats[exp_name] = {}
        signal_values[exp_name] = {}
        for sig in ALL_SIGNALS:
            vals = [s[sig] for s in steps if sig in s and s[sig] is not None]
            signal_values[exp_name][sig] = vals
            if vals:
                signal_stats[exp_name][sig] = {
                    "n": len(vals),
                    "mean": safe_mean(vals),
                    "std": safe_stdev(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "median": safe_median(vals),
                }
            else:
                signal_stats[exp_name][sig] = None

    # Precompute correlations with rolling_accept_rate and confidence
    print("  Computing correlations...")
    corr_with_rar = {}  # {exp_name: {signal_name: r}}
    corr_with_conf = {}  # {exp_name: {signal_name: r}}

    for exp_name in SIGNAL_EXPERIMENTS:
        corr_with_rar[exp_name] = {}
        corr_with_conf[exp_name] = {}
        steps = signal_data[exp_name]["steps"]
        rar_vals = [s.get("rolling_accept_rate") for s in steps]
        conf_vals = [s.get("confidence") for s in steps]

        for sig in ALL_SIGNALS:
            if sig == "rolling_accept_rate":
                corr_with_rar[exp_name][sig] = 1.0
            else:
                sig_vals = [s.get(sig) for s in steps]
                # Filter pairs where both are not None
                pairs_rar = [(sv, rv) for sv, rv in zip(sig_vals, rar_vals)
                             if sv is not None and rv is not None]
                if len(pairs_rar) >= 3:
                    corr_with_rar[exp_name][sig] = pearson_r(
                        [p[0] for p in pairs_rar], [p[1] for p in pairs_rar])
                else:
                    corr_with_rar[exp_name][sig] = float("nan")

            pairs_conf = [(sv, cv) for sv, cv in
                          zip([s.get(sig) for s in steps], conf_vals)
                          if sv is not None and cv is not None]
            if len(pairs_conf) >= 3:
                corr_with_conf[exp_name][sig] = pearson_r(
                    [p[0] for p in pairs_conf], [p[1] for p in pairs_conf])
            else:
                corr_with_conf[exp_name][sig] = float("nan")

    # Now write each signal section
    sig_index = 0
    for sig_name in ALL_SIGNALS:
        sig_index += 1
        meta = SIGNAL_META.get(sig_name, {})
        category = meta.get("category", "Unknown")
        status = meta.get("status", "ACTIVE")
        definition = meta.get("definition", "No definition available.")
        hypothesis = meta.get("hypothesis", "No hypothesis documented.")
        collection = meta.get("collection", "Not documented.")
        removal_reason = meta.get("removal_reason", "")

        w(f"### 2.{sig_index} {sig_name} -- {category} [{status}]")
        w()
        w(f"**What it is.** {definition}")
        w()
        w(f"**Hypothesis.** {hypothesis}")
        w()
        w(f"**Collection.** {collection}")
        w()
        if removal_reason:
            w(f"**Removal reason.** {removal_reason}")
            w()

        # Raw data table
        w("**Raw data across experiments:**")
        w()
        w("| Experiment | N steps | Mean | Std | Min | Max | Median |")
        w("|------------|---------|------|-----|-----|-----|--------|")
        for exp_name, exp_info in SIGNAL_EXPERIMENTS.items():
            st = signal_stats[exp_name].get(sig_name)
            label = exp_info["label"]
            if st is None:
                w(f"| {label} | N/A | N/A | N/A | N/A | N/A | N/A |")
            else:
                w(f"| {label} | {st['n']} | {fmt(st['mean'])} | {fmt(st['std'])} | {fmt(st['min'])} | {fmt(st['max'])} | {fmt(st['median'])} |")
        w()

        # Correlation with RAR
        w("**Correlation with acceptance (rolling_accept_rate):**")
        w()
        if sig_name == "rolling_accept_rate":
            w("N/A (this IS the acceptance rate signal).")
        else:
            # Compute ranks per experiment
            w("| Experiment | r(signal, RAR) | Rank (of N signals) |")
            w("|------------|----------------|---------------------|")
            for exp_name, exp_info in SIGNAL_EXPERIMENTS.items():
                r_val = corr_with_rar[exp_name].get(sig_name, float("nan"))
                if math.isnan(r_val):
                    w(f"| {exp_info['label']} | N/A | N/A |")
                else:
                    # Compute rank among signals in this experiment
                    exp_signals = exp_info["signals"]
                    r_abs_pairs = []
                    for s in exp_signals:
                        if s == "rolling_accept_rate":
                            continue
                        rv = corr_with_rar[exp_name].get(s, float("nan"))
                        if not math.isnan(rv):
                            r_abs_pairs.append((s, abs(rv)))
                    r_abs_pairs.sort(key=lambda x: x[1], reverse=True)
                    rank = "N/A"
                    for i, (s, _) in enumerate(r_abs_pairs):
                        if s == sig_name:
                            rank = f"{i+1}/{len(r_abs_pairs)}"
                            break
                    w(f"| {exp_info['label']} | {fmt(r_val)} | {rank} |")
        w()

        # Correlation with confidence
        w("**Correlation with confidence:**")
        w()
        w("| Experiment | r(signal, confidence) |")
        w("|------------|----------------------|")
        for exp_name, exp_info in SIGNAL_EXPERIMENTS.items():
            r_val = corr_with_conf[exp_name].get(sig_name, float("nan"))
            w(f"| {exp_info['label']} | {fmt(r_val) if not math.isnan(r_val) else 'N/A'} |")
        w()

        # Result and Impact
        # Auto-generate result summary based on correlations
        rar_corrs = []
        for exp_name in SIGNAL_EXPERIMENTS:
            r_val = corr_with_rar[exp_name].get(sig_name, float("nan"))
            if not math.isnan(r_val):
                rar_corrs.append(r_val)

        if not rar_corrs:
            result_text = "Insufficient data to assess predictive power."
        else:
            mean_abs_r = safe_mean([abs(r) for r in rar_corrs])
            mean_r = safe_mean(rar_corrs)
            sign_consistent = all(r > 0 for r in rar_corrs) or all(r < 0 for r in rar_corrs)
            strength = "strong" if mean_abs_r > 0.5 else "moderate" if mean_abs_r > 0.3 else "weak" if mean_abs_r > 0.1 else "negligible"
            direction = "positive" if mean_r > 0 else "negative"
            consistency = "consistent across all experiments" if sign_consistent else "inconsistent in sign across experiments"
            result_text = (
                f"Mean |r| with RAR = {fmt(mean_abs_r)}. "
                f"The relationship is {strength} and {direction}, {consistency}. "
                f"Range of r values: [{fmt(min(rar_corrs))} to {fmt(max(rar_corrs))}]."
            )

        w(f"**Result.** {result_text}")
        w()

        if "REMOVED" in status:
            impact = f"Removed from the signal set. {removal_reason.split(': ', 1)[-1] if removal_reason else 'See removal reason above.'}"
        else:
            if rar_corrs and safe_mean([abs(r) for r in rar_corrs]) > 0.3:
                impact = "Strong predictor. Recommended for upweighting in the policy confidence computation."
            elif rar_corrs and safe_mean([abs(r) for r in rar_corrs]) > 0.15:
                impact = "Moderate predictor. Keep at standard weight in the policy."
            elif rar_corrs:
                impact = "Weak predictor. Consider downweighting or monitoring for improvement with more data."
            else:
                impact = "Insufficient data."

        w(f"**Impact.** {impact}")
        w()
        w("---")
        w()

    # -----------------------------------------------------------------------
    # Section 3: Signal Interaction Analysis
    # -----------------------------------------------------------------------
    print("\nWriting Section 3: Signal Interaction Analysis...")
    w("## 3. Signal Interaction Analysis")
    w()
    w("Inter-signal Pearson correlation matrices computed on the 11sig datasets (most signals available).")
    w()

    for exp_name in ["11signals_llama8b", "11signals_deepseek8b_llamadraft"]:
        exp_info = SIGNAL_EXPERIMENTS[exp_name]
        sigs = exp_info["signals"]
        w(f"### Correlation Matrix: {exp_info['label']}")
        w()

        # Compute correlation matrix
        corr_matrix = {}
        for s1 in sigs:
            corr_matrix[s1] = {}
            for s2 in sigs:
                if s1 == s2:
                    corr_matrix[s1][s2] = 1.0
                else:
                    steps = signal_data[exp_name]["steps"]
                    pairs = [(st.get(s1), st.get(s2)) for st in steps]
                    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                    if len(pairs) >= 3:
                        corr_matrix[s1][s2] = pearson_r(
                            [p[0] for p in pairs], [p[1] for p in pairs])
                    else:
                        corr_matrix[s1][s2] = float("nan")

        # Short names for header
        short = {s: s[:12] for s in sigs}

        # Print matrix
        header = "| Signal | " + " | ".join(short[s] for s in sigs) + " |"
        sep = "|--------|" + "|".join("------" for _ in sigs) + "|"
        w(header)
        w(sep)
        for s1 in sigs:
            row = f"| {short[s1]:12s} |"
            for s2 in sigs:
                v = corr_matrix[s1][s2]
                if math.isnan(v):
                    row += " N/A  |"
                else:
                    row += f" {v:+.2f} |"
            w(row)
        w()

        # Redundancy pairs
        w(f"**Redundancy pairs (|r| > 0.7) for {exp_info['label']}:**")
        w()
        redundant = []
        seen = set()
        for i, s1 in enumerate(sigs):
            for j, s2 in enumerate(sigs):
                if j <= i:
                    continue
                r = corr_matrix[s1][s2]
                if not math.isnan(r) and abs(r) > 0.7:
                    key = tuple(sorted([s1, s2]))
                    if key not in seen:
                        seen.add(key)
                        redundant.append((s1, s2, r))
        if redundant:
            redundant.sort(key=lambda x: abs(x[2]), reverse=True)
            w("| Signal A | Signal B | r |")
            w("|----------|----------|---|")
            for s1, s2, r in redundant:
                w(f"| {s1} | {s2} | {fmt(r)} |")
        else:
            w("No pairs with |r| > 0.7 found.")
        w()

    # -----------------------------------------------------------------------
    # Section 4: Policy Performance Across Versions
    # -----------------------------------------------------------------------
    print("\nWriting Section 4: Policy Performance...")
    w("## 4. Policy Performance Across Versions")
    w()

    # Config distributions for dynamic experiments
    w("### Config Distributions")
    w()
    w("| Experiment | Mean topk | Mean steps | Mean ndt | Vanilla config (1,3,4) % | Mean confidence |")
    w("|------------|-----------|------------|----------|--------------------------|-----------------|")

    for exp_name, exp_info in SIGNAL_EXPERIMENTS.items():
        steps = signal_data[exp_name]["steps"]
        if not steps:
            w(f"| {exp_info['label']} | N/A | N/A | N/A | N/A | N/A |")
            continue

        topks = [s["chosen_topk"] for s in steps if "chosen_topk" in s]
        num_steps_vals = [s["chosen_num_steps"] for s in steps if "chosen_num_steps" in s]
        ndts = [s["chosen_num_draft_tokens"] for s in steps if "chosen_num_draft_tokens" in s]
        confs = [s["confidence"] for s in steps if "confidence" in s]

        # Vanilla config = topk=1, steps=3, ndt=4
        vanilla_count = sum(
            1 for s in steps
            if s.get("chosen_topk") == 1
            and s.get("chosen_num_steps") == 3
            and s.get("chosen_num_draft_tokens") == 4
        )
        vanilla_pct = (vanilla_count / len(steps) * 100) if steps else 0

        w(f"| {exp_info['label']} | {fmt2(safe_mean(topks))} | {fmt2(safe_mean(num_steps_vals))} | {fmt2(safe_mean(ndts))} | {fmt2(vanilla_pct)}% | {fmt(safe_mean(confs))} |")

    w()

    # Signal dilution explanation
    w("### Signal Dilution Effect")
    w()
    w("As the number of signals increases from 7 to 14 to 11, the policy's confidence")
    w("computation averages over more normalized signals. This creates a **dilution effect**:")
    w()
    w("- **7sig**: Fewer signals mean each signal has higher individual weight in the")
    w("  confidence score. The policy can respond more strongly to any single signal change.")
    w("- **14sig**: With 14 signals (many weakly correlated or redundant), the confidence")
    w("  score becomes an average of many noisy inputs. Strong signals like `rolling_accept_rate`")
    w("  get diluted by noise signals like `hidden_max` (r < 0.05). This reduces policy")
    w("  responsiveness and can cause it to default to vanilla configs more often.")
    w("- **11sig**: After pruning the 3 weakest/most redundant signals (hidden_var, hidden_max,")
    w("  entropy_gap_neg) and adding the targeted `draft_oracle_gate`, the signal set is leaner.")
    w("  Each remaining signal carries more weight, and the gate signals provide multiplicative")
    w("  (not just additive) information.")
    w()

    # Compare throughputs
    w("### Throughput Comparison (Steady-State)")
    w()
    w("| Config | SS Throughput (tok/s) | SS Accept Len | vs Vanilla Baseline |")
    w("|--------|---------------------|---------------|---------------------|")

    # Group by target model for comparison
    vanilla_baselines = {
        "Llama-3.1-8B-Instruct": "vanilla_llama8b",
        "DeepSeek-R1-Distill-Llama-8B_matched": "vanilla_deepseek8b",
        "DeepSeek-R1-Distill-Llama-8B_mismatched": "vanilla_deepseek8b_wrongdraft",
    }

    for name, exp in EXPERIMENTS.items():
        ld = log_data[name]
        tp_ss = safe_mean(ld["ss_throughputs"]) if ld["ss_throughputs"] else float("nan")
        al_ss = safe_mean(ld["ss_accept_lens"]) if ld["ss_accept_lens"] else float("nan")

        # Find baseline
        if "wrongdraft" in name or "llamadraft" in name:
            baseline_key = "vanilla_deepseek8b_wrongdraft"
        elif "deepseek" in name:
            baseline_key = "vanilla_deepseek8b"
        else:
            baseline_key = "vanilla_llama8b"

        bl_tp = safe_mean(log_data[baseline_key]["ss_throughputs"]) if log_data[baseline_key]["ss_throughputs"] else float("nan")

        if not math.isnan(tp_ss) and not math.isnan(bl_tp) and bl_tp > 0:
            delta = ((tp_ss - bl_tp) / bl_tp) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "baseline" if "vanilla" in name else "N/A"

        w(f"| {exp['label']} | {fmt2(tp_ss)} | {fmt2(al_ss)} | {delta_str} |")

    w()

    # -----------------------------------------------------------------------
    # Section 5: Recommended Signal Configuration
    # -----------------------------------------------------------------------
    print("\nWriting Section 5: Recommended Signal Configuration...")
    w("## 5. Recommended Signal Configuration")
    w()

    # Final ranking by consensus |r| with RAR
    w("### Final Signal Ranking by Consensus |r| with Rolling Accept Rate")
    w()
    w("Consensus |r| is the mean of |r| across all experiments where the signal is present.")
    w()

    ranking = []
    for sig in ALL_SIGNALS:
        if sig == "rolling_accept_rate":
            continue
        rs = []
        for exp_name in SIGNAL_EXPERIMENTS:
            r_val = corr_with_rar[exp_name].get(sig, float("nan"))
            if not math.isnan(r_val):
                rs.append(abs(r_val))
        if rs:
            consensus = safe_mean(rs)
            n_exps = len(rs)
        else:
            consensus = float("nan")
            n_exps = 0

        status = SIGNAL_META.get(sig, {}).get("status", "ACTIVE")
        ranking.append((sig, consensus, n_exps, status))

    ranking.sort(key=lambda x: x[1] if not math.isnan(x[1]) else -1, reverse=True)

    w("| Rank | Signal | Consensus |r| | # Experiments | Status | Tier |")
    w("|------|--------|----------------|---------------|--------|------|")
    for i, (sig, cons_r, n_exps, status) in enumerate(ranking):
        if math.isnan(cons_r):
            tier = "N/A"
        elif cons_r > 0.4:
            tier = "Tier 1 (high weight)"
        elif cons_r > 0.2:
            tier = "Tier 2 (standard weight)"
        elif cons_r > 0.1:
            tier = "Tier 3 (low weight)"
        else:
            tier = "Tier 4 (candidate for removal)"
        w(f"| {i+1} | {sig} | {fmt(cons_r)} | {n_exps} | {status} | {tier} |")

    w()

    # Proposed weight tiers
    w("### Proposed Weight Tiers")
    w()
    w("Based on consensus |r| and cross-experiment consistency:")
    w()
    w("**Tier 1 (High Weight, 1.5x-2x):** Signals with consensus |r| > 0.4. These are the")
    w("strongest predictors and should dominate the confidence computation.")
    w()
    tier1 = [sig for sig, cons_r, _, status in ranking
             if not math.isnan(cons_r) and cons_r > 0.4 and "REMOVED" not in status]
    if tier1:
        for s in tier1:
            w(f"- `{s}`")
    else:
        w("- (none meet this threshold)")
    w()

    w("**Tier 2 (Standard Weight, 1.0x):** Signals with consensus |r| between 0.2 and 0.4.")
    w("Solid contributors that add independent information.")
    w()
    tier2 = [sig for sig, cons_r, _, status in ranking
             if not math.isnan(cons_r) and 0.2 <= cons_r <= 0.4 and "REMOVED" not in status]
    if tier2:
        for s in tier2:
            w(f"- `{s}`")
    else:
        w("- (none meet this threshold)")
    w()

    w("**Tier 3 (Low Weight, 0.5x):** Signals with consensus |r| between 0.1 and 0.2.")
    w("Marginal contributors; keep for diversity but downweight.")
    w()
    tier3 = [sig for sig, cons_r, _, status in ranking
             if not math.isnan(cons_r) and 0.1 <= cons_r < 0.2 and "REMOVED" not in status]
    if tier3:
        for s in tier3:
            w(f"- `{s}`")
    else:
        w("- (none meet this threshold)")
    w()

    w("**Tier 4 (Candidate for Removal):** Signals with consensus |r| < 0.1 or already removed.")
    w()
    tier4 = [sig for sig, cons_r, _, status in ranking
             if (not math.isnan(cons_r) and cons_r < 0.1) or "REMOVED" in status]
    if tier4:
        for s in tier4:
            w(f"- `{s}`")
    else:
        w("- (none)")
    w()

    w("### rolling_accept_rate (Special)")
    w()
    w("`rolling_accept_rate` is both a signal and the primary target variable. It serves as")
    w("the adaptive baseline and should always be included with Tier 1 weight. Its EMA nature")
    w("provides temporal smoothing that complements the instantaneous signals above.")
    w()

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    print(f"\nWriting output to {OUT_FILE}...")
    output = "\n".join(lines) + "\n"
    with open(OUT_FILE, "w") as f:
        f.write(output)

    line_count = output.count("\n")
    print(f"Done. Output file: {OUT_FILE}")
    print(f"Total lines: {line_count}")


if __name__ == "__main__":
    main()
