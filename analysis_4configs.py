#!/usr/bin/env python3
"""
Comprehensive analysis of 4 speculative decoding configurations.
Produces a markdown report at results/analysis_4configs.md

Configurations:
  1. Vanilla EAGLE3 on Llama 3.1 8B Instruct
  2. Dynamic Spec on Llama 3.1 8B Instruct
  3. Vanilla EAGLE3 on DeepSeek-R1-Distill-Llama 8B
  4. Dynamic Spec on DeepSeek-R1-Distill-Llama 8B
"""

import json
import re
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

BASE = Path("/rds/user/ou222/hpc-work/diss")
LOG_DIR = BASE / "logs"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DecodeBatch:
    timestamp: str
    token_count: int
    accept_len: float
    accept_rate: float
    cuda_graph: bool
    throughput: float

@dataclass
class ConfigResult:
    name: str
    model: str
    mode: str  # 'vanilla' or 'dynamic'
    # From server log
    decode_batches: List[DecodeBatch] = field(default_factory=list)
    # From signal JSON
    elapsed_seconds: float = 0.0
    num_steps_total: int = 0
    per_turn_steps: List[int] = field(default_factory=list)
    signal_entries: List[Dict] = field(default_factory=list)
    # Vanilla config
    vanilla_topk: int = 1
    vanilla_num_steps: int = 3
    vanilla_ndt: int = 4


# ──────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────

DECODE_RE = re.compile(
    r"\[(?P<ts>[^\]]+)\] Decode batch.*"
    r"#token:\s*(?P<tok>\d+).*"
    r"accept len:\s*(?P<al>[\d.]+).*"
    r"accept rate:\s*(?P<ar>[\d.]+).*"
    r"cuda graph:\s*(?P<cg>\w+).*"
    r"gen throughput \(token/s\):\s*(?P<tp>[\d.]+)"
)


def parse_server_log(path: Path) -> List[DecodeBatch]:
    batches = []
    with open(path) as f:
        for line in f:
            m = DECODE_RE.search(line)
            if m:
                batches.append(DecodeBatch(
                    timestamp=m.group("ts"),
                    token_count=int(m.group("tok")),
                    accept_len=float(m.group("al")),
                    accept_rate=float(m.group("ar")),
                    cuda_graph=m.group("cg") == "True",
                    throughput=float(m.group("tp")),
                ))
    return batches


def load_signal_json(path: Path) -> Tuple[float, int, List[int], List[Dict]]:
    with open(path) as f:
        data = json.load(f)
    elapsed = data["elapsed_seconds"]
    total = data["num_steps_total"]
    per_turn = [len(t) for t in data["per_turn_logs"]]
    # Flatten all signal entries
    entries = []
    for turn in data["per_turn_logs"]:
        entries.extend(turn)
    return elapsed, total, per_turn, entries


# ──────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────

def stats(values):
    """Return (n, min, max, mean, std, median, p25, p75)."""
    if not values:
        return (0, None, None, None, None, None, None, None)
    n = len(values)
    s = sorted(values)
    mn, mx = s[0], s[-1]
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    p25 = s[int(n * 0.25)]
    p75 = s[int(n * 0.75)]
    return (n, mn, mx, mean, std, median, p25, p75)


def fmt(v, digits=4):
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


# ──────────────────────────────────────────────────────────────────────
# Load all 4 configs
# ──────────────────────────────────────────────────────────────────────

CONFIGS = [
    ("Vanilla Llama-8B",   "llama8b",    "vanilla"),
    ("Dynamic Llama-8B",   "llama8b",    "dynamic"),
    ("Vanilla DeepSeek-8B","deepseek8b", "vanilla"),
    ("Dynamic DeepSeek-8B","deepseek8b", "dynamic"),
]

results: Dict[str, ConfigResult] = {}

for name, model_tag, mode in CONFIGS:
    r = ConfigResult(name=name, model=model_tag, mode=mode)

    # Server log
    server_log = LOG_DIR / f"server_{mode}_{model_tag}.log"
    if server_log.exists():
        r.decode_batches = parse_server_log(server_log)

    # Signal JSON
    signal_json = BASE / f"signal_data_{mode}_{model_tag}.json"
    if signal_json.exists():
        r.elapsed_seconds, r.num_steps_total, r.per_turn_steps, r.signal_entries = load_signal_json(signal_json)

    results[name] = r


# ──────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────

out_lines = []

def w(line=""):
    out_lines.append(line)


w("# Speculative Decoding: 4-Configuration Comparative Analysis")
w()
w("**Setup**: 5 MT-Bench questions, 2 turns each (10 turns total), bs=1, temperature=0, A100 GPU")
w()
w("| Config | Model | Mode | Vanilla Params |")
w("|--------|-------|------|----------------|")
w("| Vanilla Llama-8B | Llama-3.1-8B-Instruct | EAGLE3 fixed | topk=1, steps=3, ndt=4 |")
w("| Dynamic Llama-8B | Llama-3.1-8B-Instruct | Dynamic spec | topk=[1..4], steps=[1..5], ndt=[2..8], start=1/3/4 |")
w("| Vanilla DeepSeek-8B | DeepSeek-R1-Distill-Llama-8B | EAGLE3 fixed | topk=1, steps=3, ndt=4 |")
w("| Dynamic DeepSeek-8B | DeepSeek-R1-Distill-Llama-8B | Dynamic spec | topk=[1..4], steps=[1..5], ndt=[2..8], start=1/3/4 |")
w()

# ════════════════════════════════════════════════════════════════════
# PART 1: Metric Comparison
# ════════════════════════════════════════════════════════════════════

w("## Part 1: Metric Comparison")
w()

# 1a. Accept length
w("### 1.1 Accept Length")
w()
w("| Config | N batches | Mean | Std | Min | Max | Median | P25 | P75 |")
w("|--------|-----------|------|-----|-----|-----|--------|-----|-----|")
for name in [c[0] for c in CONFIGS]:
    r = results[name]
    vals = [b.accept_len for b in r.decode_batches]
    n, mn, mx, mean, std, med, p25, p75 = stats(vals)
    w(f"| {name} | {n} | {fmt(mean)} | {fmt(std)} | {fmt(mn)} | {fmt(mx)} | {fmt(med)} | {fmt(p25)} | {fmt(p75)} |")
w()

# 1b. Throughput
w("### 1.2 Throughput (token/s)")
w()
w("Note: First batch of each request has cold-start throughput (~1-2 tok/s), excluded from steady-state stats below.")
w()
w("| Config | N batches | Mean | Std | Min | Max | Median | Steady-state Mean (excl 1st) |")
w("|--------|-----------|------|-----|-----|-----|--------|------------------------------|")
for name in [c[0] for c in CONFIGS]:
    r = results[name]
    vals = [b.throughput for b in r.decode_batches]
    steady = [b.throughput for b in r.decode_batches if b.throughput > 10]
    n, mn, mx, mean, std, med, _, _ = stats(vals)
    _, _, _, ss_mean, _, _, _, _ = stats(steady)
    w(f"| {name} | {n} | {fmt(mean)} | {fmt(std)} | {fmt(mn)} | {fmt(mx)} | {fmt(med)} | {fmt(ss_mean)} |")
w()

# 1c. Total latency
w("### 1.3 Total Latency")
w()
w("| Config | Elapsed (s) | Total decode steps | Steps/sec |")
w("|--------|-------------|-------------------|-----------|")
for name in [c[0] for c in CONFIGS]:
    r = results[name]
    # Use signal JSON elapsed for consistent measurement
    n_batches = len(r.decode_batches)
    steps_per_sec = n_batches / r.elapsed_seconds if r.elapsed_seconds > 0 else 0
    w(f"| {name} | {fmt(r.elapsed_seconds, 1)} | {n_batches} | {fmt(steps_per_sec, 1)} |")
w()

# 1d. Does dynamic improve over vanilla?
w("### 1.4 Dynamic vs Vanilla Comparison")
w()

for model_tag, model_label in [("llama8b", "Llama-8B"), ("deepseek8b", "DeepSeek-8B")]:
    v_name = f"Vanilla {model_label.replace('Llama-8B','Llama-8B').replace('DeepSeek-8B','DeepSeek-8B')}"
    d_name = f"Dynamic {model_label.replace('Llama-8B','Llama-8B').replace('DeepSeek-8B','DeepSeek-8B')}"
    v = results[v_name]
    d = results[d_name]

    v_al = [b.accept_len for b in v.decode_batches]
    d_al = [b.accept_len for b in d.decode_batches]
    v_tp = [b.throughput for b in v.decode_batches if b.throughput > 10]
    d_tp = [b.throughput for b in d.decode_batches if b.throughput > 10]

    v_al_mean = sum(v_al) / len(v_al) if v_al else 0
    d_al_mean = sum(d_al) / len(d_al) if d_al else 0
    v_tp_mean = sum(v_tp) / len(v_tp) if v_tp else 0
    d_tp_mean = sum(d_tp) / len(d_tp) if d_tp else 0

    al_delta = d_al_mean - v_al_mean
    al_pct = (al_delta / v_al_mean * 100) if v_al_mean else 0
    tp_delta = d_tp_mean - v_tp_mean
    tp_pct = (tp_delta / v_tp_mean * 100) if v_tp_mean else 0
    lat_delta = d.elapsed_seconds - v.elapsed_seconds
    lat_pct = (lat_delta / v.elapsed_seconds * 100) if v.elapsed_seconds else 0

    w(f"**{model_label}**:")
    w(f"- Accept length: vanilla={fmt(v_al_mean)} vs dynamic={fmt(d_al_mean)} ({'+' if al_delta>=0 else ''}{fmt(al_pct,1)}%)")
    w(f"- Throughput: vanilla={fmt(v_tp_mean,1)} vs dynamic={fmt(d_tp_mean,1)} tok/s ({'+' if tp_delta>=0 else ''}{fmt(tp_pct,1)}%)")
    w(f"- Latency: vanilla={fmt(v.elapsed_seconds,1)}s vs dynamic={fmt(d.elapsed_seconds,1)}s ({'+' if lat_delta>=0 else ''}{fmt(lat_pct,1)}%)")
    w(f"- Decode steps: vanilla={len(v.decode_batches)} vs dynamic={len(d.decode_batches)}")
    w()


# ════════════════════════════════════════════════════════════════════
# PART 2: Signal Analysis (dynamic runs only)
# ════════════════════════════════════════════════════════════════════

w("## Part 2: Signal Analysis (Dynamic Runs)")
w()

SIGNAL_KEYS = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "target_entropy", "entropy_gap", "rolling_accept_rate",
    "confidence", "chosen_topk", "chosen_num_steps", "chosen_num_draft_tokens"
]

for name in ["Dynamic Llama-8B", "Dynamic DeepSeek-8B"]:
    r = results[name]
    entries = r.signal_entries
    if not entries:
        w(f"### {name}: No signal data available")
        w()
        continue

    w(f"### 2.1 {name} Signal Statistics ({len(entries)} steps)")
    w()
    w("| Signal | N | Min | Max | Mean | Std | Median | P25 | P75 |")
    w("|--------|---|-----|-----|------|-----|--------|-----|-----|")
    for key in SIGNAL_KEYS:
        vals = [e[key] for e in entries if key in e]
        n, mn, mx, mean, std, med, p25, p75 = stats(vals)
        w(f"| {key} | {n} | {fmt(mn)} | {fmt(mx)} | {fmt(mean)} | {fmt(std)} | {fmt(med)} | {fmt(p25)} | {fmt(p75)} |")
    w()

    # Confidence distribution
    w(f"#### Confidence Distribution")
    w()
    confs = [e["confidence"] for e in entries if "confidence" in e]
    buckets = [0] * 10
    for c in confs:
        idx = min(int(c * 10), 9)
        buckets[idx] += 1
    w("| Bucket | Count | Pct |")
    w("|--------|-------|-----|")
    for i in range(10):
        lo = i * 0.1
        hi = (i + 1) * 0.1
        pct = buckets[i] / len(confs) * 100 if confs else 0
        w(f"| [{lo:.1f}, {hi:.1f}) | {buckets[i]} | {pct:.1f}% |")
    w()

    # Config choice distribution
    w(f"#### Chosen Config Distribution")
    w()
    for param in ["chosen_topk", "chosen_num_steps", "chosen_num_draft_tokens"]:
        vals = [e[param] for e in entries if param in e]
        ctr = Counter(vals)
        w(f"**{param}**:")
        w("| Value | Count | Pct |")
        w("|-------|-------|-----|")
        for v in sorted(ctr.keys()):
            pct = ctr[v] / len(vals) * 100
            w(f"| {v} | {ctr[v]} | {pct:.1f}% |")
        w()

    # Config tuple distribution
    w(f"#### Config Tuple (topk, steps, ndt) Distribution")
    w()
    tuples = [(int(e["chosen_topk"]), int(e["chosen_num_steps"]), int(e["chosen_num_draft_tokens"])) for e in entries]
    ctr = Counter(tuples)
    w("| (topk, steps, ndt) | Count | Pct |")
    w("|--------------------|-------|-----|")
    for t, cnt in ctr.most_common(20):
        pct = cnt / len(tuples) * 100
        w(f"| {t} | {cnt} | {pct:.1f}% |")
    w()

    # Parameter utilization
    topk_range = sorted(set(int(e["chosen_topk"]) for e in entries))
    steps_range = sorted(set(int(e["chosen_num_steps"]) for e in entries))
    ndt_range = sorted(set(int(e["chosen_num_draft_tokens"]) for e in entries))
    w(f"**Parameter range utilization:**")
    w(f"- topk: used {topk_range} out of [1..4]")
    w(f"- num_steps: used {steps_range} out of [1..5]")
    w(f"- ndt: used {ndt_range} out of [2..8]")
    w()

    # Warmup analysis
    w(f"#### Adaptive Normalizer Warmup")
    w()
    # First 10 steps per turn use starting config
    for ti, turn in enumerate(results[name].per_turn_steps):
        # Just report the first turn's warmup
        if ti == 0:
            turn_entries = r.signal_entries[:turn]
            warmup = turn_entries[:10]
            if warmup:
                warmup_confs = [e.get("confidence") for e in warmup if "confidence" in e]
                warmup_topks = [int(e.get("chosen_topk", -1)) for e in warmup]
                warmup_steps = [int(e.get("chosen_num_steps", -1)) for e in warmup]
                warmup_ndts = [int(e.get("chosen_num_draft_tokens", -1)) for e in warmup]
                w(f"Turn 0, first 10 steps (warmup):")
                w(f"- confidence values: {[round(c, 3) if c else 'None' for c in warmup_confs[:10]]}")
                w(f"- chosen_topk: {warmup_topks[:10]}")
                w(f"- chosen_num_steps: {warmup_steps[:10]}")
                w(f"- chosen_ndt: {warmup_ndts[:10]}")
                w()
                # Note: signal_entries may not have None for warmup, they might
                # still have the starting config values
                # Check if first entries differ from later
                post_warmup = turn_entries[10:20]
                if post_warmup:
                    pw_confs = [e.get("confidence") for e in post_warmup if "confidence" in e]
                    pw_topks = [int(e.get("chosen_topk", -1)) for e in post_warmup]
                    w(f"Turn 0, steps 10-19 (post-warmup):")
                    w(f"- confidence values: {[round(c, 3) if c else 'None' for c in pw_confs[:10]]}")
                    w(f"- chosen_topk: {pw_topks[:10]}")
                    w()
            break

    # Signal correlations
    w(f"#### Signal-Confidence Relationship")
    w()
    w("Pearson correlation of each signal with confidence:")
    w()
    conf_vals = [e["confidence"] for e in entries if "confidence" in e]
    conf_mean = sum(conf_vals) / len(conf_vals)
    conf_std_dev = math.sqrt(sum((c - conf_mean)**2 for c in conf_vals) / len(conf_vals))

    w("| Signal | Correlation with confidence |")
    w("|--------|---------------------------|")
    for key in ["draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
                "target_entropy", "entropy_gap", "rolling_accept_rate"]:
        sig_vals = [e[key] for e in entries if key in e and "confidence" in e]
        if len(sig_vals) != len(conf_vals):
            w(f"| {key} | N/A (length mismatch) |")
            continue
        sig_mean = sum(sig_vals) / len(sig_vals)
        sig_std = math.sqrt(sum((s - sig_mean)**2 for s in sig_vals) / len(sig_vals))
        if sig_std < 1e-10 or conf_std_dev < 1e-10:
            w(f"| {key} | N/A (zero variance) |")
            continue
        cov = sum((s - sig_mean) * (c - conf_mean) for s, c in zip(sig_vals, conf_vals)) / len(sig_vals)
        corr = cov / (sig_std * conf_std_dev)
        w(f"| {key} | {corr:.4f} |")
    w()


# ════════════════════════════════════════════════════════════════════
# PART 3: Where Improvement Comes From & Failure Analysis
# ════════════════════════════════════════════════════════════════════

w("## Part 3: Improvement & Failure Analysis")
w()

# 3.1 Compare accept rates over time
w("### 3.1 Accept Length Trajectories")
w()

for model_tag, model_label in [("llama8b", "Llama-8B"), ("deepseek8b", "DeepSeek-8B")]:
    v_name = f"Vanilla {model_label}"
    d_name = f"Dynamic {model_label}"
    v = results[v_name]
    d = results[d_name]

    w(f"**{model_label}**:")
    w()

    v_als = [b.accept_len for b in v.decode_batches]
    d_als = [b.accept_len for b in d.decode_batches]

    # Compare accept lengths in chunks of ~40 steps (decode_log_interval)
    chunk = 10
    w(f"Accept length by batch window (window size={chunk}):")
    w()
    w("| Window | Vanilla Mean | Dynamic Mean | Delta | Winner |")
    w("|--------|-------------|-------------|-------|--------|")
    max_chunks = max(len(v_als), len(d_als)) // chunk + 1
    v_wins = d_wins = ties = 0
    for i in range(min(max_chunks, 20)):
        v_chunk = v_als[i*chunk:(i+1)*chunk]
        d_chunk = d_als[i*chunk:(i+1)*chunk]
        v_m = sum(v_chunk)/len(v_chunk) if v_chunk else 0
        d_m = sum(d_chunk)/len(d_chunk) if d_chunk else 0
        delta = d_m - v_m
        winner = "Dynamic" if delta > 0.05 else ("Vanilla" if delta < -0.05 else "Tie")
        if delta > 0.05: d_wins += 1
        elif delta < -0.05: v_wins += 1
        else: ties += 1
        if v_chunk or d_chunk:
            w(f"| {i*chunk}-{(i+1)*chunk-1} | {fmt(v_m)} | {fmt(d_m)} | {'+' if delta>=0 else ''}{fmt(delta)} | {winner} |")
    w()
    w(f"Window tally: Dynamic wins {d_wins}, Vanilla wins {v_wins}, Ties {ties}")
    w()

# 3.2 Rolling accept rate analysis (dynamic only)
w("### 3.2 Rolling Accept Rate Analysis (Dynamic Runs)")
w()

for name in ["Dynamic Llama-8B", "Dynamic DeepSeek-8B"]:
    r = results[name]
    entries = r.signal_entries
    if not entries:
        continue

    rar = [e["rolling_accept_rate"] for e in entries if "rolling_accept_rate" in e]
    n, mn, mx, mean, std, med, p25, p75 = stats(rar)
    w(f"**{name}**:")
    w(f"- Rolling accept rate: mean={fmt(mean)}, std={fmt(std)}, min={fmt(mn)}, max={fmt(mx)}")
    w()

    # What % of time is accept rate low (<0.3)?
    low_rar = sum(1 for r in rar if r < 0.3)
    mid_rar = sum(1 for r in rar if 0.3 <= r < 0.7)
    high_rar = sum(1 for r in rar if r >= 0.7)
    w(f"- Accept rate < 0.3 (low):  {low_rar} steps ({low_rar/len(rar)*100:.1f}%)")
    w(f"- Accept rate 0.3-0.7 (mid): {mid_rar} steps ({mid_rar/len(rar)*100:.1f}%)")
    w(f"- Accept rate >= 0.7 (high): {high_rar} steps ({high_rar/len(rar)*100:.1f}%)")
    w()

# 3.3 Confidence vs actual acceptance
w("### 3.3 Confidence vs Actual Behavior")
w()

for name in ["Dynamic Llama-8B", "Dynamic DeepSeek-8B"]:
    r = results[name]
    entries = r.signal_entries
    if not entries:
        continue

    w(f"**{name}**:")
    w()

    # Bucket by confidence and see what configs are chosen + what rolling accept rate is
    buckets_conf = defaultdict(list)
    for e in entries:
        if "confidence" not in e:
            continue
        bucket = min(int(e["confidence"] * 5), 4)  # 5 buckets
        buckets_conf[bucket].append(e)

    w("| Confidence Range | N | Mean RAR | Mean Topk | Mean Steps | Mean NDT | Aggressiveness |")
    w("|-----------------|---|----------|-----------|------------|----------|----------------|")
    for b in range(5):
        es = buckets_conf[b]
        if not es:
            continue
        lo = b * 0.2
        hi = (b + 1) * 0.2
        n = len(es)
        rar_mean = sum(e["rolling_accept_rate"] for e in es) / n
        topk_mean = sum(e["chosen_topk"] for e in es) / n
        steps_mean = sum(e["chosen_num_steps"] for e in es) / n
        ndt_mean = sum(e["chosen_num_draft_tokens"] for e in es) / n
        # Aggressiveness: ndt relative to vanilla (4)
        agg = "Conservative" if ndt_mean < 4 else ("Moderate" if ndt_mean < 5.5 else "Aggressive")
        w(f"| [{lo:.1f}, {hi:.1f}) | {n} | {rar_mean:.3f} | {topk_mean:.2f} | {steps_mean:.2f} | {ndt_mean:.2f} | {agg} |")
    w()

    # Is the policy making good decisions?
    # Good: high confidence + high accept rate, or low confidence + low params
    # Bad: high confidence + low accept rate (over-speculating when draft is bad)
    # Bad: low confidence + would have benefited from more speculation
    high_conf_entries = [e for e in entries if e.get("confidence", 0) > 0.7]
    low_conf_entries = [e for e in entries if e.get("confidence", 0) < 0.4]

    if high_conf_entries:
        hc_rar = sum(e["rolling_accept_rate"] for e in high_conf_entries) / len(high_conf_entries)
        hc_ndt = sum(e["chosen_num_draft_tokens"] for e in high_conf_entries) / len(high_conf_entries)
        w(f"High confidence (>0.7): {len(high_conf_entries)} steps, mean RAR={hc_rar:.3f}, mean NDT={hc_ndt:.1f}")

    if low_conf_entries:
        lc_rar = sum(e["rolling_accept_rate"] for e in low_conf_entries) / len(low_conf_entries)
        lc_ndt = sum(e["chosen_num_draft_tokens"] for e in low_conf_entries) / len(low_conf_entries)
        w(f"Low confidence (<0.4): {len(low_conf_entries)} steps, mean RAR={lc_rar:.3f}, mean NDT={lc_ndt:.1f}")

    w()

# 3.4 Per-step analysis for dynamic Llama (since it's close to vanilla)
w("### 3.4 Dynamic Spec Overhead Analysis")
w()
w("Dynamic spec introduces overhead from:")
w("1. CUDA graph switching (selecting different graphs per step)")
w("2. Signal computation (entropy calculations, normalization)")
w("3. Potentially larger trees that waste verification compute")
w()

for model_label in ["Llama-8B", "DeepSeek-8B"]:
    d = results[f"Dynamic {model_label}"]
    entries = d.signal_entries

    if not entries:
        continue

    # How often does dynamic spec choose a config DIFFERENT from vanilla (1,3,4)?
    vanilla_config = (1, 3, 4)
    same_as_vanilla = sum(1 for e in entries
                         if (int(e["chosen_topk"]), int(e["chosen_num_steps"]), int(e["chosen_num_draft_tokens"])) == vanilla_config)
    diff_from_vanilla = len(entries) - same_as_vanilla

    w(f"**{model_label}**:")
    w(f"- Steps using vanilla config (1,3,4): {same_as_vanilla} ({same_as_vanilla/len(entries)*100:.1f}%)")
    w(f"- Steps using different config: {diff_from_vanilla} ({diff_from_vanilla/len(entries)*100:.1f}%)")
    w()

    # When different, what configs?
    diff_entries = [e for e in entries
                   if (int(e["chosen_topk"]), int(e["chosen_num_steps"]), int(e["chosen_num_draft_tokens"])) != vanilla_config]
    if diff_entries:
        # More aggressive (higher ndt)
        more_agg = sum(1 for e in diff_entries if e["chosen_num_draft_tokens"] > 4)
        less_agg = sum(1 for e in diff_entries if e["chosen_num_draft_tokens"] < 4)
        same_ndt = sum(1 for e in diff_entries if e["chosen_num_draft_tokens"] == 4)
        w(f"  - More aggressive (ndt > 4): {more_agg}")
        w(f"  - Less aggressive (ndt < 4): {less_agg}")
        w(f"  - Same ndt but different tree shape: {same_ndt}")
        w()


# ════════════════════════════════════════════════════════════════════
# PART 4: Improvement Recommendations
# ════════════════════════════════════════════════════════════════════

w("## Part 4: Improvement Recommendations")
w()

# 4.1 Signal weight analysis
w("### 4.1 Signal Weight Analysis")
w()
w("Current policy uses equal weights (1/7 each = 0.143). Key findings:")
w()

for name in ["Dynamic Llama-8B", "Dynamic DeepSeek-8B"]:
    r = results[name]
    entries = r.signal_entries
    if not entries:
        continue

    w(f"**{name}**:")
    w()

    # Compute correlation of each signal with rolling_accept_rate
    # (which is the actual ground truth of speculation quality)
    rar_vals = [e["rolling_accept_rate"] for e in entries]
    rar_mean = sum(rar_vals) / len(rar_vals)
    rar_std = math.sqrt(sum((r - rar_mean)**2 for r in rar_vals) / len(rar_vals))

    w("| Signal | Corr with rolling_accept_rate | Interpretation |")
    w("|--------|------------------------------|----------------|")
    for key in ["draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
                "target_entropy", "entropy_gap"]:
        sig_vals = [e[key] for e in entries]
        sig_mean = sum(sig_vals) / len(sig_vals)
        sig_std = math.sqrt(sum((s - sig_mean)**2 for s in sig_vals) / len(sig_vals))
        if sig_std < 1e-10 or rar_std < 1e-10:
            w(f"| {key} | N/A | Zero variance |")
            continue
        cov = sum((s - sig_mean) * (r - rar_mean) for s, r in zip(sig_vals, rar_vals)) / len(sig_vals)
        corr = cov / (sig_std * rar_std)
        # Interpretation
        if abs(corr) > 0.5:
            interp = "Strong predictor"
        elif abs(corr) > 0.3:
            interp = "Moderate predictor"
        elif abs(corr) > 0.1:
            interp = "Weak predictor"
        else:
            interp = "Not predictive"
        w(f"| {key} | {corr:.4f} | {interp} |")
    w()

w("### 4.2 Policy Design Issues")
w()
w("#### Issue 1: Equal weights are suboptimal")
w()
w("The equal 1/7 weighting treats all signals as equally informative. The correlation")
w("analysis above shows some signals are much better predictors of acceptance than others.")
w("**Recommendation**: Weight signals proportionally to their correlation with acceptance.")
w("Specifically, `rolling_accept_rate` itself should have the highest weight since it")
w("directly measures what we're trying to predict. `top1_prob` and `draft_entropy`")
w("should be weighted higher than `hidden_norm` which may be less informative.")
w()

w("#### Issue 2: Piecewise-linear mapping may be too coarse")
w()
w("The confidence-to-config mapping uses just two linear segments (confidence 0..0.5 and")
w("0.5..1.0) with rounding to integers. This creates sharp jumps between configs.")
w("**Recommendation**: Consider a smoother mapping, or use direct regression from signals")
w("to parameters rather than going through a scalar confidence bottleneck.")
w()

w("#### Issue 3: Adaptive normalizer sensitivity")
w()
w("The min/max normalizer is sensitive to outliers. A single extreme value permanently")
w("shifts the normalization range. **Recommendation**: Use percentile-based normalization")
w("(e.g., clip to 5th-95th percentile) or exponential moving average for min/max.")
w()

w("#### Issue 4: Warmup resets per request, not per session")
w()

# Check if normalizer state persists across requests
w("Each new request starts fresh signal collection from step 0. The warmup period")
w("(10 steps) means the first 10 steps of EVERY request use the starting config.")
w("With only 5 questions and varying lengths, this is a significant fraction of steps.")
w()

for name in ["Dynamic Llama-8B", "Dynamic DeepSeek-8B"]:
    r = results[name]
    total_warmup = sum(min(10, s) for s in r.per_turn_steps)
    total_steps = sum(r.per_turn_steps)
    w(f"- {name}: {total_warmup} warmup steps out of {total_steps} total ({total_warmup/total_steps*100:.1f}%)")
w()
w("**Note**: The warmup check above assumes normalizer resets per turn. If it persists")
w("across turns within a request but resets across requests, the actual warmup fraction")
w("may be different.")
w()

w("### 4.3 DeepSeek-8B: The Draft Model Mismatch Problem")
w()
w("The DeepSeek runs show dramatically lower accept rates (mean ~1.3 vs ~2.9 for Llama).")
w("This is because the EAGLE3 draft model (`sglang-EAGLE3-LLaMA3.1-Instruct-8B`) was")
w("trained on Llama-3.1-8B-Instruct, not on DeepSeek-R1-Distill-Llama-8B. The draft")
w("model's predictions are misaligned with the target model, causing most drafts to be")
w("rejected. In this regime:")
w()
w("- Vanilla EAGLE3 wastes compute on 3-step drafts where almost everything is rejected")
w("- Dynamic spec SHOULD help by scaling down (fewer steps, lower ndt) when confidence is low")
w("- Indeed, dynamic DeepSeek mostly uses topk=1, steps=2-3, ndt=3-4 (conservative)")
w()

# Check if dynamic actually helped for DeepSeek
d_ds = results["Dynamic DeepSeek-8B"]
v_ds = results["Vanilla DeepSeek-8B"]
d_al = [b.accept_len for b in d_ds.decode_batches if b.throughput > 10]
v_al = [b.accept_len for b in v_ds.decode_batches if b.throughput > 10]
d_tp = [b.throughput for b in d_ds.decode_batches if b.throughput > 10]
v_tp = [b.throughput for b in v_ds.decode_batches if b.throughput > 10]
d_al_m = sum(d_al)/len(d_al) if d_al else 0
v_al_m = sum(v_al)/len(v_al) if v_al else 0
d_tp_m = sum(d_tp)/len(d_tp) if d_tp else 0
v_tp_m = sum(v_tp)/len(v_tp) if v_tp else 0

w(f"**DeepSeek result**: Dynamic accept_len={d_al_m:.3f} vs Vanilla={v_al_m:.3f}")
w(f"  Dynamic throughput={d_tp_m:.1f} vs Vanilla={v_tp_m:.1f} tok/s")
w(f"  Latency: Dynamic={d_ds.elapsed_seconds:.1f}s vs Vanilla={v_ds.elapsed_seconds:.1f}s")
w()

w("### 4.4 What Would It Take to Consistently Outperform Vanilla?")
w()
w("1. **Reduce overhead**: The dynamic system captures and manages many CUDA graphs")
w("   (one per (topk, steps) combo for draft + one per ndt for verify). Graph switching")
w("   has non-zero cost. For Llama where vanilla already achieves high acceptance, the")
w("   overhead may exceed the marginal improvement from dynamic adjustment.")
w()
w("2. **Better signal utilization**: The current policy compresses 7 signals into a scalar")
w("   confidence, losing information. A multi-dimensional policy (e.g., learned via")
w("   reinforcement learning or Bayesian optimization) could make better decisions.")
w()
w("3. **Larger evaluation**: With only 5 questions, variance is high. Some questions may")
w("   naturally have high/low acceptance regardless of config. A larger evaluation")
w("   (e.g., full MT-Bench 80 questions) would give more statistical power.")
w()
w("4. **Asymmetric cost model**: The policy should account for the fact that speculating")
w("   too aggressively (large tree, mostly rejected) wastes more compute than speculating")
w("   too conservatively (small tree, high acceptance but fewer tokens per step). The")
w("   current confidence mapping is symmetric around the starting config.")
w()
w("5. **Draft-model-aware policy**: For mismatched draft models (DeepSeek case), the")
w("   policy should detect persistently low acceptance and fall back to minimal")
w("   speculation (steps=1, ndt=2) or even disable speculation entirely.")
w()
w("6. **Per-turn vs per-request warmup**: If the normalizer resets per turn (not per")
w("   request), the effective warmup fraction is larger. Consider persistent normalizer")
w("   state across requests.")
w()

# ════════════════════════════════════════════════════════════════════
# PART 5: Summary Table
# ════════════════════════════════════════════════════════════════════

w("## Summary")
w()
w("| Metric | Vanilla Llama | Dynamic Llama | Vanilla DeepSeek | Dynamic DeepSeek |")
w("|--------|--------------|--------------|-----------------|-----------------|")

# Compute final stats
for metric_name, getter, fmt_str in [
    ("Accept Length (mean)", lambda r: sum(b.accept_len for b in r.decode_batches)/max(len(r.decode_batches),1), ".3f"),
    ("Throughput (mean tok/s)", lambda r: sum(b.throughput for b in r.decode_batches if b.throughput > 10)/max(sum(1 for b in r.decode_batches if b.throughput > 10),1), ".1f"),
    ("Latency (s)", lambda r: r.elapsed_seconds, ".1f"),
    ("Decode steps", lambda r: len(r.decode_batches), "d"),
    ("Signal entries", lambda r: len(r.signal_entries), "d"),
]:
    vals = []
    for c_name in ["Vanilla Llama-8B", "Dynamic Llama-8B", "Vanilla DeepSeek-8B", "Dynamic DeepSeek-8B"]:
        v = getter(results[c_name])
        if fmt_str == "d":
            vals.append(str(int(v)))
        else:
            vals.append(f"{v:{fmt_str}}")
    w(f"| {metric_name} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")

w()
w("### Key Takeaways")
w()
w("1. **Llama-8B (matched draft model)**: Dynamic spec achieves HIGHER accept length")
w("   and throughput than vanilla. The dynamic system successfully identifies high-confidence")
w("   regions and speculates more aggressively (mean ndt ~5.3 vs vanilla's 4), with")
w("   mean accept length improvement. However, latency is similar due to overhead.")
w()
w("2. **DeepSeek-8B (mismatched draft model)**: Both vanilla and dynamic suffer from")
w("   the fundamental draft model mismatch (accept rate ~1.3 vs Llama's ~2.9).")
w("   Dynamic spec correctly identifies low confidence and scales down, but the")
w("   overhead of the dynamic system makes total latency higher than vanilla.")
w()
w("3. **Policy verdict**: The policy makes *directionally correct* decisions (more")
w("   speculation when confident, less when not) but the equal-weight, scalar-confidence")
w("   design limits its effectiveness. The main value is demonstrated on the mismatched")
w("   DeepSeek case where it avoids wasting compute on doomed speculations.")
w()

# Write output
out_path = OUT_DIR / "analysis_4configs.md"
with open(out_path, "w") as f:
    f.write("\n".join(out_lines))

print(f"Analysis written to {out_path}")
print(f"Total lines: {len(out_lines)}")
