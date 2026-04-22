#!/usr/bin/env python3
"""
probe_signals.py — Signal validation and threshold calibration for dynamic speculative decoding.

Planned Work #1 from CLAUDE.md: verify all signals are correctly collected and understand
their natural value ranges before tuning policy thresholds.

─── Modes ──────────────────────────────────────────────────────────────────────

1. Single-run analysis (default):
   Server must be running with --enable-dynamic-spec and signal_log_path set.

   python probe_signals.py --port 30000

2. List sweep configs:
   Print the --dynamic-spec-config JSON for every signal config you need to test.

   python probe_signals.py --list-configs

3. Run one signal config (repeat for each signal, restarting the server between runs):
   python probe_signals.py --port 30000 --signal draft_entropy
   python probe_signals.py --port 30000 --signal vanilla
   python probe_signals.py --port 30000 --signal all

4. Compare saved sweep results:
   python probe_signals.py --compare

─── Example server launch (all signals + logging, bidirectional policy) ────────

python3 -m sglang.launch_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --speculative-algorithm EAGLE3 \\
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \\
    --speculative-num-steps 5 --speculative-eagle-topk 1 \\
    --speculative-num-draft-tokens 6 \\
    --speculative-num-steps-startpoint 3 \\
    --speculative-num-draft-tokens-startpoint 4 \\
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \\
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16 \\
    --enable-dynamic-spec \\
    --dynamic-spec-config '{"draft_entropy":true,...,"signal_log_path":"probe_logs/run.jsonl"}'
"""

import argparse
import json
import os
import sys

# ── Path setup (must precede all sglang / benchmarker imports) ─────────────────
# Problem: running from repo root, Python finds `sglang/` (the git submodule dir,
# no __init__.py) as a namespace package before the editable install at
# sglang/python/sglang/.  Fix: prepend the real package directory first.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_REPO_ROOT, "sglang", "python"),          # real sglang package
    os.path.join(_REPO_ROOT, "SpecForge", "benchmarks"),   # benchmarker package
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import requests
from benchmarker import BENCHMARKS

# ── Constants ──────────────────────────────────────────────────────────────────

SIGNAL_NAMES = [
    "draft_entropy",
    "top1_prob",
    "top1_minus_top2",
    "hidden_norm",
    "path_score",
    "target_entropy",
    "entropy_gap",
    "rolling_accept_rate",
]

HIGH_CONF_THRESHOLD = 0.35
LOW_CONF_THRESHOLD = 0.65

SWEEP_CONFIGS = ["vanilla", "all"] + SIGNAL_NAMES


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe dynamic spec signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--num-samples", type=int, default=5,
                   help="Number of MT-Bench questions (max 80)")
    p.add_argument("--signal-log", default="probe_logs/run.jsonl",
                   help="JSONL signal log written by the server (single-run/analysis mode)")
    p.add_argument("--output-dir", default="probe_logs",
                   help="Directory for sweep results and per-signal logs")
    p.add_argument("--high-conf-threshold", type=float, default=HIGH_CONF_THRESHOLD)
    p.add_argument("--low-conf-threshold", type=float, default=LOW_CONF_THRESHOLD)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--signal",
        choices=SWEEP_CONFIGS,
        metavar="NAME",
        help=(
            f"Run benchmark for one signal config and save results. "
            f"Choices: {', '.join(SWEEP_CONFIGS)}"
        ),
    )
    mode.add_argument(
        "--list-configs",
        action="store_true",
        help="Print --dynamic-spec-config JSON for every sweep config, then exit.",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Load all saved sweep results from --output-dir and print comparison table.",
    )
    mode.add_argument(
        "--save-configs",
        action="store_true",
        help="Save sweep configs as probe_logs/sweep_configs.json and exit.",
    )
    mode.add_argument(
        "--get-config",
        choices=SWEEP_CONFIGS,
        metavar="NAME",
        help=(
            "Print the --dynamic-spec-config JSON for one config (for use in shell scripts). "
            f"Choices: {', '.join(SWEEP_CONFIGS)}"
        ),
    )
    mode.add_argument(
        "--reanalyse",
        action="store_true",
        help="Re-run analysis on all existing signal logs in --output-dir and save analysis JSONs.",
    )
    return p.parse_args()


# ── Server config builders ─────────────────────────────────────────────────────


def _signal_config_json(signals: List[str], log_path: str) -> str:
    cfg = {s: (s in signals) for s in SIGNAL_NAMES}
    cfg["signal_log_path"] = log_path
    return json.dumps(cfg)


def get_sweep_config(name: str, output_dir: str):
    """Return (enable_dynamic_spec: bool, dynamic_cfg_json: str|None, log_path: str|None)."""
    if name == "vanilla":
        return False, None, None
    log_path = os.path.join(output_dir, f"signals_{name}.jsonl")
    if name == "all":
        return True, _signal_config_json(SIGNAL_NAMES, log_path), log_path
    return True, _signal_config_json([name], log_path), log_path


def save_configs(output_dir: str):
    """Save all sweep configs as JSON to probe_logs/sweep_configs.json."""
    os.makedirs(output_dir, exist_ok=True)
    entries = []
    for name in SWEEP_CONFIGS:
        enable_dyn, cfg_json, log_path = get_sweep_config(name, output_dir)
        entries.append({
            "name": name,
            "enable_dynamic_spec": enable_dyn,
            "dynamic_spec_config": json.loads(cfg_json) if cfg_json else None,
            "signal_log": log_path,
        })
    out = os.path.join(output_dir, "sweep_configs.json")
    with open(out, "w") as f:
        json.dump({"configs": entries}, f, indent=2)
    print(f"Saved {len(entries)} configs to {out}")
    return out


def list_configs(output_dir: str):
    """Print server flags for each sweep config."""
    base_cmd = (
        "python3 -m sglang.launch_server \\\n"
        "    --model meta-llama/Llama-3.1-8B-Instruct \\\n"
        "    --speculative-algorithm EAGLE3 \\\n"
        "    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \\\n"
        "    --speculative-num-steps 5 --speculative-eagle-topk 1 \\\n"
        "    --speculative-num-draft-tokens 6 \\\n"
        "    --speculative-num-steps-startpoint 3 \\\n"
        "    --speculative-num-draft-tokens-startpoint 4 \\\n"
        "    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \\\n"
        "    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16"
    )
    for name in SWEEP_CONFIGS:
        enable_dyn, cfg_json, _ = get_sweep_config(name, output_dir)
        print(f"\n# ── {name} {'─'*(40 - len(name))}")
        if not enable_dyn:
            print(base_cmd)
        else:
            print(base_cmd + " \\\n"
                  f"    --enable-dynamic-spec \\\n"
                  f"    --dynamic-spec-config '{cfg_json}'")


# ── Benchmarking ───────────────────────────────────────────────────────────────


def send_flush_cache(base_url: str):
    try:
        requests.post(base_url + "/flush_cache")
    except Exception:
        pass


def run_benchmark(host: str, port: int, num_samples: int):
    """Run MTBench exactly as bench_eagle3.py does."""
    benchmarker_cls = BENCHMARKS.get("mtbench")
    benchmarker = benchmarker_cls(num_samples=num_samples)
    metrics_list = benchmarker.run(host=host, port=port, batch_size=num_samples)
    send_flush_cache(f"http://{host}:{port}")
    return metrics_list


def run_signal_config(args, name: str):
    """Run benchmark for one signal config and save metrics to output_dir."""
    os.makedirs(args.output_dir, exist_ok=True)

    enable_dyn, cfg_json, log_path = get_sweep_config(name, args.output_dir)

    print(f"\nRunning config: {name}")
    if not enable_dyn:
        print("  (vanilla — no dynamic spec)")
    else:
        print(f"  dynamic-spec-config: {cfg_json}")
    print(f"  signal-log: {log_path}")
    print(f"  num-samples: {args.num_samples}")

    # Truncate signal log so we don't mix runs
    if log_path and os.path.exists(log_path):
        open(log_path, "w").close()

    metrics_list = run_benchmark(args.host, args.port, args.num_samples)
    if not metrics_list:
        print("  [ERROR] No metrics returned.")
        return

    m = metrics_list[0]
    print(f"\n  τ (accept_length): {m.accept_length:.4f}")
    print(f"  Throughput:        {m.output_throughput:.2f} tok/s")
    print(f"  Latency:           {m.latency:.3f} s")

    # Save metrics
    result_path = os.path.join(args.output_dir, f"result_{name}.json")
    with open(result_path, "w") as f:
        json.dump(asdict(m), f, indent=2)
    print(f"\n  Saved to {result_path}")

    # Analyse signal log if available and save structured results
    if log_path and os.path.exists(log_path):
        analysis_path = os.path.join(args.output_dir, f"analysis_{name}.json")
        analyse_signal_log(log_path, args.high_conf_threshold, args.low_conf_threshold,
                           save_path=analysis_path)


# ── Signal log analysis ────────────────────────────────────────────────────────


def _mean(lst):
    return sum(lst) / len(lst) if lst else None


def load_signal_log(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_signal_series(records: List[dict]) -> Dict[str, List[float]]:
    """Extract per-step scalar for each signal. Lists are averaged across draft steps."""
    series: Dict[str, List[float]] = {s: [] for s in SIGNAL_NAMES}
    for r in records:
        for sig, key in [
            ("draft_entropy", "draft_entropies"),
            ("top1_prob", "top1_probs"),
            ("top1_minus_top2", "top1_margins"),
            ("hidden_norm", "hidden_norms"),
        ]:
            lst = r.get(key, [])
            if lst:
                series[sig].append(_mean(lst))

        for sig, key in [("path_score", "path_score"), ("target_entropy", "target_entropy")]:
            v = r.get(key)
            if v is not None:
                series[sig].append(v)

        d_ents = r.get("draft_entropies", [])
        t_ent = r.get("target_entropy")
        if d_ents and t_ent is not None:
            series["entropy_gap"].append(_mean(d_ents) - t_ent)

        # rolling_accept_rate_value is logged by decide() in dynamic_spec_config.py
        rar = r.get("rolling_accept_rate_value")
        if rar is not None:
            series["rolling_accept_rate"].append(rar)

    return series


def analyse_signal_log(
    path: str,
    high_thr: float,
    low_thr: float,
    save_path: Optional[str] = None,
) -> Optional[dict]:
    """Analyse a signal JSONL log. Prints results and returns a structured dict.

    If save_path is set, the dict is also written as JSON to that path.
    Returns None if the log is missing or empty.
    """
    if not os.path.exists(path):
        print(f"\n  [WARN] Signal log not found: {path}")
        return None

    records = load_signal_log(path)
    if not records:
        print(f"\n  [WARN] Signal log is empty: {path}")
        return None

    print(f"\n  Loaded {len(records)} verify-step records from {path}")

    series = extract_signal_series(records)
    accept_rates = [r["step_accept_rate"] for r in records if r.get("step_accept_rate") is not None]

    analysis: dict = {
        "source_log": path,
        "n_records": len(records),
        "signals": {},
        "tier_balance": {},
        "chosen_steps": [],
        "chosen_dtn": [],
        "accept_rate": {},
    }

    # ── Per-signal stats + correlation ────────────────────────────────────────
    print(f"\n  {'Signal':<22} {'N':>5} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Corr(accept)':>14}")
    print("  " + "-" * 71)
    for sig in SIGNAL_NAMES:
        vals = series[sig]
        if not vals:
            print(f"  {sig:<22} {'—':>5}  (not collected)")
            analysis["signals"][sig] = {"n": 0}
            continue
        arr = np.array(vals, dtype=float)
        corr_val = None
        corr_str = "N/A"
        if accept_rates and len(accept_rates) == len(arr):
            corr_val = float(np.corrcoef(arr, np.array(accept_rates, dtype=float))[0, 1])
            corr_str = f"{corr_val:+.3f}"
        elif accept_rates and len(accept_rates) != len(arr):
            corr_str = "(n mismatch)"
        print(
            f"  {sig:<22} {len(arr):>5} {arr.min():>8.4f} {arr.max():>8.4f}"
            f" {arr.mean():>8.4f} {arr.std():>8.4f} {corr_str:>14}"
        )
        analysis["signals"][sig] = {
            "n": len(arr),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "corr_accept": corr_val,
        }

    # ── Score → tier histogram ─────────────────────────────────────────────────
    scores = [r["score"] for r in records if r.get("score") is not None]
    if scores:
        total = len(scores)
        hi  = sum(1 for s in scores if s < high_thr)
        mid = sum(1 for s in scores if high_thr <= s < low_thr)
        lo  = sum(1 for s in scores if s >= low_thr)
        print(f"\n  Tier balance (n={total}, high<{high_thr}, low≥{low_thr}):")
        print(f"    High  score < {high_thr}:  {hi:4d} ({100*hi/total:5.1f}%)  → max (CUDA graph)")
        print(f"    Mid   {high_thr} ≤ s < {low_thr}: {mid:4d} ({100*mid/total:5.1f}%)  → baseline (eager)")
        print(f"    Low   score ≥ {low_thr}:  {lo:4d} ({100*lo/total:5.1f}%)  → minimal (eager)")
        analysis["tier_balance"] = {
            "total": total,
            "high_n": hi,  "high_pct": round(100 * hi / total, 2),
            "mid_n": mid,  "mid_pct":  round(100 * mid / total, 2),
            "low_n": lo,   "low_pct":  round(100 * lo / total, 2),
        }

    # ── chosen_steps / chosen_dtn variation ──────────────────────────────────
    steps_seen = sorted(set(r["chosen_steps"] for r in records if r.get("chosen_steps") is not None))
    dtn_seen   = sorted(set(r["chosen_dtn"]   for r in records if r.get("chosen_dtn")   is not None))
    print(f"\n  chosen_steps values seen: {steps_seen}")
    print(f"  chosen_dtn   values seen: {dtn_seen}")
    if len(steps_seen) <= 1:
        print("  WARNING: chosen_steps never varied — thresholds may need adjustment.")
    analysis["chosen_steps"] = steps_seen
    analysis["chosen_dtn"] = dtn_seen

    # ── Per-step accept rate distribution ─────────────────────────────────────
    if accept_rates:
        arr = np.array(accept_rates, dtype=float)
        print(f"\n  Per-step accept rate (n={len(arr)}): "
              f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        bins   = [0.0, 0.25, 0.5, 0.75, 1.01]
        labels = ["  0–25%", " 25–50%", " 50–75%", "75–100%"]
        bin_keys = ["0_25", "25_50", "50_75", "75_100"]
        bin_counts = {}
        for i, label in enumerate(labels):
            cnt = sum(1 for v in accept_rates if bins[i] <= v < bins[i + 1])
            bar = "█" * max(1, int(20 * cnt / len(accept_rates))) if cnt else ""
            print(f"    {label}: {cnt:4d} ({100*cnt/len(accept_rates):5.1f}%)  {bar}")
            bin_counts[bin_keys[i]] = cnt
        analysis["accept_rate"] = {
            "n": len(arr),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()),   "max": float(arr.max()),
            "bins": bin_counts,
        }

    if save_path:
        with open(save_path, "w") as _f:
            json.dump(analysis, _f, indent=2)
        print(f"\n  Analysis saved to {save_path}")

    return analysis


# ── Reanalyse saved signal logs ───────────────────────────────────────────────


def reanalyse_all(output_dir: str, high_thr: float, low_thr: float):
    """Re-run analysis on all existing signal logs in output_dir and save analysis JSONs."""
    found = 0
    for name in SWEEP_CONFIGS:
        log_path = os.path.join(output_dir, f"signals_{name}.jsonl")
        save_path = os.path.join(output_dir, f"analysis_{name}.json")
        if os.path.exists(log_path):
            found += 1
            print(f"\n{'='*50}")
            print(f"  Config: {name}")
            print(f"{'='*50}")
            analyse_signal_log(log_path, high_thr, low_thr, save_path=save_path)
        else:
            print(f"\n  [{name}] No signal log found — skipping (vanilla or not yet run).")
    print(f"\nDone. Analysed {found} config(s).")


# ── Compare saved results ──────────────────────────────────────────────────────


def compare_results(output_dir: str):
    print(f"\n{'='*96}")
    print("  SIGNAL SWEEP COMPARISON")
    print(f"{'='*96}")
    print(f"  {'Config':<22} {'τ':>6} {'tok/s':>8} {'Speedup':>9}  {'High%':>6} {'Mid%':>6} {'Low%':>5}  Steps seen")
    print(f"  {'-'*92}")

    vanilla_throughput = None
    rows = []

    for name in SWEEP_CONFIGS:
        result_path = os.path.join(output_dir, f"result_{name}.json")
        analysis_path = os.path.join(output_dir, f"analysis_{name}.json")
        result = None
        analysis = None
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
        if os.path.exists(analysis_path):
            with open(analysis_path) as f:
                analysis = json.load(f)
        rows.append((name, result, analysis))
        if name == "vanilla" and result:
            vanilla_throughput = result.get("output_throughput")

    for name, result, analysis in rows:
        if result is None:
            print(f"  {name:<22} {'(not run)'}")
            continue
        tau = result.get("accept_length", 0.0)
        thr = result.get("output_throughput", 0.0)
        speedup = f"{thr / vanilla_throughput:.3f}x" if vanilla_throughput and name != "vanilla" else "baseline"

        if analysis and analysis.get("tier_balance"):
            tb = analysis["tier_balance"]
            hi_pct  = f"{tb['high_pct']:5.1f}"
            mid_pct = f"{tb['mid_pct']:5.1f}"
            lo_pct  = f"{tb['low_pct']:5.1f}"
        else:
            hi_pct = mid_pct = lo_pct = "  —  "

        steps_seen = str(analysis["chosen_steps"]) if analysis and analysis.get("chosen_steps") else "—"

        print(f"  {name:<22} {tau:>6.3f} {thr:>8.1f} {speedup:>9}  {hi_pct:>6} {mid_pct:>6} {lo_pct:>5}  {steps_seen}")

    print(f"{'='*96}")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.list_configs:
        list_configs(args.output_dir)
        return

    if args.save_configs:
        save_configs(args.output_dir)
        return

    if args.get_config:
        _, cfg_json, _ = get_sweep_config(args.get_config, args.output_dir)
        print(cfg_json if cfg_json else "none")
        return

    if args.compare:
        compare_results(args.output_dir)
        return

    if args.reanalyse:
        reanalyse_all(args.output_dir, args.high_conf_threshold, args.low_conf_threshold)
        return

    if args.signal:
        run_signal_config(args, args.signal)
        return

    # Default: single-run analysis — benchmark + analyse signal log
    print(f"\nRunning MTBench ({args.num_samples} samples) against {args.host}:{args.port} ...")
    print(f"Signal log: {args.signal_log}")
    print("(Server must be running with --enable-dynamic-spec and signal_log_path set)\n")

    if os.path.exists(args.signal_log):
        print(f"[INFO] Existing signal log found — will append (delete manually to start fresh)")

    metrics_list = run_benchmark(args.host, args.port, args.num_samples)
    if metrics_list:
        m = metrics_list[0]
        print(f"\n{'='*45}")
        print(f"  Accept length (τ):  {m.accept_length:.4f}")
        print(f"  Throughput:         {m.output_throughput:.2f} tok/s")
        print(f"  Latency:            {m.latency:.3f} s")
        print(f"  Questions:          {m.num_questions}")
        print(f"{'='*45}")

    analyse_signal_log(args.signal_log, args.high_conf_threshold, args.low_conf_threshold)


if __name__ == "__main__":
    main()
