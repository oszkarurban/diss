#!/usr/bin/env python3
"""
Dynamic Speculative Decoding Grid Search
----------------------------------------
For each configuration in GRID:
  1. Start SGLang server with dynamic spec flags (if enabled)
  2. Run bench_eagle3.py --skip-launch-server
  3. Kill server
  4. Extract metrics from the resulting JSON file
  5. Save all results to a summary JSON
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# ----------------------------------------------------------------------
# Configuration (edit these to match your setup)
# ----------------------------------------------------------------------
BENCH_EAGLE3_PATH = "./SpecForge/benchmarks/bench_eagle3.py"  # adjust as needed
SGLANG_PYTHON_PATH = "./sglang/python"  # set to "" if you don't need to add to PYTHONPATH

# MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# DRAFT = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DRAFT = "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
PORT = 30000
SERVER_STARTUP_TIMEOUT = 200
MEM_FRACTION = 0.75
CUDA_GRAPH_MAX_BS = 1
TP = 1
DTYPE = "bfloat16"

# ----------------------------------------------------------------------
# Dynamic spec signal names
# ----------------------------------------------------------------------
ALL_SIGNAL_NAMES = [
    "draft_entropy", "top1_prob", "top1_minus_top2", "hidden_norm",
    "path_score", "target_entropy", "entropy_gap", "rolling_accept_rate",
]

# ----------------------------------------------------------------------
# Grid configurations
# ----------------------------------------------------------------------
GRID: List[Dict[str, Any]] = [
    {
        "name": "00_baseline_static",
        "description": "Static EAGLE3 — steps=3, dtn=4 always",
        "enable_dynamic_spec": False,
    },
    {
        "name": "01_rolling_only",
        "description": "Rolling accept rate only",
        "enable_dynamic_spec": True,
        "signals": {"rolling_accept_rate": 1.0},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "02_entropy_gap_only",
        "description": "Entropy gap (draft minus target) only",
        "enable_dynamic_spec": True,
        "signals": {"entropy_gap": 1.0},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "03_draft_entropy_only",
        "description": "Draft model entropy only",
        "enable_dynamic_spec": True,
        "signals": {"draft_entropy": 1.0},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "04_path_score_only",
        "description": "Draft tree path score only",
        "enable_dynamic_spec": True,
        "signals": {"path_score": 1.0},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "05_target_entropy_only",
        "description": "Target model entropy only",
        "enable_dynamic_spec": True,
        "signals": {"target_entropy": 1.0},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "06_all_equal",
        "description": "All 8 signals, equal weight 1.0",
        "enable_dynamic_spec": True,
        "signals": {s: 1.0 for s in ALL_SIGNAL_NAMES},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "07_entropy_combo",
        "description": "Draft + target entropy + gap (w=1.5) + rolling (w=0.5)",
        "enable_dynamic_spec": True,
        "signals": {
            "draft_entropy": 1.0, "target_entropy": 1.0,
            "entropy_gap": 1.5,   "rolling_accept_rate": 0.5,
        },
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "08_entropy_gap_rolling",
        "description": "Entropy gap + rolling accept rate, equal weight",
        "enable_dynamic_spec": True,
        "signals": {"entropy_gap": 1.0, "rolling_accept_rate": 1.0},
        "high_conf_threshold": 0.35, "low_conf_threshold": 0.65,
    },
    {
        "name": "09_tight_thresholds",
        "description": "Entropy gap + rolling, tight thresholds [0.25, 0.55]",
        "enable_dynamic_spec": True,
        "signals": {"entropy_gap": 1.0, "rolling_accept_rate": 1.0},
        "high_conf_threshold": 0.25, "low_conf_threshold": 0.55,
    },
    {
        "name": "10_wide_thresholds",
        "description": "Entropy gap + rolling, wide thresholds [0.45, 0.75]",
        "enable_dynamic_spec": True,
        "signals": {"entropy_gap": 1.0, "rolling_accept_rate": 1.0},
        "high_conf_threshold": 0.45, "low_conf_threshold": 0.75,
    },
    {
        "name": "11_conservative",
        "description": "All signals, very wide [0.55, 0.85] — rarely shrinks tree",
        "enable_dynamic_spec": True,
        "signals": {s: 1.0 for s in ALL_SIGNAL_NAMES},
        "high_conf_threshold": 0.55, "low_conf_threshold": 0.85,
    },
]

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def build_server_command(cfg: Dict[str, Any]) -> List[str]:
    """Build the server command for a given grid entry."""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model", MODEL,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", DRAFT,
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        "--mem-fraction-static", str(MEM_FRACTION),
        "--cuda-graph-max-bs", str(CUDA_GRAPH_MAX_BS),
        "--tp", str(TP),
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--dtype", DTYPE,
    ]
    if not cfg.get("enable_dynamic_spec"):
        return cmd

    cmd.append("--enable-dynamic-spec")
    override: Dict[str, Any] = {}
    signals = cfg.get("signals", {})
    for sig in ALL_SIGNAL_NAMES:
        if sig in signals:
            override[sig] = True
            override[f"weight_{sig}"] = float(signals[sig])
        else:
            override[sig] = False
    override["high_conf_threshold"] = cfg.get("high_conf_threshold", 0.35)
    override["low_conf_threshold"] = cfg.get("low_conf_threshold", 0.65)
    cmd.extend(["--dynamic-spec-config", json.dumps(override)])
    return cmd


def build_bench_command(cfg: Dict[str, Any], benchmark_list: str, output_dir: str) -> List[str]:
    """Build the bench_eagle3.py command (skip-launch-server)."""
    return [
        "python3", BENCH_EAGLE3_PATH,
        "--model-path", MODEL,
        "--port", str(PORT),
        "--config-list", "1,0,0,0",   # dummy, not used because server already running
        "--benchmark-list", benchmark_list,
        "--dtype", DTYPE,
        "--skip-launch-server",
        "--output-dir", output_dir,
        "--name", cfg["name"],
    ]


def wait_for_server(port: int, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Wait for the server health endpoint."""
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def start_server(cfg: Dict[str, Any], log_dir: str) -> Optional[subprocess.Popen]:
    """Start the server and return the process."""
    cmd = build_server_command(cfg)
    log_path = Path(log_dir) / f"server_{cfg['name']}.log"

    env = os.environ.copy()
    if SGLANG_PYTHON_PATH:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{SGLANG_PYTHON_PATH}:{existing}" if existing else SGLANG_PYTHON_PATH

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    log_fh.close()
    print(f"  server pid={proc.pid}  log={log_path}")

    if not wait_for_server(PORT):
        print(f"  ERROR: server did not start in {SERVER_STARTUP_TIMEOUT}s")
        stop_server(proc)
        return None
    print("  server ready.")
    return proc


def stop_server(proc: subprocess.Popen):
    """Kill the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    time.sleep(3)


def run_benchmark(cfg: Dict[str, Any], benchmark_list: str, output_dir: str, log_dir: str) -> Optional[Dict]:
    """Run bench_eagle3.py and return the parsed JSON result."""
    cmd = build_bench_command(cfg, benchmark_list, output_dir)
    log_path = Path(log_dir) / f"bench_{cfg['name']}.log"

    env = os.environ.copy()
    if SGLANG_PYTHON_PATH:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{SGLANG_PYTHON_PATH}:{existing}" if existing else SGLANG_PYTHON_PATH

    print(f"  bench cmd: {' '.join(cmd)}")
    print(f"  bench log: {log_path}")

    with open(log_path, "w") as log_fh:
        ret = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)

    if ret.returncode != 0:
        print(f"  ERROR: bench_eagle3.py exited {ret.returncode} — see {log_path}")
        return None

    # Find the result file bench_eagle3.py just wrote
    result_files = sorted(
        Path(output_dir).glob(f"{cfg['name']}_results_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not result_files:
        print(f"  ERROR: no result file found for {cfg['name']} in {output_dir}")
        return None

    result_file = result_files[0]
    print(f"  result: {result_file}")

    with open(result_file) as f:
        data = json.load(f)

    return data


def extract_metrics(raw: Dict, bench_key: str) -> Dict[str, float]:
    """Extract throughput and acceptance length from the JSON."""
    entries = raw.get(bench_key, [])
    if not entries:
        return {}
    entry = entries[0]
    metrics_list = entry.get("metrics", [])
    if not metrics_list:
        return {}
    # Average across metrics (should be only one, but just in case)
    throughputs = [m.get("throughput", 0) for m in metrics_list if "throughput" in m]
    accept_lengths = [m.get("accept_length", 0) for m in metrics_list if "accept_length" in m]
    return {
        "throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
        "accept_length": sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0,
        "num_samples": entry.get("num_samples"),
    }


def print_summary(results: List[Dict], benchmark_list: str):
    """Print a summary table with speedup relative to baseline."""
    valid = [r for r in results if "metrics" in r and r["metrics"]]
    if not valid:
        print("No valid results.")
        return

    baseline = next((r for r in valid if r["name"] == "00_baseline_static"), None)
    baseline_tps = baseline["metrics"]["throughput"] if baseline else None

    print("\n" + "=" * 100)
    print(f"GRID SEARCH RESULTS — {benchmark_list}")
    print("  τ       = total_output_tokens / total_verify_steps")
    print("  speedup = config_throughput / baseline_throughput")
    print("=" * 100)
    print(f"{'Config':<32} {'Speedup':>9} {'τ':>8} {'TPS':>9}  Description")
    print("-" * 100)

    for r in sorted(valid, key=lambda x: -(x["metrics"].get("throughput", 0))):
        m = r["metrics"]
        tps = m.get("throughput", 0)
        tau = m.get("accept_length", 0)
        speedup = f"{tps / baseline_tps:.3f}x" if baseline_tps else "N/A"
        print(f"{r['name']:<32} {speedup:>9} {tau:>8.3f} {tps:>9.2f}  {r.get('description', '')[:40]}")

    print("\nInterpretation:")
    print("  τ = 1.0  → no speculation benefit (pure autoregressive)")
    print("  τ = 2.0  → each verify step accepted 1 draft token + 1 bonus on average")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dynamic spec grid search")
    parser.add_argument("--benchmark-list", type=str, default="mtbench:80",
                        help="e.g., mtbench:80, gsm8k:200")
    parser.add_argument("--output-dir", type=str, default="./gridsearch_results",
                        help="Directory where bench_eagle3.py saves its JSON files")
    parser.add_argument("--log-dir", type=str, default="./gridsearch_logs",
                        help="Directory for server and bench logs")
    parser.add_argument("--output-json", type=str, default="dynspec_gridsearch.json",
                        help="Output file for aggregated results")
    parser.add_argument("--configs", nargs="+",
                        help="Config names to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print server and bench commands without running")
    args = parser.parse_args()

    # Validate bench_eagle3.py
    if not os.path.isfile(BENCH_EAGLE3_PATH):
        print(f"ERROR: bench_eagle3.py not found at {BENCH_EAGLE3_PATH}")
        sys.exit(1)

    # Select configs to run
    configs = GRID
    if args.configs:
        configs = [c for c in GRID if c["name"] in args.configs]
        if not configs:
            print("No matching configs. Available:")
            for c in GRID:
                print(f"  {c['name']}")
            sys.exit(1)

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Estimate runtime (rough)
    nq = args.benchmark_list.split(":")[1] if ":" in args.benchmark_list else "80"
    est = len(configs) * (SERVER_STARTUP_TIMEOUT // 3 + int(nq) * 8) // 60
    print(f"Dynamic Spec Grid Search")
    print(f"  configs   : {len(configs)}")
    print(f"  benchmark : {args.benchmark_list}")
    print(f"  est. time : ~{est} minutes")
    print()

    results = []
    for cfg in configs:
        print(f"\n{'='*65}")
        print(f"  {cfg['name']}")
        print(f"  {cfg['description']}")
        if cfg.get("signals"):
            print(f"  signals   : {cfg['signals']}")
            print(f"  thresholds: [{cfg.get('high_conf_threshold', 0.35)}, "
                  f"{cfg.get('low_conf_threshold', 0.65)}]")

        if args.dry_run:
            print("  SERVER: " + " ".join(build_server_command(cfg)))
            print("  BENCH:  " + " ".join(build_bench_command(cfg, args.benchmark_list, args.output_dir)))
            results.append({"name": cfg["name"], "dry_run": True})
            continue

        # Start server
        proc = start_server(cfg, args.log_dir)
        if proc is None:
            results.append({"name": cfg["name"], "error": "server_start_failed"})
            continue

        # Run benchmark
        raw = run_benchmark(cfg, args.benchmark_list, args.output_dir, args.log_dir)
        if raw is None:
            results.append({"name": cfg["name"], "error": "benchmark_failed"})
            stop_server(proc)
            continue

        # Extract metrics
        bench_key = args.benchmark_list.split(":")[0]  # e.g., "mtbench"
        metrics = extract_metrics(raw, bench_key)
        print(f"  τ={metrics.get('accept_length', 0):.3f}  tps={metrics.get('throughput', 0):.2f}")

        results.append({
            "name": cfg["name"],
            "description": cfg["description"],
            "config": cfg,
            "metrics": metrics,
            "raw": raw,  # optional, may be large; remove if not needed
        })

        # Kill server
        stop_server(proc)

        # Save incrementally
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  → saved to {args.output_json}")

    if not args.dry_run:
        print_summary(results, args.benchmark_list)
        print(f"\nFull results  : {args.output_json}")
        print(f"Benchmark JSON : {args.output_dir}/")
        print(f"Logs           : {args.log_dir}/")


if __name__ == "__main__":
    main()