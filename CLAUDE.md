# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dissertation research on **dynamic speculative decoding** for LLM inference. The repo combines two main components:

1. **sglang/** — forked SGLang serving framework (git submodule, pinned to `oszkarurban/sglang-clean`). Contains an additive dynamic-speculative-decoding policy on top of vanilla EAGLE3 plus per-verify-step logging (draft/accepted/rejected tokens, tree topology, dynamic spec signals). See `changes_logging_1/CHANGES.md` and `chnages_dyanmicspec/DYNAMIC_SPEC_CUDA_GRAPH_CHANGES.md` for the diff inventory (the latter describes the V3 architecture that the current HEAD has returned to).
2. **SpecForge/** — upstream SGLang-ecosystem training framework for EAGLE3 draft models. Only used for its `benchmarks/bench_eagle3.py` benchmark harness.

The research question: whether runtime signals can dynamically adjust speculative decoding parameters (topk, num_steps, num_draft_tokens) per decode step to improve throughput over static EAGLE3 configurations.

## Repository Layout

```
sglang/                 # Instrumented SGLang fork (submodule, HEAD = single DOG-driven policy)
SpecForge/              # Upstream training framework (only bench_eagle3.py is used)
hpc/                    # SLURM + interactive scripts for Cambridge Wilkes3 (A100)
analysis/               # Phase-A/B signal-trace analysis (master.parquet builder, KS tests, etc.)
changes_logging_1/      # SGLang logging-instrumentation change notes
chnages_dyanmicspec/    # Dynamic-spec architecture notes (written for V3; still matches HEAD)
results/                # Benchmark JSONL outputs, organised by campaign (bench_chains, bench_dynamic, calibration, traces)
signal_data_*.json      # Legacy per-run signal dumps from test_signal_collection.py (v1 … v6 experiments)
test_signal_collection.py   # Sends requests to a running server and captures per-step signal_log + token-level data
tree_policy.py          # Auto-generated decision-tree policy (candidate alternative to DOG interpolation)
train_decision_tree.py  # Trains tree_policy.py from historical signal_data_*.json
static_eagle3_baselines.md  # Source of truth for static-config throughput SOTA per model × benchmark
V6_signal_selection_and_2x2_policy.md  # Signal-selection methodology note (Llama 2×2 policy design)
PHASE_A_MANUAL_RUN.md   # Interactive fallback for running trace collection when sbatch is impractical
```

## Model Pairs

| Target Model | Draft Model | Notes |
|-------------|------------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B` | Matched draft, best acceptance |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B` | Matched draft |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B` | **Mismatched ("LD") draft** — Llama head on DeepSeek target; weak acceptance, best regime for dynamic spec to exploit |
| `Qwen/Qwen3-8B` | `AngelSlim/Qwen3-8B_eagle3` | Matched draft |

## HPC Environment

- **Cluster**: Cambridge CSD3 Wilkes3 (A100, `ampere` partition)
- **Account**: `MASCOLO-SL2-GPU`
- **Conda env**: `sglang-dev`
- **CUDA**: 12.1
- **Submission**: `sbatch hpc/<script>.sh` or interactive via `sintr`. All scripts assume repo root at `/rds/user/ou222/hpc-work/diss` and do their own `cd`.
- **Module prep (required every shell)**: `source hpc/unload_prepare.sh` — loads `cuda/12.1`, points `CC/CXX/CUDAHOSTCXX` at the conda gcc (upstream gcc/9 module was removed; conda toolchain is now canonical), redirects `TMPDIR` and `~/.cache/flashinfer/` onto RDS (home quota is 50 GB and tight), and clears stale `flashinfer`/`tvm-ffi`/`sglang` caches on every run. Always source **after** `conda activate sglang-dev`.
- **HF cache override used by `hpc/bench_*.sh`**: `HF_HOME=/workspace/hf-cache` (node-local scratch). Login-node scripts may override this.

## Benchmark Scripts

The `hpc/bench_*.sh` family is the canonical benchmark surface. Current scripts take a **model tag** (`llama | deepseek | qwen`) as their single positional argument, pin GPU + port per tag so three tags can run concurrently on a 3× A100 box, and use the benchmark list `mtbench:80 gsm8k:100 math500:100 humaneval:100 livecodebench:100`.

| Script | Configs swept | Output dir |
|---|---|---|
| `hpc/bench_chains.sh <tag>` | Static `(3,1,4)` small chain, `(7,1,8)` deep chain | `results/bench_chains/` |
| `hpc/bench_dynamic.sh <tag>` | Dynamic, envelope `start=(3,1,4)` → `max=(7,4,8)` | `results/bench_dynamic/` |
| `hpc/calibrate_top1_dynamic.sh <tag>` | Dynamic server, 20q/dataset, per-step `spec_signal_log` for threshold calibration | `results/calibration/` |
| `hpc/bench_<tag>.sh` | Legacy 4-config sweep `(3,1,4)/(7,1,8)/(7,4,8)/(10,6,60)` on all 7 benchmarks | `results/` (flat) |
| `hpc/bench_<tag>_{chains,trees,vanilla,dynamic,dynamicsota}.sh` | Legacy per-model subsets (April 13–16 sweep) | `results/` (flat) |

Per-model pinning in the tag-dispatching scripts:

| tag | GPU_ID | Port | Target | Draft |
|---|---|---|---|---|
| llama | 0 | 30000 | Llama-3.1-8B-Instruct | sglang-EAGLE3-LLaMA3.1-Instruct-8B |
| deepseek | 1 | 30001 | DeepSeek-R1-Distill-Llama-8B | **sglang-EAGLE3-LLaMA3.1-Instruct-8B** (LD / mismatched — note this is the calibration default, not the matched draft) |
| qwen | 2 | 30002 | Qwen3-8B | Qwen3-8B_eagle3 |

A Results-jsonl row contains `batch_size, steps, topk, num_draft_tokens, metrics[{latency, output_throughput, accept_length, ...}]` — one row per `(config, benchmark)` cell.

## Analysis Pipeline

Phase-A collection → Phase-B analysis:

1. **Phase A (`hpc/run_trace_collection.sh`, `test_signal_collection.py`)** — sends 20 q/dataset through the dynamic server, records per-step signals and token-level data into `results/traces/*_signals.json`. Per-step fields (V6 layout): `top1_prob, target_top1_prob, rolling_accept_rate, top1_threshold, target_threshold, chosen_topk, chosen_num_steps, chosen_num_draft_tokens, accept_length`. For the interactive fallback (1-h `sintr`, one model at a time), see `PHASE_A_MANUAL_RUN.md`.
2. **Flatten (`analysis/extract_signal_traces.py`)** — deduplicates timestamps, emits one gzipped JSONL per `(config, model, dataset)` under `results/traces/`.
3. **Phase B (`analysis/phase_b_*.py`)** — ingests the flattened traces into `master.parquet` + `metrics_final.csv`, runs cross-cell KS stationarity, per-cell Spearman vs. `accept_length`, threshold grid search, and wasted-tail diagnostics. See `analysis/phase_b_summary.md`, `phase_b2_summary.md`, `phase_b3_eagle2_summary.md` for results.

**Finding summary (Phase B, April 2026):** the policy signal `t = clamp(DOG/0.5, 0, 1)` (DOG = `top1_prob * RAR`) has the highest median linearity (R² = 0.94) and best Spearman rank correlation with `accept_length` across the 33-cell Llama/Qwen matrix. KS up to 0.441 across models means per-model thresholds are required (Phase C scope). See `V6_signal_selection_and_2x2_policy.md` for the 2×2-policy design note that motivated the signal selection.

## Key Commands

### Launch SGLang server (static EAGLE3)

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 \
    --tp 1 --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

### Launch SGLang server (dynamic EAGLE3, current single DOG-driven policy)

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-num-steps-max 7 --speculative-eagle-topk-max 4 \
    --speculative-num-draft-tokens-max 8 \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --max-running-requests 1 \
    --tp 1 --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

### Benchmark against a running server (SpecForge harness)

```bash
python SpecForge/benchmarks/bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 --config-list 1,0,0,0 \
    --benchmark-list mtbench:80 --dtype bfloat16 \
    --skip-launch-server
```

`--config-list 1,0,0,0` runs batch-size = 1 (the only size used across this project). `--skip-launch-server` means the benchmark script connects to an already-running server; omit to let it launch and tear down its own server.

### Whole-campaign benchmark sweep (preferred)

```bash
sbatch hpc/bench_chains.sh llama     # or: bash hpc/bench_chains.sh llama  (inside sintr)
sbatch hpc/bench_dynamic.sh deepseek
sbatch hpc/bench_dynamic.sh qwen
```

## Architecture Notes

### EAGLE3 Speculative Decoding Lifecycle

Each decode iteration runs a 3-phase pipeline orchestrated by `EAGLEWorker.forward_batch_generation()` (eagle_worker.py):

```
PHASE 1: DRAFT
  for i in range(speculative_num_steps):
    select_top_k_tokens() → expand tree at depth i
    draft_model_runner.forward() → logits
    softmax → fast_topk → topk_p, topk_index
    carry forward hidden_states
  organize_draft_results() → prune to top num_draft_tokens-1 by score
  build_tree_kernel_efficient() → tree mask + retrieval indices
    ↓
PHASE 2: VERIFY
  target_worker.forward() → target logits for ALL draft tokens (tree attention via custom_mask)
  verify():
    Greedy (temp=0):  argmax(target) == draft_token → accept
    Stochastic:       tree_speculative_sampling_target_only (lossless rejection sampling)
  per-req loop: append accepted tokens, log spec_* fields, free KV for rejected tokens
    ↓
PHASE 3: DRAFT EXTEND
  draft model forward on accepted tokens → new topk_p, topk_index, hidden_states
  stored in EagleDraftInput, ready for next PHASE 1
```

All SpecForge benchmarks hardcode `temperature=0` (`benchmarker/base.py:159`), so bench runs always exercise the greedy exact-match verification path regardless of server-side settings.

### Tree Shape Parameters

| Parameter | Controls | Stored at |
|-----------|----------|-----------|
| `--speculative-num-steps` | Tree depth (# draft forwards) | `EAGLEWorker.speculative_num_steps` |
| `--speculative-eagle-topk` | Branching factor per node | `EAGLEWorker.topk` |
| `--speculative-num-draft-tokens` | Max verified tokens (prunes tree) | `EAGLEWorker.speculative_num_draft_tokens` |

Full tree before pruning = `topk + topk² + … + topk^steps`. `topk=1` → linear chain. `topk=4, steps=5` → 1364 candidates pruned to `num_draft_tokens-1` in `organize_draft_results()`.

### CUDA Graph Capture

`eagle_draft_cuda_graph_runner.py` captures all `num_steps` forwards of Phase 1 as a single CUDA graph per batch size at server startup. `num_steps` and `topk` are **baked into the graph** — dynamic speculative decoding works by capturing one draft graph per `(topk, num_steps)` combo and one target verify graph per `ndt` value across `[2, ndt_max]`, then selecting the right graph at runtime per the policy's decision.

### Dynamic Speculative Decoding (current HEAD: single DOG-driven policy)

Enabled via `--enable-dynamic-speculative-decoding` with `--speculative-{num-steps,eagle-topk,num-draft-tokens}-max` bounds. The HEAD submodule commit (`d564008d6` — "DOG=top1_prob(draft) * RollingAccRate") has reverted from the short-lived V4/top1-only/V5/V6 branches to the original V3 policy: one signal, one interpolation, cross-workload auto-calibration via the signal itself.

**Policy (`dynamic_spec.py:DynamicSpecPolicy.get_config`)** — runs per decode step:

1. Signals (one GPU→CPU sync total, see `_collect_signals`):
   - `top1_prob = topk_p[:, 0].mean()` — draft model's softmax top-1 over the batch.
   - `rolling_accept_rate` — per-request EMA (α=0.3), CPU-maintained in verify path; batch max.
2. `DOG = top1_prob × rolling_accept_rate`.
3. `t = clamp(DOG / DOG_DIVISOR, 0, 1)` with `DOG_DIVISOR = 0.5`.
4. `num_steps = interp(t, start, max)` — deeper when confident.
5. `topk = interp(1 − t, start, max)` — wider when uncertain.
6. `ndt = num_steps + 1` if `topk == 1`; else `min(ndt_max, topk*num_steps + 1)`, clamped to ≥ 2.

**Circuit breaker** (`eagle_worker._apply_dynamic_spec_config`): if **any** request has `consecutive_spec_failures ≥ 5`, override to the starting config for that step.

**Why single-signal is sufficient**: DOG auto-scales across workloads because RAR already encodes draft-target compatibility. Mean DOG: Llama ≈ 0.25 → `t ≈ 0.50` (mid-deep); Qwen ≈ 0.15 → `t ≈ 0.30` (wide-shallow); DeepSeek-LD ≈ 0.09 → `t ≈ 0.18` (wide-shallow). A top1-only variant that was tested in between saturated to the deep chain in low-RAR cells — exactly where the draft over-reports confidence — and was reverted.

| Component | File | Purpose |
|-----------|------|---------|
| Policy | `speculative/dynamic_spec.py` | `DynamicSpecPolicy.get_config()` |
| Hook | `speculative/eagle_worker.py:_apply_dynamic_spec_config` | Signal collection + backend/CUDA-graph swap + logging |
| Signal capture | `speculative/eagle_worker.py:_collect_signals` | top1_prob + RAR batch read |
| Draft CUDA graphs | `speculative/eagle_worker.py:_init_cuda_graphs_dynamic` | One graph per `(topk, num_steps)` across `[1, max]` |
| Target verify graphs | `model_executor/cuda_graph_runner.py:capture` | One graph per `ndt ∈ [2, ndt_max]` |
| CLI flags | `server_args.py` (≈L487–513, L4652–4680) | `--enable-dynamic-speculative-decoding` + `-max` variants |

`chnages_dyanmicspec/DYNAMIC_SPEC_CUDA_GRAPH_CHANGES.md` was written for the same V3 policy that HEAD has returned to and is accurate for the current backend/graph-swap path. Previous CLAUDE.md references to a `--dynamic-spec-full-logging` flag and a dual V3/V4 mode are **out of date** — no such flag exists on HEAD; there is only the single policy above.

### spec_signal_log entry (per decode step, per request)

The dynamic path appends a dict to each request's `spec_signal_log` on every step:

```
{
  "top1_prob": float,
  "rolling_accept_rate": float,
  "chosen_topk": int,
  "chosen_num_steps": int,
  "chosen_num_draft_tokens": int,
  "circuit_breaker_fired": bool,
}
```

(V6 `test_signal_collection.py` additionally post-processes `target_top1_prob`, threshold values, and `accept_length` into the per-turn traces.)

### SGLang Instrumentation Data Flow

Per-verify-step metrics flow through the SGLang pipeline:
```
eagle_info.py:verify() → schedule_batch.py (Req fields)
  → scheduler_output_processor_mixin.py (batch collection)
  → detokenizer_manager.py (IPC) → tokenizer_manager.py
  → HTTP response meta_info dict
```

Client-side: `response.json()["meta_info"]["spec_*"]`. Key fields: `spec_draft_tokens`, `spec_accepted_tokens_log`, `spec_rejected_tokens_log`, `spec_accept_index_log`, tree pointers (`spec_retrive_next_token`, `spec_retrive_next_sibling`), per-step hyperparams (`spec_logged_topk`, `spec_logged_num_steps`, …), and the `spec_signal_log` above for dynamic-server runs.

### Key Speculative Decoding Files

| File | Role |
|------|------|
| `sglang/python/sglang/srt/speculative/eagle_worker.py` | 3-phase orchestration + draft loop + `_apply_dynamic_spec_config` |
| `sglang/python/sglang/srt/speculative/eagle_info.py` | EagleVerifyInput/EagleDraftInput + `verify()` rejection sampling |
| `sglang/python/sglang/srt/speculative/eagle_utils.py` | `organize_draft_results()` pruning + `build_tree_kernel_efficient()` |
| `sglang/python/sglang/srt/speculative/spec_utils.py` | `select_top_k_tokens()` per-step tree expansion |
| `sglang/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` | Draft-phase CUDA graph capture/replay |
| `sglang/python/sglang/srt/speculative/dynamic_spec.py` | `DynamicSpecPolicy` (single DOG-driven policy) |
| `sglang/python/sglang/srt/model_executor/cuda_graph_runner.py` | Target-verify CUDA graphs (multi-ndt capture) |
| `sglang/python/sglang/srt/server_args.py` | Speculative decoding CLI args |
| `sglang/python/sglang/srt/managers/schedule_batch.py` | `Req` class with `spec_*` fields incl. `spec_signal_log`, `consecutive_spec_failures` |
| `SpecForge/benchmarks/benchmarker/base.py` | Benchmark runner (hardcodes temperature=0) |

## Benchmark Results

`static_eagle3_baselines.md` is the source of truth for static-config throughput SOTA per model × benchmark × tree shape (Llama / Qwen / DeepSeek-matched / DeepSeek-LD). It is generated from the April 13–16 2026 vanilla-EAGLE3 sweep on SGLang commit `8dc32bbb8` and includes per-cell JSONL pointers. Keep it in sync when new sweeps land.

**Notable targets the dynamic policy must meet or beat on MT-Bench 80q (bs=1, temp=0, A100):**

| Model | Static SOTA | Config | Regime for dynamic spec |
|---|---|---|---|
| Llama | 219.4 tok/s | (10,6,60) | Small margin over (7,4,8) — dynamic gains via step-cost reduction, not acceptance |
| Qwen | 178.5 tok/s | (10,6,60) | Similar — broad tree helps but not much |
| DeepSeek-matched (mtbench) | 69.7 tok/s | (3,1,4) ≈ (7,4,8) | Reasoning benchmarks prefer (10,6,60); mtbench April-13 numbers are environmental-regression-suspect (April-9 was ~118) |
| DeepSeek-LD (all benchmarks) | 59.7 tok/s | (3,1,4) | Weak draft — the target regime dynamic spec should dominate |

GSM8k is anomalous for Llama — throughput barely scales with tree size because the draft produces long chains-of-thought the target rejects at low rates. Small trees win.

See `static_eagle3_baselines.md` for the full table.
