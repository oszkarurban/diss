# Dynamic Speculative Decoding — Implementation Reference

This document is a self-contained technical description of the dynamic speculative decoding feature added to the SGLang fork. It describes what was changed, why, how each piece works, and all assumptions made. It is intended to be pasted into a future prompt so that an LLM can understand EVERYTHING that was done.

## 1. What This Feature Does

When `--enable-dynamic-speculative-decoding` is passed to the SGLang server, the EAGLE3 speculative decoding engine dynamically adjusts three tree-shape parameters **per decode step** based on 7 runtime signals:

| Parameter | CLI arg | Controls |
|-----------|---------|----------|
| `topk` | `--speculative-eagle-topk` (start) / `--speculative-eagle-topk-max` (max) | Branching factor per tree node |
| `num_steps` | `--speculative-num-steps` (start) / `--speculative-num-steps-max` (max) | Tree depth (# draft forwards) |
| `num_draft_tokens` | `--speculative-num-draft-tokens` (start) / `--speculative-num-draft-tokens-max` (max) | Max tokens verified (post-draft pruning) |

High model confidence → speculate more aggressively (higher topk/steps/ndt).
Low confidence → speculate conservatively (lower values, down to 1).

When `--enable-dynamic-speculative-decoding` is **not** passed, every code path is identical to vanilla EAGLE3. All dynamic-spec logic is behind guards (`self._dynamic_spec_enabled` or `get_global_server_args().enable_dynamic_speculative_decoding`).

---

## 2. The 7 Signals

| # | Signal | Source phase | Where computed | How |
|---|--------|-------------|----------------|-----|
| 1 | `draft_entropy` | Draft (Phase 1) | `eagle_worker.py:draft_forward()` inside the draft loop, after `probs = softmax(logits)` | Exact: `-sum(probs * log(probs + 1e-10), dim=-1)`. Written to pre-allocated `signal_draft_entropy` buffer (works inside CUDA graph). Read from buffer after graph replay. |
| 2 | `top1_prob` | Draft extend (Phase 3, prev round) | `eagle_worker.py:_collect_signals()` | `spec_info.topk_p[:n, 0].mean().item()` — reads the top-1 probability from the previous round's extend-phase output. |
| 3 | `top1_minus_top2` | Draft extend (Phase 3, prev round) | `eagle_worker.py:_collect_signals()` | `(topk_p[:, 0] - topk_p[:, 1]).mean().item()`. Falls back to `top1_prob` when `topk == 1` (single column). |
| 4 | `hidden_norm` | Draft extend (Phase 3, prev round) | `eagle_worker.py:_collect_signals()` | `torch.norm(spec_info.hidden_states[:n], dim=-1).mean().item()` — L2 norm of draft model's hidden states. |
| 5 | `target_entropy` | Verify (Phase 2) | `eagle_info.py:verify()`, after both greedy and stochastic branches | Exact: `F.softmax(raw_logits, dim=-1)` → `-sum(p * log(p))`. Uses **unscaled** logits (before temperature) so the value is comparable regardless of sampling strategy. Stored on `self._dynamic_spec_target_entropy`, read by worker via `getattr(spec_info, ...)`. |
| 6 | `entropy_gap` | Derived | `eagle_worker.py:_collect_signals()` | `target_entropy - draft_entropy`. |
| 7 | `rolling_accept_rate` | Verify (Phase 2) | `eagle_info.py:verify()` per-request loop | Exponential moving average: `alpha * (accepted/denom) + (1-alpha) * prev_rate`, `alpha=0.3`. Stored on `Req.rolling_accept_rate` (init 1.0). Averaged across batch in `_collect_signals()`. |

### Signal flow diagram

```
Round N:
  draft_forward() ──────────────────────────────────────────────────┐
    probs = softmax(logits)                                         │
    signal_draft_entropy[:n] = -sum(probs * log(probs))    [CUDA graph or eager]
    signal_hidden_norm[:n] = norm(hidden_states)           [CUDA graph or eager]
                                                                    │
  draft() reads signal_draft_entropy buffer ──► self._last_draft_entropy
                                                                    │
  verify() ─────────────────────────────────────────────────────────┤
    target_probs = softmax(raw_logits)  [eager, always]             │
    self._dynamic_spec_target_entropy = -sum(tp * log(tp)).mean()   │
    per req: rolling_accept_rate EMA update                         │
                                                                    │
  worker.verify() reads ──► self._last_target_entropy               │
                                                                    │
  extend_after_decode() ──► updates spec_info.topk_p, hidden_states │
                                                                    ▼
Round N+1:
  _apply_dynamic_spec_config(batch):
    _collect_signals(batch) reads:
      spec_info.topk_p      ──► top1_prob, top1_minus_top2
      spec_info.hidden_states ──► hidden_norm
      self._last_draft_entropy  ──► draft_entropy
      self._last_target_entropy ──► target_entropy
      derived                   ──► entropy_gap
      batch.reqs[*].rolling_accept_rate ──► rolling_accept_rate
    policy.get_config(signals) ──► (topk, num_steps, ndt)
    swaps self.topk, self.speculative_num_steps, self.speculative_num_draft_tokens
    swaps active CUDA graph runner + attention backend
```

---

## 3. CUDA Graph Strategy

### Problem
`topk` and `num_steps` are baked into the CUDA graph at capture time. Changing them requires different graphs.

### Solution
At startup, capture a **library** of CUDA graphs for every `(topk, num_steps)` combination in `[1..topk_max] × [1..num_steps_max]`. At runtime, select the matching graph.

### What's captured per config
Each `(topk, num_steps)` config gets:
- One `EAGLEDraftCudaGraphRunner` instance (for `num_steps > 1`; `num_steps == 1` runs eager)
- One `FlashInferMultiStepDraftBackend` attention backend (has `num_steps-1` sub-backends, sized to `max_bs * topk`)

Each distinct `num_steps` value also gets:
- One `EAGLEDraftExtendCudaGraphRunner` instance (uses `num_tokens_per_bs = num_steps + 1`)

`num_draft_tokens` does NOT require separate graphs — it only affects post-graph pruning in `organize_draft_results()`.

### Capture process (`_init_cuda_graphs_dynamic`)
1. The starting config's graph is captured first (by the vanilla code path)
2. For each remaining `(topk, num_steps)` combo:
   - Temporarily swap `self.topk` and `self.speculative_num_steps`
   - Set the matching attention backend on `self.draft_model_runner`
   - Instantiate a new `EAGLEDraftCudaGraphRunner(self)` — this captures `draft_forward()` with the swapped params
   - Store in `self.cuda_graph_runners_map[(topk, num_steps)]`
3. Restore starting config values

### Signal buffers in CUDA graphs
- `signal_draft_entropy` and `signal_hidden_norm` are pre-allocated tensors on `EagleDraftInputBuffers`
- During capture, they are passed into `EagleDraftInput` and referenced by `draft_forward()`
- The entropy computation `-(probs * log(probs)).sum()` is captured as part of the graph
- After graph replay, the values sit in `self.cuda_graph_runner.buffers.signal_draft_entropy`
- **Guard**: `if spec_info.signal_draft_entropy is not None:` — when dynamic spec is off, buffers are None and the ops are never captured

### Eager path signal buffers
- Lazily allocated on first use in `draft()`: `self._eager_signal_draft_entropy`
- Assigned to `spec_info.signal_draft_entropy` before calling `draft_forward()`
- `draft_forward()` writes into them exactly as it does during CUDA graph capture

### Memory and startup cost (with `--cuda-graph-max-bs 1`)
- 1 graph per config, each ~2-3 MB
- `topk_max=4, steps_max=5` → 20 draft + 5 extend configs ≈ 75 MB
- Capture time: ~seconds per graph, total ~1-2 minutes

---

## 4. Policy Engine (`dynamic_spec.py`)

### Adaptive normalization
No hardcoded constants for signal scales. Instead, `AdaptiveSignalNormalizer` tracks running min/max for each signal:
- First 10 steps (configurable `warmup_steps`): accumulate statistics, return `None` (policy uses starting config)
- After warmup: normalize each signal to [0, 1] using observed `(value - min) / (max - min)`
- Signals where lower = more confident (`draft_entropy`, `target_entropy`, `entropy_gap`) are inverted after normalization

### Confidence score
Equal weight (1/7) for each signal. Weighted sum → clamp to [0, 1].

### Parameter mapping
Piecewise-linear interpolation:
- `confidence = 0.0` → all params at minimum (topk=1, steps=1, ndt=2)
- `confidence = 0.5` → starting config (the `--speculative-*` values)
- `confidence = 1.0` → maximum (the `--speculative-*-max` values)

### Constraints enforced
- `topk == 1` → `num_draft_tokens = num_steps + 1` (linear chain, SGLang requirement)
- `num_draft_tokens ≤ full_tree_size + 1`
- `num_draft_tokens ≥ 2`

---

## 5. Assumptions & Decisions

| # | Assumption/Decision | Rationale |
|---|---------------------|-----------|
| 1 | **Batch size = 1 only tested** (`--cuda-graph-max-bs 1`). With bs=1, per-batch signal averaging = per-request signals. Larger batch sizes work architecturally (signals averaged across requests) but are untested. | Dissertation benchmarks use bs=1 on single A100. |
| 2 | **`_collect_signals()` reads `spec_info.topk_p` which has the previous round's topk shape** (not the current round's). The topk may have just been swapped by `_apply_dynamic_spec_config()`. This is correct: we read signals from the previous step to decide the current step. Operations `[:, 0]` and `[:, 1]` are valid for any topk ≥ 1. | Documented in `_collect_signals()` with detailed shape reasoning block. |
| 3 | **`rolling_accept_rate` initializes to 1.0** (optimistic). During the warmup period, the policy returns the starting config regardless, so the initial value only affects the first few observations fed to the adaptive normalizer. | Optimistic init ensures the first few warmup observations have reasonable variance for the normalizer to learn from. |
| 4 | **`target_entropy` computed from unscaled logits** (no temperature division), even when `temperature > 0`. This makes the signal comparable across sampling strategies. | Benchmarks use `temperature=0`. For `temperature > 0`, using unscaled entropy measures the model's inherent uncertainty rather than the sampling distribution's entropy. |
| 5 | **`draft_entropy` is from the last step** of the draft loop (not averaged across steps). The draft loop runs `num_steps-1` forwards; the signal buffer is overwritten each step, so after completion it holds the last step's entropy. | The last step reflects the draft model's confidence at the deepest tree level, which is the most relevant for deciding speculation depth. |
| 6 | **`num_steps = 1` always uses eager mode** (no CUDA graph). The vanilla code already has this check: `if self.speculative_num_steps > 1: capture graph`. | With 1 step, there's only 1 token from prefill and no draft forward loop — too little computation to benefit from a CUDA graph. |
| 7 | **EMA alpha = 0.3 for rolling_accept_rate**. This gives ~70% weight to history, ~30% to the latest observation. | Moderate smoothing. Can be tuned empirically. Hardcoded for simplicity, could be made configurable. |
| 8 | **Signal weights are equal (1/7 each)**. No signal is assumed more informative than another a priori. | All normalization is learned from the model. After empirical analysis, weights can be tuned. |

---

## 6. Files Modified

### New file
| File | Purpose |
|------|---------|
| `sglang/python/sglang/srt/speculative/dynamic_spec.py` | `DynamicSpecSignals` dataclass, `AdaptiveSignalNormalizer`, `DynamicSpecPolicy` |

### Modified files (sglang submodule)

| File | What changed | Lines |
|------|-------------|-------|
| `server_args.py` | 4 new dataclass fields (`enable_dynamic_speculative_decoding`, 3× `*_max`), 4 CLI args, validation block with assertions + defaults + logging | ~510, ~2886, ~4651 |
| `eagle_worker.py` | Init: store bounds + policy + signal state + maps. `init_attention_backend()`: create backends for all (topk, steps) combos. `init_cuda_graphs()` + `_init_cuda_graphs_dynamic()`: capture graph library. `draft()`: config selection before preprocess, eager buffer allocation, entropy extraction after graph replay. `_apply_dynamic_spec_config()`: signal collection → policy → param swap → backend/runner swap → logging. `_collect_signals()`: assemble 7 signals from batch/spec_info/worker state. `draft_forward()`: entropy + hidden_norm computation into pre-allocated buffers. `verify()`: capture target_entropy from spec_info. | ~95-118, ~252-270, ~318-400, ~653-730, ~772-895, ~965-975, ~1046-1050 |
| `eagle_info.py` | `EagleDraftInput`: add `signal_draft_entropy`, `signal_hidden_norm` optional tensor fields. `verify()`: compute `target_entropy` from raw logits (works for T=0 and T>0), update `rolling_accept_rate` EMA per-req. | ~670, ~388-398, ~469-478 |
| `eagle_draft_cuda_graph_runner.py` | `EagleDraftInputBuffers`: add `signal_draft_entropy`, `signal_hidden_norm` optional fields. `__init__()`: allocate signal buffers when dynamic spec enabled. Capture: pass signal buffers into `EagleDraftInput`. | ~55-57, ~148-162, ~299-300 |
| `schedule_batch.py` | `Req` class: add `rolling_accept_rate: float = 1.0`, `spec_signal_log: List[Dict] = []` | ~815-816 |
| `io_struct.py` | `SpeculativeDecodingMetricsMixin`: add `spec_signal_log: List[List[dict]]` | ~105 |
| `scheduler_output_processor_mixin.py` | Init `spec_signal_log = []`, append per-req, pass to batch output | ~919, ~1050, ~1165 |
| `detokenizer_manager.py` | Pass-through `spec_signal_log` | ~392 |
| `multi_tokenizer_mixin.py` | Extract `spec_signal_log` by index (2 locations) | ~145, ~242 |
| `tokenizer_manager.py` | Write `meta_info["spec_signal_log"]` | ~1924 |

---

## 7. How to Use

### Server launch (dynamic spec enabled)
```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-num-steps-max 5 \
    --speculative-eagle-topk 1 --speculative-eagle-topk-max 4 \
    --speculative-num-draft-tokens 4 --speculative-num-draft-tokens-max 8 \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

### Server launch (vanilla, unchanged)
Omit `--enable-dynamic-speculative-decoding` and the `*-max` args. Everything behaves identically to upstream EAGLE3.

### Accessing signal logs
```python
response = requests.post("http://localhost:30000/generate", json={...})
meta = response.json()["meta_info"]
signal_log = meta["spec_signal_log"]  # List[Dict] per step
# Each entry: {"confidence": 0.73, "draft_entropy": 1.2, "chosen_topk": 2, ...}
```

---

## 8. What Was NOT Implemented (and Why)

| Item | Reason |
|------|--------|
| `path_score` signal (originally planned as 8th signal) | Cannot be computed inside a CUDA graph without a dedicated output buffer. The tree scores (`score_list`) are internal to `draft_forward()` and not returned from the graph replay. Removed entirely rather than using a proxy or approximation. |
| Per-request config selection with different configs for different requests in the same batch | All requests in a batch share the same CUDA graph, so they must use the same (topk, num_steps). With `--cuda-graph-max-bs 1` (bs=1), this is a non-issue. |
| Learned signal weights | Equal weights (1/7) are used. Empirical tuning should follow signal analysis from benchmark runs. |
| Exact hidden_norm from draft CUDA graph | The `signal_hidden_norm` buffer IS written inside `draft_forward()` and captured in the CUDA graph. However, `_collect_signals()` reads hidden_norm from `spec_info.hidden_states` (the extend-phase output) instead of from the signal buffer, because the extend-phase value is more recent. The draft-phase signal buffer write is retained for future use. |
