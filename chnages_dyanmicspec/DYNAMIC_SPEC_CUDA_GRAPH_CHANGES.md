# Dynamic Speculative Decoding — CUDA Graph Implementation

## Overview

This document describes all changes made to the SGLang codebase to implement dynamic speculative decoding with full CUDA graph support. The feature dynamically adjusts all three EAGLE3 tree-shape parameters (`topk`, `num_steps`, `num_draft_tokens`) per decode step based on a single runtime signal (`draft_oracle_gate = top1_prob × rolling_accept_rate`), while maintaining CUDA graph acceleration.

**Key result**: Dynamic EAGLE3 (start=3,1,4 max=7,4,8) achieves +4-8% throughput over static EAGLE3 7,4,8 across three model pairs (Llama, DeepSeek, Qwen) by reducing draft cost when draft quality is low.

**Policy signal**: `draft_oracle_gate` — the sole signal driving config selection. 8 signals are logged for analysis, 20 more are defined but disabled (0.0).

**Files changed**: 13 files, +1553/-45 lines.

---

## File-by-file changes

### 1. `flashinfer_backend.py` (+12 lines)

**Problem solved**: When multiple target verify CUDA graphs are captured at the same batch size but different `num_draft_tokens` (ndt), each capture creates new `BatchPrefillWithPagedKVCacheWrapper` objects. These wrappers contain internal C++ state (plan buffers) that the CUDA graph kernels read from at replay time. In vanilla SGLang, only one graph exists per batch size, so `prefill_cuda_graph_metadata[bs]` works. With multiple ndt values, each capture overwrites the previous wrapper, leaving earlier graphs pointing to stale/freed internal buffers. This caused `cudaErrorIllegalAddress` crashes.

**Lines 619-624 (`init_forward_metadata_capture_cuda_graph`, TARGET_VERIFY branch)**:
```python
ndt = getattr(spec_info, "draft_token_num", None)
if ndt is not None:
    self.prefill_cuda_graph_metadata[(bs, ndt)] = prefill_wrappers
```
During CUDA graph capture for target verify mode, additionally store the FlashInfer wrappers under a `(bs, ndt)` tuple key. The plain `bs` key is still written for backward compatibility with vanilla (non-dynamic) mode.

**Lines 713-717 (`init_forward_metadata_replay_cuda_graph`, TARGET_VERIFY branch)**:
```python
ndt = getattr(spec_info, "draft_token_num", None)
metadata_key = (bs, ndt) if ndt is not None and (bs, ndt) in self.prefill_cuda_graph_metadata else bs
```
At replay time, look up the wrappers by `(bs, ndt)` first. This ensures the graph replays with the exact wrappers it was captured with. Falls back to plain `bs` key for vanilla mode or if the tuple key doesn't exist.

---

### 2. Pipeline files (6 files, +12 lines total)

These files pass the new `spec_signal_log` field through SGLang's output pipeline. Each change is a single line following the existing pattern for `spec_retrive_next_token`, `spec_retrive_next_sibling`, etc.

**`schedule_batch.py` (Req class, +4 lines)**:
```python
self.rolling_accept_rate: float = 1.0  # EMA, init optimistic
self.spec_signal_log: List[Dict] = []  # per-step signal + config log
```
- `rolling_accept_rate`: Exponential moving average of draft token acceptance, initialized optimistically to 1.0. Updated by `eagle_info.py:verify()` after each verification step. Used as one of 7 signals for the dynamic policy.
- `spec_signal_log`: Accumulates per-step signal values and chosen configuration for each request. Returned to the client via HTTP response `meta_info["spec_signal_log"]` for analysis.

**`io_struct.py` (+1 line)**: Adds `spec_signal_log: List[List[dict]]` to `SpeculativeDecodingMetricsMixin` — the batch-level output structure (shape: `[req][step]`).

**`scheduler_output_processor_mixin.py` (+3 lines)**: Initializes `spec_signal_log = []`, appends `req.spec_signal_log` per request, and passes it to the batch output constructor.

**`detokenizer_manager.py` (+1 line)**: Passes `spec_signal_log` through the IPC boundary from scheduler to detokenizer.

**`multi_tokenizer_mixin.py` (+2 lines)**: Extracts `spec_signal_log` by index for both `BatchTokenIDOutput` and `BatchStrOutput` paths.

**`tokenizer_manager.py` (+1 line)**: Writes `meta_info["spec_signal_log"] = recv_obj.spec_signal_log[i]` to the HTTP response.

---

### 3. `cuda_graph_runner.py` (+146 lines, -20 lines)

This is the target model's CUDA graph runner. In vanilla SGLang, it captures one graph per batch size. For dynamic spec, it captures one graph per `(batch_size, ndt)` pair so that each graph processes exactly the right number of token positions — no stale `out_cache_loc` entries that would corrupt the KV cache.

**Lines 544-566 (`__init__`, buffer initialization)**:
```python
self._is_dynamic_spec = (sa.enable_dynamic_speculative_decoding
                         and sa.speculative_num_draft_tokens_max is not None)
if self._is_dynamic_spec:
    self.num_tokens_per_bs = sa.speculative_num_draft_tokens_max
    self._ndt_min = min(sa.speculative_num_draft_tokens, 2)
    self._ndt_max = sa.speculative_num_draft_tokens_max
```
- `_is_dynamic_spec`: Feature flag checked throughout the file via `getattr(self, "_is_dynamic_spec", False)` to avoid AttributeError on non-spec models.
- `num_tokens_per_bs = ndt_max`: Buffers (input_ids, positions, out_cache_loc, etc.) are allocated for the largest possible ndt. All per-ndt graphs share these buffers but each graph only accesses its own slice `[:bs*ndt]`.
- `_ndt_min = min(starting_ndt, 2)`: Lower bound. The policy enforces `ndt >= 2` (root + at least 1 candidate).
- `_ndt_max = ndt_max`: Upper bound from server args.

**Lines 674-704 (`can_run`, eligibility check)**:
```python
if getattr(self, "_is_dynamic_spec", False):
    ndt = getattr(forward_batch.spec_info, "draft_token_num", None)
    graph_key = (cuda_graph_bs, ndt)
    ...
    is_bs_supported = (cuda_graph_bs <= self.max_bs
                       and ndt is not None
                       and self._ndt_min <= ndt <= self._ndt_max)
```
Constructs the graph key as `(bs, ndt)` tuple and checks both batch size and ndt range. If ndt is outside the captured range, falls back to eager mode.

**Lines 796-858 (`capture`, the capture loop)**:
```python
if getattr(self, "_is_dynamic_spec", False):
    for ndt in range(self._ndt_min, self._ndt_max + 1):
        self.num_tokens_per_bs = ndt
        with patch_model(..., num_tokens=bs * ndt, ...):
            graph, output_buffers = self.capture_one_batch_size(bs, forward, stream_idx)
            key = (bs, ndt)
            self.graphs[key] = graph
            self.output_buffers[key] = output_buffers
    self.num_tokens_per_bs = self._ndt_max  # restore for buffer consistency
```
For each batch size, iterates over all ndt values `[ndt_min..ndt_max]`. Before each capture, sets `self.num_tokens_per_bs = ndt` so that `capture_one_batch_size` creates a ForwardBatch with exactly `bs * ndt` tokens. The graph records model forward for that exact token count. After the loop, restores `num_tokens_per_bs` to max for buffer sizing.

**Lines 1155-1175 (`replay_prepare`, pre-replay setup)**:
```python
if forward_batch.forward_mode.is_target_verify():
    logger.info(f"[DYNSPEC CG-REPLAY] ...")
...
self._replay_ndt = getattr(forward_batch.spec_info, "draft_token_num", None)
```
Logs diagnostic info for target verify replays. Stores `_replay_ndt` for use in the subsequent `replay()` call.

**Lines 1188-1205 (`replay`, graph selection)**:
```python
ndt = getattr(self, "_replay_ndt", None)
if ndt is not None and getattr(self, "_is_dynamic_spec", False):
    graph_key = (self.bs, ndt)
else:
    graph_key = self.bs
self.graphs[graph_key].replay()
output = self.output_buffers[graph_key]
```
Selects the graph by `(padded_bs, ndt)` tuple when dynamic spec is active.

**Lines 1236-1270 (`get_spec_info`, capture-time spec info)**:
```python
capture_ndt = self.num_tokens_per_bs  # set per-ndt in capture loop
capture_steps = sa.speculative_num_steps
capture_topk = sa.speculative_eagle_topk
```
Uses `self.num_tokens_per_bs` (which varies per ndt in the capture loop) instead of the old `server_args.speculative_num_draft_tokens`. The `spec_steps` and `topk` use starting values — these set up FlashInfer tree metadata during capture, but `init_forward_metadata_replay_cuda_graph` re-writes the metadata from the actual spec_info at replay time. What matters is that `draft_token_num` matches the number of positions the graph processes.

---

### 4. `server_args.py` (+70 lines)

**Lines 509-513 (new dataclass fields)**:
```python
enable_dynamic_speculative_decoding: bool = False
speculative_num_steps_max: Optional[int] = None
speculative_eagle_topk_max: Optional[int] = None
speculative_num_draft_tokens_max: Optional[int] = None
```
The master switch and upper bounds for the three tree-shape parameters.

**Lines 2882-2918 (validation block)**: Auto-fills `None` max values with starting values (so `--enable-dynamic-speculative-decoding` alone works without explicit max args). Asserts `max >= starting` for all three params. Logs the full configuration range.

**Lines 4647-4681 (argparse)**: Registers `--enable-dynamic-speculative-decoding`, `--speculative-num-steps-max`, `--speculative-eagle-topk-max`, `--speculative-num-draft-tokens-max`.

---

### 5. `dynamic_spec_config.py` (-318 lines, deleted)

Removes the old `DynamicSpecConfig` class (entirely commented out). This was an earlier prototype that used a different signal collection approach with `_is_capturing()` guards and hardcoded normalization. Replaced by `dynamic_spec.py` (the new policy engine, not part of this diff — it was added as a new file in the submodule).

---

### 6. `eagle_draft_cuda_graph_runner.py` (+54 lines)

**Lines 51-56 (`EagleDraftInputBuffers` dataclass)**:
```python
signal_draft_entropy: Optional[torch.Tensor] = None
signal_hidden_norm: Optional[torch.Tensor] = None
signal_hidden_cosine: Optional[torch.Tensor] = None
signal_hidden_projection: Optional[torch.Tensor] = None
signal_hidden_entropy: Optional[torch.Tensor] = None
prev_hidden_for_cosine: Optional[torch.Tensor] = None
```
Optional signal output fields on the draft CUDA graph buffer dataclass.

**Lines 148-184 (buffer allocation — ALL DISABLED)**:
All 6 signal buffers are set to `None`. The commented-out code shows what would be allocated if re-enabled. No active signal requires draft-phase GPU computation — the policy uses only `top1_prob` (from the extend phase's `topk_p`) and `rolling_accept_rate` (pure CPU). With buffers `= None`, `draft_forward()` auto-skips signal writes (gated by `if spec_info.signal_draft_entropy is not None`).

---

### 7. `eagle_info.py` (+139 lines)

**Signal computation in `verify()` — only `target_top1_prob` is active**:
```python
if get_global_server_args().enable_dynamic_speculative_decoding:
    with torch.no_grad():
        raw_logits = logits_output.next_token_logits
        t_probs = F.softmax(raw_logits, dim=-1)
        t_topk = torch.topk(t_probs, min(2, t_probs.shape[-1]), dim=-1).values
        self._dynamic_spec_target_top1_prob = t_topk[:, 0].mean()
```
Computes target model top-1 probability from raw logits. Stored as a scalar GPU tensor (deferred `.item()` in `_collect_signals()`). **Disabled signals**: target_entropy, target_top1_gap, target_varentropy, kl_approx_target_draft, target_draft_agree, all target_hidden_* — all commented out with `TODO(DISABLED)` tags.

**accept_index global→local conversion**: Bugfix for bs>1 — converts global indices to local per-request indices.

**Rolling accept rate EMA** (α=0.3):
```python
denom = max(self.draft_token_num - 1, 1)
step_rate = accepted_draft_tokens / denom
req.rolling_accept_rate = alpha * step_rate + (1 - alpha) * req.rolling_accept_rate
```

**Rolling accept length EMA** (α=0.3):
```python
req.rolling_accept_length = alpha * accepted_draft_tokens + (1 - alpha) * req.rolling_accept_length
```

**Consecutive spec failures** (circuit breaker counter):
```python
if accepted_draft_tokens == 0:
    req.consecutive_spec_failures += 1
else:
    req.consecutive_spec_failures = 0
```

**`EagleDraftInput` signal fields**: 6 optional tensor fields (signal_draft_entropy, signal_hidden_norm, signal_hidden_cosine, signal_hidden_projection, signal_hidden_entropy, prev_hidden_for_cosine) — all remain `None` as draft-side signal buffers are disabled.

---

### 8. `eagle_worker.py` (+512 lines, -7 lines)

This is the main orchestration file. Changes fall into six categories.

#### 8a. Initialization (lines 98-121)

```python
self._dynamic_spec_enabled = server_args.enable_dynamic_speculative_decoding
if self._dynamic_spec_enabled:
    self.topk_max = server_args.speculative_eagle_topk_max
    self.num_steps_max = server_args.speculative_num_steps_max
    self.ndt_max = server_args.speculative_num_draft_tokens_max
    self.dynamic_spec_policy = DynamicSpecPolicy(server_args)
    self.draft_attn_backends_map = {}
    self.cuda_graph_runners_map = {}
    self.cuda_graph_runners_extend_map = {}
    self._last_draft_entropy = 0.0
    self._last_target_entropy = 0.0
    self._eager_signal_draft_entropy = None
    self._eager_signal_hidden_norm = None
```
- `_dynamic_spec_enabled`: Master gate for all dynamic spec code paths. When False, zero overhead.
- `topk_max, num_steps_max, ndt_max`: Upper bounds from server args.
- `dynamic_spec_policy`: The policy engine (in `dynamic_spec.py`). Takes 7 signals, returns `(topk, num_steps, ndt)`.
- `*_map` dicts: Lookup tables for pre-created backends and CUDA graph runners, populated during init.
- `_last_*_entropy`: Scalar values (or tensors) carried between draft/verify/next-draft for signal collection.
- `_eager_signal_*`: Lazily allocated GPU buffers for signal computation in eager mode (CUDA graph mode uses the graph runner's pre-allocated buffers instead).

#### 8b. Multi-config attention backends (lines 254-272)

```python
if self._dynamic_spec_enabled:
    for topk in range(1, self.topk_max + 1):
        for num_steps in range(1, self.num_steps_max + 1):
            ...
            self.draft_attn_backends_map[(topk, num_steps)] = factory.create_decode_backend()
```
Creates one `FlashInferMultiStepDraftBackend` per `(topk, num_steps)` combo. Each backend is an isolated instance with its own FlashInfer wrapper state. At runtime, `_apply_dynamic_spec_config()` swaps to the correct backend before each draft phase.

#### 8c. Multi-config CUDA graph capture (`_init_cuda_graphs_dynamic`, lines 320-420)

**Draft graph capture loop (lines 340-377)**:
```python
for topk in range(1, self.topk_max + 1):
    for num_steps in range(1, self.num_steps_max + 1):
        ...
        self.cuda_graph_runners_map[(topk, num_steps)] = Device2DraftCudaGraphRunner(self)
```
Before each capture, sets the full config on both `self` and `server_args` (the draft graph runner reads `topk` and `num_steps` from server_args to size its internal buffers). Computes valid ndt per config: `topk=1 → ndt=steps+1` (linear chain), `topk>1 → ndt=min(ndt_max, tree_size+1)`.

**Extend graph capture loop (lines 379-417)**:
```python
self.draft_extend_backends_map = {}
for num_steps in range(1, self.num_steps_max + 1):
    extend_factory = DraftBackendFactory(self.server_args, self.draft_model_runner, self.topk_max, num_steps)
    extend_backend = extend_factory.create_draft_extend_backend()
    self.draft_extend_backends_map[num_steps] = extend_backend
    self.draft_extend_attn_backend = extend_backend
    self.cuda_graph_runners_extend_map[num_steps] = EAGLEDraftExtendCudaGraphRunner(self)
```
Each `num_steps` gets its OWN `draft_extend_attn_backend`. This was the root cause of the accept_len collapse: `EAGLEDraftExtendCudaGraphRunner.__init__` calls `init_cuda_graph_state()` on the backend, which reallocates FlashInfer internal buffers. With a shared backend, each subsequent capture overwrote the previous one's buffers, leaving earlier graphs with stale FlashInfer wrappers that produced garbage `topk_p` and `hidden_states`. All extend graphs use `topk_max` so that `capture_for_decode` (baked inside the graph) always produces `topk_p` with `topk_max` columns.

**Config restoration (lines 399-415)**: Restores `self.topk`, `self.speculative_num_steps`, `self.speculative_num_draft_tokens`, `server_args.*`, `self.draft_attn_backend`, `self.cuda_graph_runner`, and `self.cuda_graph_runner_for_draft_extend` to the starting config.

#### 8d. Per-step config application (`_apply_dynamic_spec_config`, lines 846-918)

```python
signals = self._collect_signals(batch)
topk, num_steps, ndt, confidence = self.dynamic_spec_policy.get_config(signals)

self.topk = topk
self.speculative_num_steps = num_steps
self.speculative_num_draft_tokens = ndt

# Truncate topk_p from (bs, topk_max) to (bs, topk)
spec_info.topk_p = spec_info.topk_p[:, :topk].contiguous()
spec_info.topk_index = spec_info.topk_index[:, :topk].contiguous()

# Swap backends and runners
self.draft_attn_backend = self.draft_attn_backends_map[(topk, num_steps)]
self.cuda_graph_runner = self.cuda_graph_runners_map[(topk, num_steps)]
self.cuda_graph_runner_for_draft_extend = self.cuda_graph_runners_extend_map[num_steps]
self.draft_extend_attn_backend = self.draft_extend_backends_map[num_steps]
```
Called at the start of each `draft()`. Ordering is critical: signals are collected FIRST (from the previous round's spec_info), THEN config is swapped. This ensures signals reflect the previous round's behavior, not the yet-to-run current round.

The extend backend swap (`draft_extend_attn_backend`) is essential: the extend CUDA graph runner reads `self.eagle_worker.draft_extend_attn_backend` at replay time, so it must point to the backend that was used during that graph's capture.

#### 8e. Signal collection (`_collect_signals`, lines 990-1069)

Simplified from the original 5-signal GPU batch to 2 active GPU signals:

```python
top1 = topk_p[:, 0].mean()
signals_cpu = torch.stack([
    top1,
    _t(self._last_target_top1_prob),
]).tolist()
```

Only 2 signals require GPU computation:
- `top1_prob`: from `spec_info.topk_p[:, 0].mean()` (previous extend phase)
- `target_top1_prob`: cached as `self._last_target_top1_prob` from `eagle_info.py:verify()`

These are batched into a single `torch.stack().tolist()` — one GPU→CPU sync per step.

All other signals are pure CPU:
- `rolling_accept_rate`, `rolling_accept_length`: averaged from per-request EMA values
- `draft_oracle_gate = top1_prob × rolling_accept_rate`: CPU multiply
- `target_oracle_gate_fixed = target_top1_prob × rolling_accept_rate`: CPU multiply
- `joint_confidence_product_fixed = top1_prob × target_top1_prob`: CPU multiply
- `confidence_agreement = 1 - |top1_prob - target_top1_prob|`: CPU arithmetic

20 signals remain at 0.0 (disabled): draft_entropy, hidden_norm, hidden_cosine_sim, hidden_projection_score, hidden_entropy, target_entropy, target_top1_gap, target_varentropy, all target_hidden_*, kl_approx_target_draft, target_draft_agree, joint_entropy_gate, target_oracle_gate, joint_confidence_product, draft_oracle_gate_fixed.

Returns a `DynamicSpecSignals` dataclass with 8 active fields populated and 20 defaulting to 0.0.

#### 8f. Draft forward split return (`draft_forward`, lines 1069-1088)

```python
if self._dynamic_spec_enabled:
    flat_scores = torch.cat(score_list, dim=1).flatten(1)
    flat_tokens = torch.cat(token_list, dim=1)
    flat_parents = torch.cat(parents_list[:-1], dim=1) if len(parents_list) > 1 else ...
    return flat_scores, flat_tokens, flat_parents
else:
    parent_list, top_scores_index, draft_tokens = organize_draft_results(...)
    return parent_list, top_scores_index, draft_tokens
```
When dynamic spec is enabled, `draft_forward()` returns concatenated raw tree data instead of calling `organize_draft_results()` inside the graph. This is because `organize_draft_results` does `torch.topk(scores, ndt-1)` which bakes `ndt` as a constant when captured in a CUDA graph. Since ndt varies at runtime, the topk must run OUTSIDE the graph. The `torch.cat` operations have fixed output shapes (determined by `topk × num_steps`) and are safe inside the graph.

Then in `draft()` (lines 786-809):
```python
top_scores = torch.topk(flat_scores, ndt - 1, dim=-1)
top_scores_index = torch.sort(top_scores.indices).values
draft_tokens = torch.gather(flat_tokens, index=top_scores_index, dim=1)
```
This is mathematically identical to `organize_draft_results()` but runs outside the graph with the runtime ndt value.

#### 8g. Verify `num_tokens_per_bs` override (lines 1133-1195)

```python
spec_info.num_tokens_per_req = spec_info.draft_token_num
if graph_runner and spec_info.draft_token_num != graph_runner.num_tokens_per_bs:
    _saved_ndt_per_bs = graph_runner.num_tokens_per_bs
    graph_runner.num_tokens_per_bs = spec_info.draft_token_num
...
# After forward:
if _saved_ndt_per_bs is not None:
    graph_runner.num_tokens_per_bs = _saved_ndt_per_bs
```
The target verify CUDA graph runner uses `num_tokens_per_bs` to compute `raw_num_token = raw_bs * num_tokens_per_bs`. Since `num_tokens_per_bs` was initialized to `ndt_max` for buffer allocation, it must be overridden to the actual ndt before replay so that `replay_prepare()` copies the correct number of tokens and selects the right `(bs, ndt)` graph. Restored after forward to avoid leaking state.

#### 8h. `capture_for_decode` topk_max (lines 1477-1509)

```python
topk_for_extend = self.topk_max if self._dynamic_spec_enabled else self.topk
draft_input.topk_p, draft_input.topk_index = fast_topk(probs, topk_for_extend, dim=-1)
```
The extend phase always produces `topk_p` with `topk_max` columns when dynamic spec is enabled. This ensures the next draft round can truncate to any `topk <= topk_max` via `_apply_dynamic_spec_config`. Without this, switching from topk=1 to topk=3 would fail because the previous round's `topk_p` only had 1 column.

#### 8i. Other small changes

- **Line 854**: `draft_token_num=self.speculative_num_draft_tokens` instead of `self.server_args.speculative_num_draft_tokens`. Uses the instance variable (which is swapped per step) instead of the server_args value (which is stale after `_init_cuda_graphs_dynamic` restored it to starting config).
- **Signal entropy computation in `draft_forward`** (lines 1064-1068): Writes entropy and hidden norm to pre-allocated signal buffers when they exist. These operations are baked into the CUDA graph during capture. When dynamic spec is off, buffers are None and this block is skipped.
- **Logging** (`[DYNSPEC ...]` tags): Diagnostic logging throughout for monitoring dynamic spec behavior. Guarded by `self._dynamic_spec_enabled` — zero overhead when off.

---

### 9. `dynamic_spec.py` (+407 lines, new file)

The policy engine that maps runtime signals to tree-shape parameters. Contains three components:

**`DynamicSpecSignals` dataclass (28 fields)**:
Container for all signals. Organized into categories:
- Draft-side (7): draft_entropy, top1_prob, top1_minus_top2, hidden_norm, hidden_cosine_sim, hidden_projection_score, hidden_entropy
- Target-side (4): target_entropy, target_top1_prob, target_top1_gap, target_varentropy
- Target hidden (4): target_hidden_norm, target_hidden_cosine_sim, target_projection_score, target_hidden_entropy
- Distribution divergence (2): kl_approx_target_draft, target_draft_agree
- Joint gates — original (4): joint_entropy_gate, draft_oracle_gate, target_oracle_gate, joint_confidence_product
- Joint gates — fixed (4): draft_oracle_gate_fixed, target_oracle_gate_fixed, joint_confidence_product_fixed, confidence_agreement
- Historical (2): rolling_accept_rate, rolling_accept_length

All fields default to 0.0. Only 8 are actively computed (see Section 8e).

**`AdaptiveSignalNormalizer`**:
Normalizes signals to [0, 1] using fixed known bounds (no warmup period). Signals in `_INVERTED_SIGNALS` (entropy-like signals where lower = more confident) are inverted after normalization. Active from step 1.

**`DynamicSpecPolicy` — V3 universal policy**:
The core policy. Uses `draft_oracle_gate` (= `top1_prob × rolling_accept_rate`) as the single driving signal. Derived from decision tree analysis on 51k steps: draft_oracle_gate has 66% feature importance for predicting accept_length (R²=0.38 alone).

```python
# Single signal: draft_oracle_gate = top1_prob × rolling_accept_rate
dog = signals.draft_oracle_gate
t = min(1.0, max(0.0, dog / 0.5))   # normalize to [0, 1]

# Inverted mapping: wider when uncertain, deeper when confident
num_steps = _interp(t, start, max)      # deeper when confident
topk = _interp(1-t, start, max)         # wider when uncertain
ndt = topk*steps+1 (tree) or steps+1 (chain), capped at ndt_max
```

Key design decisions:
- **DOG divisor = 0.5**: Universal across model pairs. Llama (dog mean≈0.37) → t≈0.74 (deep+narrow), DeepSeek (dog mean≈0.15) → t≈0.30 (wide+shallow). The signal itself carries model-specific information.
- **Inverted topk**: Validated experimentally — wider trees when uncertain catch more candidates; deeper chains when confident exploit sequential coherence.
- **Never below starting config**: `max(start, ...)` clamp on all parameters. This was the critical fix — going below start always hurt throughput.
- **`compute_confidence()`**: Weighted average of 6 active signals (equal weight 1/6 each), used only for logging in `spec_signal_log`. The actual policy ignores this and uses raw `draft_oracle_gate` directly.

---

## CUDA graph architecture summary

| Graph type | Key | Count per bs | What varies | Isolated state |
|------------|-----|-------------|-------------|----------------|
| Target verify | `(bs, ndt)` | `ndt_max - ndt_min + 1` | Token count | FlashInfer `prefill_cuda_graph_metadata[(bs, ndt)]` |
| Draft | `(topk, num_steps)` → per-runner `bs` | `topk_max × num_steps_max` | Tree shape | Per-backend `FlashInferMultiStepDraftBackend` instance |
| Extend | `num_steps` → per-runner `bs` | `num_steps_max` | Accept length | Per-backend `FlashInferAttnBackend` via `draft_extend_backends_map` |

---

## Bugs found and fixed during development

### Bug 1: Target verify KV cache corruption
**Symptom**: `cudaErrorIllegalAddress` crash or accept_len degradation.
**Root cause**: Single target verify graph captured with `ndt_max=8`. At runtime with `ndt=4`, positions 4-7 had stale `out_cache_loc`. `set_kv_buffer()` wrote garbage KV to those stale locations.
**Fix**: Multi-ndt capture (one graph per ndt value). Each graph processes exactly ndt positions.

### Bug 2: Target verify FlashInfer wrapper overwrite
**Symptom**: `cudaErrorIllegalAddress` on first Graph-3 replay.
**Root cause**: `prefill_cuda_graph_metadata[bs]` overwritten by each subsequent ndt capture. Graph-3's wrappers pointed to freed buffers.
**Fix**: Key by `(bs, ndt)` in `flashinfer_backend.py`.

### Bug 3: Extend graph FlashInfer wrapper overwrite (root cause of accept_len=1.0)
**Symptom**: Accept_len collapsed to 1.0 at bs=1, in both eager and CUDA graph modes.
**Root cause**: Single shared `draft_extend_attn_backend`. Each `EAGLEDraftExtendCudaGraphRunner.__init__` called `init_cuda_graph_state()` which reallocated FlashInfer buffers. Only the last-captured extend graph had valid wrappers. The starting config's extend graph replayed with stale wrappers, producing garbage `topk_p` → draft model received corrupted input → all tokens rejected.
**Fix**: Per-`num_steps` extend backends in `draft_extend_backends_map`. Each extend graph captures with its own isolated backend.

### Bug 4: accept_index global→local conversion
**Symptom**: `IndexError` when bs>1 and logging accepted/rejected tokens.
**Root cause**: `accept_index` values are global indices into the flattened array. For request `i`, local index = global - `i * draft_token_num`.
**Fix**: `base = i * self.draft_token_num; acc_indices_local = [idx - base for idx in acc_indices_global]`.

### Bug 5: GPU→CPU sync overhead
**Symptom**: 5 implicit `torch.cuda.synchronize()` per decode step from `.item()` calls.
**Fix**: Batch all signal computations into a single `torch.stack([...]).tolist()` — one sync instead of five.

---

## Performance results (MT-Bench, 80 questions, bs=1, A100)

### Original EAGLE3 (static) vs Dynamic EAGLE3 (V3, start→max)

| Model Pair | Config | Throughput (tok/s) | Accept Length |
|------------|--------|-------------------|---------------|
| **Llama 3.1 8B** | Original EAGLE3 3,1,4 | 204.0 | 3.15 |
| | Original EAGLE3 7,4,8 | 209.0 | 3.70 |
| | Dynamic EAGLE3 3,1,4→7,4,8 | 216.1 | 3.45 |
| | *Improvement over 7,4,8* | *+3.4%* | |
| **DeepSeek-R1 8B** | Original EAGLE3 3,1,4 | 113.2 | 2.15 |
| | Original EAGLE3 7,4,8 | 118.8 | 2.36 |
| | Dynamic EAGLE3 3,1,4→7,4,8 | 121.3 | 2.29 |
| | *Improvement over 7,4,8* | *+2.1%* | |
| **Qwen3-8B** | Original EAGLE3 3,1,4 | 160.2 | 2.59 |
| | Original EAGLE3 7,4,8 | 169.7 | 3.35 |
| | Dynamic EAGLE3 3,1,4→7,4,8 | 176.0 | 3.28 |
| | *Improvement over 7,4,8* | *+3.7%* | |

Dynamic EAGLE3 achieves +2-4% throughput over the best static configuration across all three model pairs. The throughput gain comes from **reducing draft cost when draft quality is low** — the V3 policy's inverted topk mapping reduces average topk from 4.0 to ~2.15 on Llama, saving ~47% draft compute while maintaining similar acceptance rates. Accept length is slightly lower than static 7,4,8 because the policy sometimes uses smaller trees, but the reduced draft compute more than compensates.
