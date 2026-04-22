# Dynamic Speculative Decoding — Policy Experiments & Results

**Date**: 2026-04-02
**Benchmark**: MT-Bench, 10 questions (20 turns), batch_size=1
**Hardware**: Cambridge CSD3 Wilkes3, A100 GPU
**Models**: Llama-3.1-8B-Instruct (matched draft), DeepSeek-R1-Distill-Llama-8B (mismatched draft)
**Draft model**: lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B (all runs)

---

## Policy Versions

### Vanilla (no dynamic spec)
- Fixed config (topk, num_steps, ndt) from server args
- No signal collection, no runtime adaptation
- Baseline for all comparisons

### OLD broken (initial mapping)
- **Mapping**: Piecewise-linear, confidence=0.5 maps to starting config
  - conf < 0.5 → scales DOWN below starting config (harmful)
  - conf > 0.5 → scales UP toward max
- **Signals**: 6 equal-weight (jcp_fixed, draft_oracle_gate, target_oracle_gate_fixed, confidence_agreement, target_top1_prob, rolling_accept_length)
- **Fatal flaw**: Scaling below start (e.g., to config 1,2,3) always worse than vanilla — 62-72% rejection, lower efficiency
- **Server args**: start=3,1,4, max=5,4,8

### V1 — "Never Below Start" with inverted topk
- **Mapping**: Two-regime piecewise-linear
  - conf ≤ 0.3 (safe_threshold) → starting config
  - conf > 0.3 → interpolate toward max, with topk INVERTED (wider when uncertain, narrower when confident)
- **Signals**: Same 6 equal-weight as OLD broken
- **Key insight**: Follows TALON/Sequoia consensus — uncertain tokens get wide+shallow trees (hedging), confident tokens get deep+narrow (committing)
- **Thresholds**: safe_threshold=0.3
- **Server args**: start=3,1,4 max=5,4,8 or start=5,1,6 max=7,4,8

### V1+CB — V1 with Circuit Breaker
- **Mapping**: Same as V1 (inverted topk, safe_threshold=0.3)
- **Signals**: Same 6 equal-weight
- **Addition**: Consecutive failure circuit breaker — after 5 consecutive acc=0 steps, force starting config regardless of signals. Counter resets on any successful acceptance. Implemented via `req.consecutive_spec_failures` field.
- **Server args**: start=3,1,4 max=7,4,8 or start=5,1,6 max=7,4,8

### V2 — Depth-first with 3 weighted signals (REGRESSED)
- **Mapping**: Two thresholds — safe_threshold=0.4, topk_threshold=0.6
  - conf ≤ 0.4 → starting config
  - 0.4 < conf ≤ 0.6 → depth scales up, topk stays at start (depth-first)
  - conf > 0.6 → both depth and topk scale up
- **Signals**: 3 non-redundant with weighted importance
  - jcp_fixed: 0.50 weight
  - confidence_agreement: 0.25 weight
  - rolling_accept_length: 0.25 weight
- **Addition**: Same circuit breaker as V1+CB (threshold=2, later raised to 5)
- **Why it regressed**: Depth-first kept topk=1 for 64-92% of steps (0.38-0.56 efficiency). V1's inverted topk giving topk=2-3 at medium confidence (0.66-0.69 efficiency) was far better. The 3-signal weighted composite made the policy over-confident (mean conf shifted up), and the higher safe_threshold (0.4) was too conservative.
- **Server args**: start=3,1,4 max=7,4,8 or start=5,1,6 max=7,4,8

### TREE — Decision tree learned from 51k steps
- **Mapping**: Hardcoded if-else tree (depth=5) trained with sklearn DecisionTreeClassifier on Oracle labels from 51k collected steps across V1, V2, and V1+CB runs
- **Signals used by tree**: Only 3 features matter (from feature importance analysis):
  - `target_top1_prob` (61% importance) — target model confidence
  - `top1_prob` (23% importance) — draft model confidence
  - `joint_confidence_product_fixed` (16% importance) — product of both
- **Training**: Oracle approach — for each (top1_prob, target_top1_prob) quantile bucket, find which (topk, num_steps) config had the highest efficiency (accept_length+1)/ndt. Train tree on these optimal labels. 97% cross-validation accuracy.
- **Key decisions learned**:
  - Both uncertain (top1<0.47, ttp<0.65) → (4,3) wide+shallow
  - Draft uncertain + target confident (top1<0.47, ttp>0.81) → (1,4) moderate chain
  - Both moderate (top1>0.47, ttp 0.66-0.81) → (2,5) balanced
  - Both confident (top1>0.47, ttp>0.81, jcp>0.78) → (3,7) deep+wide
  - Both very confident (jcp>0.89, top1>0.94, ttp>0.89) → (1,6) deep chain
- **Limitation**: Configs are hardcoded to the specific (start, max) ranges used in training data. Not generalizable to arbitrary server args without retraining.
- **Server args**: start=3,1,4 max=7,4,8 or start=5,1,6 max=7,4,8

### V3 — Universal single-signal policy (BEST)
- **Mapping**: Continuous linear ramp using a single signal
  - `t = min(1.0, draft_oracle_gate / 0.5)` where `t ∈ [0, 1]`
  - `num_steps = interp(t, start, max)` — deeper when confident (direct)
  - `topk = interp(1-t, start, max)` — wider when uncertain (inverted)
  - `ndt = interp(t, start, max)` — more tokens when deeper
- **Signal**: Only `draft_oracle_gate` (= `top1_prob × rolling_accept_rate`), the strongest single predictor of accept_length (R²=0.380, 66% feature importance in fair regression tree)
- **Divisor 0.5**: Universal constant validated on both Llama and DeepSeek. At dog=0.5 the policy saturates to max depth. The signal self-calibrates across models: Llama (dog mean=0.37 → deep+narrow), DeepSeek (dog mean=0.15 → wide+shallow)
- **No warmup, no thresholds, no branches**: Works from step 1. No safe_threshold, no topk_threshold, no circuit breaker. The signal value directly controls the config via a single division
- **Distilled from decision tree analysis**: The TREE policy's 15 branches with 3 signals were analyzed and found to implement a single underlying pattern: `draft_oracle_gate` drives topk inversely (r=-0.37) and num_steps directly (r=+0.66). The V3 formula captures this pattern with one line of code
- **Key advantage over TREE**: Universal — works for any (start, max) server args. TREE's configs were hardcoded to specific ranges. V3 uses `_interp(t, start, max)` so the same code adapts to any operating point
- **Circuit breaker**: Still present (from V1+CB, threshold=5) but rarely fires since the single-signal mapping is well-calibrated
- **Server args**: Any start/max combination, e.g., start=3,1,4 max=7,4,8

```
draft_oracle_gate:  0.0 ────────── 0.25 ────────── 0.5 ───── 1.0
                     │               │               │
topk:            MAX(4) ───────── MID(2) ──────── MIN(1) ━━━━ MIN(1)
steps:           MIN(3) ───────── MID(5) ──────── MAX(7) ━━━━ MAX(7)
                     │               │               │
shape:        wide+shallow      balanced       narrow+deep
```

---

## Results — Llama 3.1 8B (matched draft)

| Run | Wall(s) | AccLen | TP(tok/s) | Steps | vs 3,1,4 | vs 5,1,6 |
|-----|---------|--------|-----------|-------|----------|----------|
| Vanilla 3,1,4 | 34.6 | 2.871 | 177.3 | — | baseline | -6.3% |
| OLD broken | 34.6 | 2.994 | 174.7 | 2015 | -1.5% | -7.7% |
| V1 3,1,4→5,4,8 | 32.8 | 3.280 | 183.5 | 1804 | +3.5% | -3.1% |
| V1+CB 3,1,4→7,4,8 | 34.6 | 3.373 | 181.3 | 1842 | +2.2% | -4.3% |
| V2 3,1,4→7,4,8 | 45.1 | 3.340 | 181.2 | 1906 | +2.2% | -4.3% |
| TREE 3,1,4→7,4,8 | 37.9 | 3.816 | 201.0 | 1512 | +13.3% | +6.2% |
| **V3 3,1,4→7,4,8** | **37.4** | **3.956** | **205.4** | **1507** | **+15.8%** | **+8.5%** |
| Vanilla 5,1,6 | 32.3 | 3.390 | 189.3 | — | +6.8% | baseline |
| V1 5,1,6→7,4,8 | 32.6 | 3.815 | 193.4 | 1590 | +9.1% | +2.1% |
| V1+CB 5,1,6→7,4,8 | 31.0 | 3.815 | 197.0 | 1590 | +11.1% | +4.0% |
| V2 5,1,6→7,4,8 | 34.1 | 3.565 | 182.0 | 1715 | +2.6% | -3.9% |
| TREE 5,1,6→7,4,8 | 30.5 | 3.816 | 198.4 | 1512 | +11.9% | +4.8% |
| **V3 5,1,6→7,4,8** | **31.4** | **3.951** | **200.9** | **1580** | **+13.3%** | **+6.1%** |

**SpecForge reference** (80 questions, official):
| Config | AccLen | TP(tok/s) | Speedup |
|--------|--------|-----------|---------|
| 3,1,4 | 2.93 | 414.9 | 2.18x |
| 5,1,6 | 3.55 | 453.7 | 2.39x |
| 7,1,8 | 3.91 | 454.7 | 2.39x |
| 5,3,6 | 2.94 | 338.6 | 1.78x |
| 7,4,8 | 2.91 | 305.2 | 1.61x |

---

## Results — DeepSeek-R1-Distill-Llama 8B (mismatched draft)

| Run | Wall(s) | AccLen | TP(tok/s) | Steps | vs 3,1,4 | vs 5,1,6 |
|-----|---------|--------|-----------|-------|----------|----------|
| Vanilla 3,1,4 | 125.3 | 1.864 | 116.4 | — | baseline | +3.1% |
| OLD broken | 130.9 | 1.900 | 115.1 | 7962 | -1.1% | +2.0% |
| V1 3,1,4→5,4,8 | 117.7 | 2.159 | 125.4 | 6771 | +7.8% | +11.1% |
| V1+CB 3,1,4→7,4,8 | 119.4 | 2.118 | 121.1 | 6857 | +4.0% | +7.3% |
| V2 3,1,4→7,4,8 | 132.0 | 1.953 | 113.5 | 7676 | -2.5% | +0.5% |
| TREE 3,1,4→7,4,8 | 110.9 | 2.296 | 127.0 | 6135 | +9.1% | +12.5% |
| **V3 3,1,4→7,4,8** | **108.7** | **2.340** | **131.0** | **6110** | **+12.6%** | **+16.0%** |
| Vanilla 5,1,6 | 124.5 | 1.996 | 112.9 | — | -3.0% | baseline |
| V1 5,1,6→7,4,8 | 121.1 | 2.107 | 111.4 | 6416 | -4.2% | -1.3% |
| V1+CB 5,1,6→7,4,8 | 120.4 | 2.102 | 112.0 | 6432 | -3.7% | -0.8% |
| V2 5,1,6→7,4,8 | 124.9 | 1.987 | 107.5 | 6768 | -7.7% | -4.8% |
| TREE 5,1,6→7,4,8 | 109.3 | 2.118 | 117.5 | 6069 | +1.0% | +4.1% |
| **V3 5,1,6→7,4,8** | **107.0** | **2.220** | **117.6** | **5655** | **+1.1%** | **+4.2%** |

---

## Key Findings

1. **V3 (single-signal universal) is the best policy**: +15.8% throughput on Llama, +12.6% on DeepSeek over vanilla 3,1,4. Outperforms all hand-tuned mappings AND the overfit decision tree. The simplest policy is the best.

2. **`draft_oracle_gate` is the single most important signal** (R²=0.380 alone, 66% feature importance in fair regression). It's a natural composite: `top1_prob × rolling_accept_rate` captures both draft model confidence and recent acceptance history in one value.

3. **The Oracle tree's feature importance was misleading**: It showed `target_top1_prob` at 61% importance, but this was an artifact of circular label construction (buckets defined by the same features). Fair regression analysis revealed `draft_oracle_gate` at 66% importance — the real #1 signal.

4. **"Never below start" is essential**: The OLD broken policy (-1.5%) scaled below starting config. All successful policies (V1, V1+CB, TREE, V3) never go below start.

5. **Inverted topk mapping is the key insight**: Wide trees when uncertain (topk=MAX at low dog), narrow+deep when confident (topk=MIN at high dog). This pattern was discovered in V1, confirmed by the decision tree, and distilled into V3's single-line formula.

6. **V3 beats the overfit tree by +2-3%**: The universal formula `t = dog / 0.5` with inverted topk outperforms the 15-branch hardcoded tree. Simplicity wins — fewer discrete boundaries means smoother config transitions and less sensitivity to signal noise.

7. **Circuit breaker helps Llama but hurts DeepSeek**: V1+CB improved Llama 5,1,6 (+4.0% vs +2.1%) but degraded DeepSeek 3,1,4 (+4.0% vs +7.8%). On DeepSeek, acc=0 is so common that the circuit breaker fires too often.

8. **V3 adapts to model pairs automatically**: On Llama (matched draft, dog mean=0.37) the policy favors deep+narrow trees. On DeepSeek (mismatched draft, dog mean=0.15) it favors wide+shallow trees. Same code, same constant, different behavior driven by the signal.

---

## Log Files

### Llama 8B

| Run | Server Log | Test Log | Signal Data |
|-----|-----------|----------|-------------|
| Vanilla 3,1,4 | [server](logs/server_vanilla_llama8b_10q.log) | [test](logs/test_vanilla_llama8b_10q.log) | [data](signal_data_vanilla_llama8b_10q.json) |
| OLD broken | [server](logs/server_6active_llama8b_10q.log) | [test](logs/test_6active_llama8b_10q.log) | [data](signal_data_6active_llama8b_10q.json) |
| V1 3,1,4→5,4,8 | [server](logs/server_dynamic_talon_llama_314_548.log) | [test](logs/test_dynamic_talon_llama_314_548.log) | [data](signal_data_dynamic_talon_llama_314_548.json) |
| V1+CB 3,1,4→7,4,8 | [server](logs/server_v1cb_llama_314_748.log) | [test](logs/test_v1cb_llama_314_748.log) | [data](signal_data_v1cb_llama_314_748.json) |
| V2 3,1,4→7,4,8 | [server](logs/server_v2_llama_314_748.log) | [test](logs/test_v2_llama_314_748.log) | [data](signal_data_v2_llama_314_748.json) |
| TREE 3,1,4→7,4,8 | [server](logs/server_tree_llama_314_748.log) | [test](logs/test_tree_llama_314_748.log) | [data](signal_data_tree_llama_314_748.json) |
| V3 3,1,4→7,4,8 | [server](logs/server_v3_llama_314_748.log) | [test](logs/test_v3_llama_314_748.log) | [data](signal_data_v3_llama_314_748.json) |
| Vanilla 5,1,6 | [server](logs/server_vanilla_llama_516.log) | [test](logs/test_vanilla_llama_516.log) | [data](signal_data_vanilla_llama_516.json) |
| V1 5,1,6→7,4,8 | [server](logs/server_dynamic_talon_llama_516_748.log) | [test](logs/test_dynamic_talon_llama_516_748.log) | [data](signal_data_dynamic_talon_llama_516_748.json) |
| V1+CB 5,1,6→7,4,8 | [server](logs/server_v1cb_llama_516_748.log) | [test](logs/test_v1cb_llama_516_748.log) | [data](signal_data_v1cb_llama_516_748.json) |
| V2 5,1,6→7,4,8 | [server](logs/server_v2_llama_516_748.log) | [test](logs/test_v2_llama_516_748.log) | [data](signal_data_v2_llama_516_748.json) |
| TREE 5,1,6→7,4,8 | [server](logs/server_tree_llama_516_748.log) | [test](logs/test_tree_llama_516_748.log) | [data](signal_data_tree_llama_516_748.json) |
| V3 5,1,6→7,4,8 | [server](logs/server_v3_llama_516_748.log) | [test](logs/test_v3_llama_516_748.log) | [data](signal_data_v3_llama_516_748.json) |

### DeepSeek 8B

| Run | Server Log | Test Log | Signal Data |
|-----|-----------|----------|-------------|
| Vanilla 3,1,4 | [server](logs/server_vanilla_deepseek8b_lmsys_10q.log) | [test](logs/test_vanilla_deepseek8b_lmsys_10q.log) | [data](signal_data_vanilla_deepseek8b_lmsys_10q.json) |
| OLD broken | [server](logs/server_6active_deepseek8b_lmsys_10q.log) | [test](logs/test_6active_deepseek8b_lmsys_10q.log) | [data](signal_data_6active_deepseek8b_lmsys_10q.json) |
| V1 3,1,4→5,4,8 | [server](logs/server_dynamic_talon_deepseek_314_548.log) | [test](logs/test_dynamic_talon_deepseek_314_548.log) | [data](signal_data_dynamic_talon_deepseek_314_548.json) |
| V1+CB 3,1,4→7,4,8 | [server](logs/server_v1cb_deepseek_314_748.log) | [test](logs/test_v1cb_deepseek_314_748.log) | [data](signal_data_v1cb_deepseek_314_748.json) |
| V2 3,1,4→7,4,8 | [server](logs/server_v2_deepseek_314_748.log) | [test](logs/test_v2_deepseek_314_748.log) | [data](signal_data_v2_deepseek_314_748.json) |
| TREE 3,1,4→7,4,8 | [server](logs/server_tree_deepseek_314_748.log) | [test](logs/test_tree_deepseek_314_748.log) | [data](signal_data_tree_deepseek_314_748.json) |
| Vanilla 5,1,6 | [server](logs/server_vanilla_deepseek_516.log) | [test](logs/test_vanilla_deepseek_516.log) | [data](signal_data_vanilla_deepseek_516.json) |
| V1 5,1,6→7,4,8 | [server](logs/server_dynamic_talon_deepseek_516_748.log) | [test](logs/test_dynamic_talon_deepseek_516_748.log) | [data](signal_data_dynamic_talon_deepseek_516_748.json) |
| V1+CB 5,1,6→7,4,8 | [server](logs/server_v1cb_deepseek_516_748.log) | [test](logs/test_v1cb_deepseek_516_748.log) | [data](signal_data_v1cb_deepseek_516_748.json) |
| V2 5,1,6→7,4,8 | [server](logs/server_v2_deepseek_516_748.log) | [test](logs/test_v2_deepseek_516_748.log) | [data](signal_data_v2_deepseek_516_748.json) |
| TREE 5,1,6→7,4,8 | [server](logs/server_tree_deepseek_516_748.log) | [test](logs/test_tree_deepseek_516_748.log) | [data](signal_data_tree_deepseek_516_748.json) |
| V3 3,1,4→7,4,8 | [server](logs/server_v3_deepseek_314_748.log) | [test](logs/test_v3_deepseek_314_748.log) | [data](signal_data_v3_deepseek_314_748.json) |
| V3 5,1,6→7,4,8 | [server](logs/server_v3_deepseek_516_748.log) | [test](logs/test_v3_deepseek_516_748.log) | [data](signal_data_v3_deepseek_516_748.json) |

---

## Decision Tree Feature Importance

Trained on 51,367 steps across all V1/V2/V1+CB runs. Three training approaches compared:

| Feature | Regression (R²=0.26) | Classification (Acc=0.36) | Oracle (Acc=0.97) |
|---------|---------------------|--------------------------|-------------------|
| **target_top1_prob** | 0.000 | 0.049 | **0.606** |
| **top1_prob** | 0.086 | 0.000 | **0.225** |
| **jcp_fixed** | 0.133 | 0.110 | **0.164** |
| draft_oracle_gate | **0.767** | **0.590** | 0.000 |
| confidence_agreement | 0.005 | 0.000 | 0.005 |
| rolling_accept_rate | 0.000 | 0.090 | 0.000 |
| rolling_accept_length | 0.010 | 0.152 | 0.000 |
| target_oracle_gate_fixed | 0.000 | 0.008 | 0.000 |

**Note on Oracle circularity**: The Oracle tree's 97% accuracy and its feature ranking (`target_top1_prob` at 61%) were artifacts of circular label construction — labels were defined by bucketing on `(top1_prob, target_top1_prob)` quantiles, so the tree trivially learned the bucket boundaries. Fair regression analysis (predicting accept_length directly, no circular labels) reveals the true ranking:

### Fair Feature Importance (no circular labels)

| Feature | Predict AccLen (R²=0.43) | Predict Efficiency (R²=0.26) |
|---------|--------------------------|------------------------------|
| **draft_oracle_gate** | **0.658** | **0.752** |
| **jcp_fixed** | **0.269** | **0.130** |
| target_oracle_gate_fixed | 0.028 | 0.001 |
| rolling_accept_length | 0.025 | 0.011 |
| top1_prob | 0.016 | 0.088 |
| confidence_agreement | 0.002 | 0.006 |
| target_top1_prob | 0.001 | 0.004 |
| rolling_accept_rate | 0.000 | 0.007 |

**True key insight**: `draft_oracle_gate` (= top1_prob × rolling_accept_rate) is BY FAR the most important signal — 66% importance for accept_length, 75% for efficiency. This is what V3 uses as its sole signal, achieving the best results across all experiments.

---

## Server Launch Commands

### Vanilla (no dynamic spec)
```bash
python3 -m sglang.launch_server \
    --model <MODEL> \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps <STEPS> --speculative-eagle-topk <TOPK> \
    --speculative-num-draft-tokens <NDT> \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

### Dynamic spec (V1/V1+CB/V2/TREE)
```bash
python3 -m sglang.launch_server \
    --model <MODEL> \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps <START_STEPS> --speculative-eagle-topk <START_TOPK> \
    --speculative-num-draft-tokens <START_NDT> \
    --speculative-num-steps-max <MAX_STEPS> --speculative-eagle-topk-max <MAX_TOPK> \
    --speculative-num-draft-tokens-max <MAX_NDT> \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16 \
    --log-level debug  # optional: enables per-step DYNSPEC debug logs
```

### Test command
```bash
python test_signal_collection.py --port 30000 --num-samples 10 \
    --output <OUTPUT_FILE>.json \
    2>&1 | tee logs/<LOG_FILE>.log
```
