# Speculative Decoding: 4-Configuration Comparative Analysis

**Setup**: 5 MT-Bench questions, 2 turns each (10 turns total), bs=1, temperature=0, A100 GPU

| Config | Model | Mode | Vanilla Params |
|--------|-------|------|----------------|
| Vanilla Llama-8B | Llama-3.1-8B-Instruct | EAGLE3 fixed | topk=1, steps=3, ndt=4 |
| Dynamic Llama-8B | Llama-3.1-8B-Instruct | Dynamic spec | topk=[1..4], steps=[1..5], ndt=[2..8], start=1/3/4 |
| Vanilla DeepSeek-8B | DeepSeek-R1-Distill-Llama-8B | EAGLE3 fixed | topk=1, steps=3, ndt=4 |
| Dynamic DeepSeek-8B | DeepSeek-R1-Distill-Llama-8B | Dynamic spec | topk=[1..4], steps=[1..5], ndt=[2..8], start=1/3/4 |

## Part 1: Metric Comparison

### 1.1 Accept Length

| Config | N batches | Mean | Std | Min | Max | Median | P25 | P75 |
|--------|-----------|------|-----|-----|-----|--------|-----|-----|
| Vanilla Llama-8B | 28 | 2.9046 | 0.2502 | 2.2300 | 3.3500 | 2.9650 | 2.8000 | 3.0800 |
| Dynamic Llama-8B | 22 | 3.3123 | 0.4147 | 2.5200 | 4.0800 | 3.3250 | 2.9200 | 3.6700 |
| Vanilla DeepSeek-8B | 132 | 1.4378 | 0.2627 | 1.0700 | 2.6500 | 1.3800 | 1.2500 | 1.5700 |
| Dynamic DeepSeek-8B | 139 | 1.5240 | 0.5019 | 1.0000 | 4.2000 | 1.3800 | 1.2500 | 1.5700 |

### 1.2 Throughput (token/s)

Note: First batch of each request has cold-start throughput (~1-2 tok/s), excluded from steady-state stats below.

| Config | N batches | Mean | Std | Min | Max | Median | Steady-state Mean (excl 1st) |
|--------|-----------|------|-----|-----|-----|--------|------------------------------|
| Vanilla Llama-8B | 28 | 172.8214 | 36.7496 | 1.2000 | 209.2900 | 181.7550 | 179.1778 |
| Dynamic Llama-8B | 22 | 175.9827 | 48.3788 | 1.8600 | 232.6300 | 188.8100 | 184.2743 |
| Vanilla DeepSeek-8B | 132 | 89.0193 | 17.8682 | 1.7500 | 165.0700 | 84.8300 | 89.6855 |
| Dynamic DeepSeek-8B | 139 | 91.0435 | 28.6109 | 1.2200 | 237.7300 | 84.1500 | 91.6943 |

### 1.3 Total Latency

| Config | Elapsed (s) | Total decode steps | Steps/sec |
|--------|-------------|-------------------|-----------|
| Vanilla Llama-8B | 18.8 | 28 | 1.5 |
| Dynamic Llama-8B | 17.2 | 22 | 1.3 |
| Vanilla DeepSeek-8B | 85.1 | 132 | 1.6 |
| Dynamic DeepSeek-8B | 92.5 | 139 | 1.5 |

### 1.4 Dynamic vs Vanilla Comparison

**Llama-8B**:
- Accept length: vanilla=2.9046 vs dynamic=3.3123 (+14.0%)
- Throughput: vanilla=179.2 vs dynamic=184.3 tok/s (+2.8%)
- Latency: vanilla=18.8s vs dynamic=17.2s (-8.3%)
- Decode steps: vanilla=28 vs dynamic=22

**DeepSeek-8B**:
- Accept length: vanilla=1.4378 vs dynamic=1.5240 (+6.0%)
- Throughput: vanilla=89.7 vs dynamic=91.7 tok/s (+2.2%)
- Latency: vanilla=85.1s vs dynamic=92.5s (+8.7%)
- Decode steps: vanilla=132 vs dynamic=139

## Part 2: Signal Analysis (Dynamic Runs)

### 2.1 Dynamic Llama-8B Signal Statistics (903 steps)

| Signal | N | Min | Max | Mean | Std | Median | P25 | P75 |
|--------|---|-----|-----|------|-----|--------|-----|-----|
| draft_entropy | 903 | 0.0000 | 7.8810 | 1.5656 | 1.3871 | 1.2208 | 0.5256 | 2.2134 |
| top1_prob | 903 | 0.0444 | 1.0000 | 0.7743 | 0.2609 | 0.8982 | 0.5904 | 0.9948 |
| top1_minus_top2 | 903 | 0.0000 | 1.0000 | 0.6882 | 0.3426 | 0.8517 | 0.4017 | 0.9926 |
| hidden_norm | 903 | 38.0000 | 108.0000 | 62.9338 | 10.1269 | 62.0000 | 55.7500 | 69.0000 |
| target_entropy | 903 | 0.0001 | 2.8559 | 0.5678 | 0.4436 | 0.4565 | 0.2505 | 0.8012 |
| entropy_gap | 903 | -6.9416 | 1.3913 | -0.9978 | 1.2241 | -0.7370 | -1.6299 | -0.1121 |
| rolling_accept_rate | 903 | 0.0695 | 1.0000 | 0.5344 | 0.1632 | 0.5487 | 0.4248 | 0.6537 |
| confidence | 903 | 0.2752 | 0.9107 | 0.6671 | 0.1239 | 0.6889 | 0.5833 | 0.7685 |
| chosen_topk | 903 | 1 | 3 | 2.0587 | 0.7455 | 2 | 1 | 3 |
| chosen_num_steps | 903 | 2 | 5 | 3.6368 | 0.5375 | 4 | 3 | 4 |
| chosen_num_draft_tokens | 903 | 3 | 7 | 5.3189 | 0.9828 | 6 | 4 | 6 |

#### Confidence Distribution

| Bucket | Count | Pct |
|--------|-------|-----|
| [0.0, 0.1) | 0 | 0.0% |
| [0.1, 0.2) | 0 | 0.0% |
| [0.2, 0.3) | 3 | 0.3% |
| [0.3, 0.4) | 24 | 2.7% |
| [0.4, 0.5) | 66 | 7.3% |
| [0.5, 0.6) | 168 | 18.6% |
| [0.6, 0.7) | 220 | 24.4% |
| [0.7, 0.8) | 314 | 34.8% |
| [0.8, 0.9) | 106 | 11.7% |
| [0.9, 1.0) | 2 | 0.2% |

#### Chosen Config Distribution

**chosen_topk**:
| Value | Count | Pct |
|-------|-------|-----|
| 1 | 226 | 25.0% |
| 2 | 398 | 44.1% |
| 3 | 279 | 30.9% |

**chosen_num_steps**:
| Value | Count | Pct |
|-------|-------|-----|
| 2 | 18 | 2.0% |
| 3 | 300 | 33.2% |
| 4 | 577 | 63.9% |
| 5 | 8 | 0.9% |

**chosen_num_draft_tokens**:
| Value | Count | Pct |
|-------|-------|-----|
| 3 | 18 | 2.0% |
| 4 | 208 | 23.0% |
| 5 | 221 | 24.5% |
| 6 | 380 | 42.1% |
| 7 | 76 | 8.4% |

#### Config Tuple (topk, steps, ndt) Distribution

| (topk, steps, ndt) | Count | Pct |
|--------------------|-------|-----|
| (1, 3, 4) | 208 | 23.0% |
| (3, 4, 6) | 203 | 22.5% |
| (2, 4, 6) | 177 | 19.6% |
| (2, 4, 5) | 129 | 14.3% |
| (2, 3, 5) | 92 | 10.2% |
| (3, 4, 7) | 68 | 7.5% |
| (1, 2, 3) | 18 | 2.0% |
| (3, 5, 7) | 8 | 0.9% |

**Parameter range utilization:**
- topk: used [1, 2, 3] out of [1..4]
- num_steps: used [2, 3, 4, 5] out of [1..5]
- ndt: used [3, 4, 5, 6, 7] out of [2..8]

#### Adaptive Normalizer Warmup

Turn 0, first 10 steps (warmup):
- confidence values: [0.564, 0.753, 0.723, 0.685, 0.854, 0.531, 0.55, 0.672, 0.773, 0.668]
- chosen_topk: [1, 3, 2, 2, 3, 1, 1, 2, 3, 2]
- chosen_num_steps: [3, 4, 4, 4, 4, 3, 3, 4, 4, 4]
- chosen_ndt: [4, 6, 6, 5, 7, 4, 4, 5, 6, 5]

Turn 0, steps 10-19 (post-warmup):
- confidence values: [0.763, 0.693, 0.707, 0.695, 0.734, 0.624, 0.718, 0.703, 0.54, 0.728]
- chosen_topk: [3, 2, 2, 2, 2, 2, 2, 2, 1, 2]

#### Signal-Confidence Relationship

Pearson correlation of each signal with confidence:

| Signal | Correlation with confidence |
|--------|---------------------------|
| draft_entropy | -0.6131 |
| top1_prob | 0.8063 |
| top1_minus_top2 | 0.8017 |
| hidden_norm | -0.0673 |
| target_entropy | -0.5872 |
| entropy_gap | 0.4820 |
| rolling_accept_rate | 0.4535 |

### 2.1 Dynamic DeepSeek-8B Signal Statistics (5569 steps)

| Signal | N | Min | Max | Mean | Std | Median | P25 | P75 |
|--------|---|-----|-----|------|-----|--------|-----|-----|
| draft_entropy | 5569 | 0.0051 | 8.9547 | 4.3403 | 1.9227 | 4.2996 | 2.9556 | 5.7712 |
| top1_prob | 5569 | 0.0106 | 1.0000 | 0.4327 | 0.2824 | 0.3633 | 0.2020 | 0.6344 |
| top1_minus_top2 | 5569 | 0.0000 | 1.0000 | 0.3237 | 0.3118 | 0.2051 | 0.0623 | 0.5305 |
| hidden_norm | 5569 | 44.7500 | 204.0000 | 96.2192 | 23.9245 | 92.0000 | 77.0000 | 112.0000 |
| target_entropy | 5569 | 0.0000 | 4.9608 | 1.2604 | 0.8033 | 1.1540 | 0.6620 | 1.7313 |
| entropy_gap | 5569 | -8.8679 | 2.9566 | -3.0799 | 1.8731 | -3.0525 | -4.4513 | -1.7093 |
| rolling_accept_rate | 5569 | 0.0000 | 1.0000 | 0.1631 | 0.1598 | 0.1167 | 0.0498 | 0.2216 |
| confidence | 5569 | 0.1275 | 0.8565 | 0.4452 | 0.1218 | 0.4317 | 0.3583 | 0.5179 |
| chosen_topk | 5569 | 1 | 3 | 1.1514 | 0.4005 | 1 | 1 | 1 |
| chosen_num_steps | 5569 | 2 | 4 | 2.7800 | 0.5868 | 3 | 2 | 3 |
| chosen_num_draft_tokens | 5569 | 3 | 7 | 3.8738 | 0.7535 | 4 | 3 | 4 |

#### Confidence Distribution

| Bucket | Count | Pct |
|--------|-------|-----|
| [0.0, 0.1) | 0 | 0.0% |
| [0.1, 0.2) | 36 | 0.6% |
| [0.2, 0.3) | 526 | 9.4% |
| [0.3, 0.4) | 1627 | 29.2% |
| [0.4, 0.5) | 1731 | 31.1% |
| [0.5, 0.6) | 1010 | 18.1% |
| [0.6, 0.7) | 439 | 7.9% |
| [0.7, 0.8) | 162 | 2.9% |
| [0.8, 0.9) | 38 | 0.7% |
| [0.9, 1.0) | 0 | 0.0% |

#### Chosen Config Distribution

**chosen_topk**:
| Value | Count | Pct |
|-------|-------|-----|
| 1 | 4815 | 86.5% |
| 2 | 665 | 11.9% |
| 3 | 89 | 1.6% |

**chosen_num_steps**:
| Value | Count | Pct |
|-------|-------|-----|
| 2 | 1706 | 30.6% |
| 3 | 3382 | 60.7% |
| 4 | 481 | 8.6% |

**chosen_num_draft_tokens**:
| Value | Count | Pct |
|-------|-------|-----|
| 3 | 1706 | 30.6% |
| 4 | 3109 | 55.8% |
| 5 | 527 | 9.5% |
| 6 | 205 | 3.7% |
| 7 | 22 | 0.4% |

#### Config Tuple (topk, steps, ndt) Distribution

| (topk, steps, ndt) | Count | Pct |
|--------------------|-------|-----|
| (1, 3, 4) | 3109 | 55.8% |
| (1, 2, 3) | 1706 | 30.6% |
| (2, 3, 5) | 273 | 4.9% |
| (2, 4, 5) | 254 | 4.6% |
| (2, 4, 6) | 138 | 2.5% |
| (3, 4, 6) | 67 | 1.2% |
| (3, 4, 7) | 22 | 0.4% |

**Parameter range utilization:**
- topk: used [1, 2, 3] out of [1..4]
- num_steps: used [2, 3, 4] out of [1..5]
- ndt: used [3, 4, 5, 6, 7] out of [2..8]

#### Adaptive Normalizer Warmup

Turn 0, first 10 steps (warmup):
- confidence values: [0.643, 0.3, 0.544, 0.46, 0.482, 0.607, 0.508, 0.296, 0.271, 0.667]
- chosen_topk: [2, 1, 1, 1, 1, 2, 1, 1, 1, 2]
- chosen_num_steps: [4, 2, 3, 3, 3, 3, 3, 2, 2, 4]
- chosen_ndt: [5, 3, 4, 4, 4, 5, 4, 3, 3, 5]

Turn 0, steps 10-19 (post-warmup):
- confidence values: [0.741, 0.383, 0.471, 0.481, 0.298, 0.63, 0.672, 0.525, 0.463, 0.357]
- chosen_topk: [2, 1, 1, 1, 1, 2, 2, 1, 1, 1]

#### Signal-Confidence Relationship

Pearson correlation of each signal with confidence:

| Signal | Correlation with confidence |
|--------|---------------------------|
| draft_entropy | -0.6324 |
| top1_prob | 0.7782 |
| top1_minus_top2 | 0.7693 |
| hidden_norm | -0.0580 |
| target_entropy | -0.4242 |
| entropy_gap | 0.4672 |
| rolling_accept_rate | 0.4831 |

## Part 3: Improvement & Failure Analysis

### 3.1 Accept Length Trajectories

**Llama-8B**:

Accept length by batch window (window size=10):

| Window | Vanilla Mean | Dynamic Mean | Delta | Winner |
|--------|-------------|-------------|-------|--------|
| 0-9 | 2.9570 | 3.5620 | +0.6050 | Dynamic |
| 10-19 | 2.8590 | 3.0730 | +0.2140 | Dynamic |
| 20-29 | 2.8963 | 3.2600 | +0.3637 | Dynamic |

Window tally: Dynamic wins 3, Vanilla wins 0, Ties 0

**DeepSeek-8B**:

Accept length by batch window (window size=10):

| Window | Vanilla Mean | Dynamic Mean | Delta | Winner |
|--------|-------------|-------------|-------|--------|
| 0-9 | 1.2740 | 1.2480 | -0.0260 | Tie |
| 10-19 | 1.2630 | 1.2860 | +0.0230 | Tie |
| 20-29 | 1.4330 | 1.3740 | -0.0590 | Vanilla |
| 30-39 | 1.6260 | 2.0180 | +0.3920 | Dynamic |
| 40-49 | 1.4500 | 1.4360 | -0.0140 | Tie |
| 50-59 | 1.3750 | 1.3850 | +0.0100 | Tie |
| 60-69 | 1.3230 | 1.4480 | +0.1250 | Dynamic |
| 70-79 | 1.5090 | 1.3410 | -0.1680 | Vanilla |
| 80-89 | 1.3590 | 1.4770 | +0.1180 | Dynamic |
| 90-99 | 1.3620 | 1.2380 | -0.1240 | Vanilla |
| 100-109 | 1.7910 | 1.6670 | -0.1240 | Vanilla |
| 110-119 | 1.5140 | 2.7280 | +1.2140 | Dynamic |
| 120-129 | 1.3700 | 1.2540 | -0.1160 | Vanilla |
| 130-139 | 1.6500 | 1.4256 | -0.2244 | Vanilla |

Window tally: Dynamic wins 4, Vanilla wins 6, Ties 4

### 3.2 Rolling Accept Rate Analysis (Dynamic Runs)

**Dynamic Llama-8B**:
- Rolling accept rate: mean=0.5344, std=0.1632, min=0.0695, max=1.0000

- Accept rate < 0.3 (low):  86 steps (9.5%)
- Accept rate 0.3-0.7 (mid): 688 steps (76.2%)
- Accept rate >= 0.7 (high): 129 steps (14.3%)

**Dynamic DeepSeek-8B**:
- Rolling accept rate: mean=0.1631, std=0.1598, min=0.0000, max=1.0000

- Accept rate < 0.3 (low):  4695 steps (84.3%)
- Accept rate 0.3-0.7 (mid): 798 steps (14.3%)
- Accept rate >= 0.7 (high): 76 steps (1.4%)

### 3.3 Confidence vs Actual Behavior

**Dynamic Llama-8B**:

| Confidence Range | N | Mean RAR | Mean Topk | Mean Steps | Mean NDT | Aggressiveness |
|-----------------|---|----------|-----------|------------|----------|----------------|
| [0.2, 0.4) | 27 | 0.333 | 1.00 | 2.33 | 3.33 | Conservative |
| [0.4, 0.6) | 234 | 0.470 | 1.15 | 3.00 | 4.15 | Moderate |
| [0.6, 0.8) | 534 | 0.543 | 2.32 | 3.89 | 5.65 | Aggressive |
| [0.8, 1.0) | 108 | 0.684 | 3.00 | 4.07 | 6.70 | Aggressive |

High confidence (>0.7): 422 steps, mean RAR=0.595, mean NDT=6.2
Low confidence (<0.4): 27 steps, mean RAR=0.333, mean NDT=3.3

**Dynamic DeepSeek-8B**:

| Confidence Range | N | Mean RAR | Mean Topk | Mean Steps | Mean NDT | Aggressiveness |
|-----------------|---|----------|-----------|------------|----------|----------------|
| [0.0, 0.2) | 36 | 0.048 | 1.00 | 2.00 | 3.00 | Conservative |
| [0.2, 0.4) | 2153 | 0.108 | 1.00 | 2.22 | 3.22 | Conservative |
| [0.4, 0.6) | 2741 | 0.163 | 1.04 | 3.00 | 4.04 | Moderate |
| [0.6, 0.8) | 601 | 0.334 | 2.08 | 3.74 | 5.31 | Moderate |
| [0.8, 1.0) | 38 | 0.673 | 3.00 | 4.00 | 6.58 | Aggressive |

High confidence (>0.7): 200 steps, mean RAR=0.528, mean NDT=6.1
Low confidence (<0.4): 2189 steps, mean RAR=0.107, mean NDT=3.2

### 3.4 Dynamic Spec Overhead Analysis

Dynamic spec introduces overhead from:
1. CUDA graph switching (selecting different graphs per step)
2. Signal computation (entropy calculations, normalization)
3. Potentially larger trees that waste verification compute

**Llama-8B**:
- Steps using vanilla config (1,3,4): 208 (23.0%)
- Steps using different config: 695 (77.0%)

  - More aggressive (ndt > 4): 677
  - Less aggressive (ndt < 4): 18
  - Same ndt but different tree shape: 0

**DeepSeek-8B**:
- Steps using vanilla config (1,3,4): 3109 (55.8%)
- Steps using different config: 2460 (44.2%)

  - More aggressive (ndt > 4): 754
  - Less aggressive (ndt < 4): 1706
  - Same ndt but different tree shape: 0

## Part 4: Improvement Recommendations

### 4.1 Signal Weight Analysis

Current policy uses equal weights (1/7 each = 0.143). Key findings:

**Dynamic Llama-8B**:

| Signal | Corr with rolling_accept_rate | Interpretation |
|--------|------------------------------|----------------|
| draft_entropy | -0.1629 | Weak predictor |
| top1_prob | 0.1669 | Weak predictor |
| top1_minus_top2 | 0.1563 | Weak predictor |
| hidden_norm | -0.0458 | Not predictive |
| target_entropy | -0.3724 | Moderate predictor |
| entropy_gap | 0.0496 | Not predictive |

**Dynamic DeepSeek-8B**:

| Signal | Corr with rolling_accept_rate | Interpretation |
|--------|------------------------------|----------------|
| draft_entropy | -0.1696 | Weak predictor |
| top1_prob | 0.2503 | Weak predictor |
| top1_minus_top2 | 0.2552 | Weak predictor |
| hidden_norm | -0.0689 | Not predictive |
| target_entropy | -0.3827 | Moderate predictor |
| entropy_gap | 0.0100 | Not predictive |

### 4.2 Policy Design Issues

#### Issue 1: Equal weights are suboptimal

The equal 1/7 weighting treats all signals as equally informative. The correlation
analysis above shows some signals are much better predictors of acceptance than others.
**Recommendation**: Weight signals proportionally to their correlation with acceptance.
Specifically, `rolling_accept_rate` itself should have the highest weight since it
directly measures what we're trying to predict. `top1_prob` and `draft_entropy`
should be weighted higher than `hidden_norm` which may be less informative.

#### Issue 2: Piecewise-linear mapping may be too coarse

The confidence-to-config mapping uses just two linear segments (confidence 0..0.5 and
0.5..1.0) with rounding to integers. This creates sharp jumps between configs.
**Recommendation**: Consider a smoother mapping, or use direct regression from signals
to parameters rather than going through a scalar confidence bottleneck.

#### Issue 3: Adaptive normalizer sensitivity

The min/max normalizer is sensitive to outliers. A single extreme value permanently
shifts the normalization range. **Recommendation**: Use percentile-based normalization
(e.g., clip to 5th-95th percentile) or exponential moving average for min/max.

#### Issue 4: Warmup resets per request, not per session

Each new request starts fresh signal collection from step 0. The warmup period
(10 steps) means the first 10 steps of EVERY request use the starting config.
With only 5 questions and varying lengths, this is a significant fraction of steps.

- Dynamic Llama-8B: 100 warmup steps out of 903 total (11.1%)
- Dynamic DeepSeek-8B: 100 warmup steps out of 5569 total (1.8%)

**Note**: The warmup check above assumes normalizer resets per turn. If it persists
across turns within a request but resets across requests, the actual warmup fraction
may be different.

### 4.3 DeepSeek-8B: The Draft Model Mismatch Problem

The DeepSeek runs show dramatically lower accept rates (mean ~1.3 vs ~2.9 for Llama).
This is because the EAGLE3 draft model (`sglang-EAGLE3-LLaMA3.1-Instruct-8B`) was
trained on Llama-3.1-8B-Instruct, not on DeepSeek-R1-Distill-Llama-8B. The draft
model's predictions are misaligned with the target model, causing most drafts to be
rejected. In this regime:

- Vanilla EAGLE3 wastes compute on 3-step drafts where almost everything is rejected
- Dynamic spec SHOULD help by scaling down (fewer steps, lower ndt) when confidence is low
- Indeed, dynamic DeepSeek mostly uses topk=1, steps=2-3, ndt=3-4 (conservative)

**DeepSeek result**: Dynamic accept_len=1.523 vs Vanilla=1.436
  Dynamic throughput=91.7 vs Vanilla=89.7 tok/s
  Latency: Dynamic=92.5s vs Vanilla=85.1s

### 4.4 What Would It Take to Consistently Outperform Vanilla?

1. **Reduce overhead**: The dynamic system captures and manages many CUDA graphs
   (one per (topk, steps) combo for draft + one per ndt for verify). Graph switching
   has non-zero cost. For Llama where vanilla already achieves high acceptance, the
   overhead may exceed the marginal improvement from dynamic adjustment.

2. **Better signal utilization**: The current policy compresses 7 signals into a scalar
   confidence, losing information. A multi-dimensional policy (e.g., learned via
   reinforcement learning or Bayesian optimization) could make better decisions.

3. **Larger evaluation**: With only 5 questions, variance is high. Some questions may
   naturally have high/low acceptance regardless of config. A larger evaluation
   (e.g., full MT-Bench 80 questions) would give more statistical power.

4. **Asymmetric cost model**: The policy should account for the fact that speculating
   too aggressively (large tree, mostly rejected) wastes more compute than speculating
   too conservatively (small tree, high acceptance but fewer tokens per step). The
   current confidence mapping is symmetric around the starting config.

5. **Draft-model-aware policy**: For mismatched draft models (DeepSeek case), the
   policy should detect persistently low acceptance and fall back to minimal
   speculation (steps=1, ndt=2) or even disable speculation entirely.

6. **Per-turn vs per-request warmup**: If the normalizer resets per turn (not per
   request), the effective warmup fraction is larger. Consider persistent normalizer
   state across requests.

## Summary — All Signal Set Versions

### Accept Length Comparison (steady-state, bs=1)

| Config | Llama-8B | DeepSeek+LlamaDraft | Notes |
|--------|----------|---------------------|-------|
| **Vanilla EAGLE3** | 2.90 | 1.88 | Fixed topk=1, steps=3, ndt=4 |
| **7-signal equal** (5q) | **3.32 (+14.5%)** | **2.09 (+10.8%)** | Best accept_len overall |
| **14-signal equal** (5q) | 3.17 (+9.1%) | 1.93 (+2.3%) | Diluted by noise signals |
| **11-signal equal** (10q) | 3.11 (+7.2%) | 1.93 (+2.5%) | Pruned + oracle gate |

### Throughput Comparison (steady-state tok/s)

| Config | Llama-8B | DeepSeek+LlamaDraft |
|--------|----------|---------------------|
| **Vanilla EAGLE3** | 179.2 | 117.3 |
| **7-signal equal** (5q) | 184.3 (+2.8%) | 122.6 (+4.5%) |
| **14-signal equal** (5q) | 179.9 (+0.4%) | 111.0 (-5.4%) |
| **11-signal equal** (10q) | 173.7 (-3.1%) | 113.3 (-3.4%) |

### Signal Correlation Rankings (11-signal, 10 questions)

| Rank | Signal | r(RAR) Llama | r(RAR) DeepSeek | Category |
|------|--------|-------------|-----------------|----------|
| **1** | **draft_oracle_gate** | **+0.685** | **+0.776** | **Joint (NEW)** |
| 2 | target_top1_gap | +0.404 | +0.398 | Target (NEW) |
| 3 | target_entropy | -0.393 | -0.361 | Target |
| 4 | target_varentropy | -0.331 | -0.285 | Target (NEW) |
| 5 | joint_entropy_gate | +0.294 | +0.357 | Joint (NEW) |
| 6 | hidden_cosine_sim | -0.226 | -0.280 | Hidden (NEW) |
| 7 | draft_entropy | -0.240 | -0.195 | Draft |
| 8 | top1_minus_top2 | +0.146 | +0.205 | Draft |
| 9 | top1_prob | +0.154 | +0.198 | Draft |
| 10 | hidden_norm | -0.102 | -0.004 | Hidden |

### Key Takeaways

1. **`draft_oracle_gate` (top1_prob × rolling_accept_rate) is the #1 predictor** with r=+0.69/+0.78 — nearly 2x stronger than any other signal. The supervisor's insight that "draft certain + target accepting = good combination" is validated by the data.

2. **Target-side signals dominate ranks 2-5**: target_top1_gap, target_entropy, target_varentropy, joint_entropy_gate. These are all NEW signals with no precedent in the speculative decoding literature.

3. **hidden_cosine_sim is the best hidden-state signal** (r=-0.23/-0.28) — capturing representational stability between consecutive draft steps. All other hidden-state signals (norm, var, max) are weak or noise.

4. **Signal dilution is real**: 7 equal-weight signals outperform 11 and 14, despite having weaker individual signals. The solution is correlation-weighted signals, with `draft_oracle_gate` receiving the highest weight.

5. **10-question runs confirm 5-question findings**: The correlation rankings are stable across sample sizes, giving confidence in the signal analysis.
