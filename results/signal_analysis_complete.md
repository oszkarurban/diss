# Comprehensive Signal Analysis for Dynamic Speculative Decoding

## 1. Experiment Overview

| Config | Target Model | Draft Model | Signal Ver. | # Questions | Accept Len (mean) | Accept Len (SS) | Throughput SS (tok/s) | # Decode Batches | # Signal Steps |
|--------|-------------|-------------|-------------|-------------|-------------------|-----------------|----------------------|------------------|----------------|
| Vanilla Llama (5q) | Llama-3.1-8B-Instruct | EAGLE3-LLaMA3.1-8B | vanilla | 5 | 2.90 | 2.90 | 179.18 | 28 | 0 |
| Vanilla DS (5q) | DeepSeek-R1-Distill-Llama | EAGLE3-DeepSeek-R1-8B | vanilla | 5 | 1.44 | 1.44 | 89.69 | 132 | 0 |
| Vanilla DS+Llama (5q) | DeepSeek-R1-Distill-Llama | EAGLE3-LLaMA3.1-8B (misma | vanilla | 5 | 1.88 | 1.88 | 117.35 | 101 | 0 |
| 7sig Llama (5q) | Llama-3.1-8B-Instruct | EAGLE3-LLaMA3.1-8B | 7sig | 5 | 3.31 | 3.32 | 184.27 | 22 | 903 |
| 7sig DS (5q) | DeepSeek-R1-Distill-Llama | EAGLE3-DeepSeek-R1-8B | 7sig | 5 | 1.52 | 1.52 | 91.69 | 139 | 5569 |
| 7sig DS+Llama (5q) | DeepSeek-R1-Distill-Llama | EAGLE3-LLaMA3.1-8B (misma | 7sig | 5 | 2.09 | 2.09 | 122.63 | 113 | 4537 |
| 14sig Llama (5q) | Llama-3.1-8B-Instruct | EAGLE3-LLaMA3.1-8B | 14sig | 5 | 3.17 | 3.17 | 179.87 | 27 | 1091 |
| 14sig DS+Llama (5q) | DeepSeek-R1-Distill-Llama | EAGLE3-LLaMA3.1-8B (misma | 14sig | 5 | 1.95 | 1.93 | 112.11 | 95 | 3803 |
| 11sig Llama (10q) | Llama-3.1-8B-Instruct | EAGLE3-LLaMA3.1-8B | 11sig | 10 | 3.11 | 3.11 | 173.68 | 49 | 1970 |
| 11sig DS+Llama (10q) | DeepSeek-R1-Distill-Llama | EAGLE3-LLaMA3.1-8B (misma | 11sig | 10 | 1.93 | 1.93 | 113.28 | 184 | 7361 |

## 2. Signal-by-Signal Encyclopedia

### 2.1 draft_entropy -- Draft Quality [ACTIVE]

**What it is.** Shannon entropy of the draft model's output probability distribution at each draft step. Higher values indicate the draft model is uncertain about its prediction.

**Hypothesis.** When draft entropy is low, the draft model is confident and more likely to match the target model's greedy pick, leading to higher acceptance.

**Collection.** Phase: Draft. Computed in `eagle_worker.py:draft_forward()` after `probs = softmax(logits)` as `-sum(probs * log(probs))`. Written to pre-allocated CUDA graph signal buffer (`signal_draft_entropy`). CUDA graph compatible. Low cost (fused with existing softmax).

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | 1.5656 | 1.3879 | 0.0000 | 7.8810 | 1.2208 |
| 7sig DS (5q) | 5569 | 4.3403 | 1.9229 | 0.0051 | 8.9547 | 4.2996 |
| 7sig DS+Llama (5q) | 4537 | 3.4625 | 2.2990 | 0.0000 | 9.2071 | 3.0569 |
| 14sig Llama (5q) | 1091 | 1.3869 | 1.3071 | 0.0000 | 7.2275 | 1.0404 |
| 14sig DS+Llama (5q) | 3803 | 3.0100 | 1.8991 | 0.0001 | 8.1605 | 2.8457 |
| 11sig Llama (10q) | 1970 | 1.5329 | 1.4709 | 0.0000 | 8.0164 | 1.1278 |
| 11sig DS+Llama (10q) | 7361 | 2.9643 | 1.9133 | 0.0000 | 8.5052 | 2.7738 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | -0.1629 | 3/6 |
| 7sig DS (5q) | -0.1696 | 4/6 |
| 7sig DS+Llama (5q) | -0.0477 | 6/6 |
| 14sig Llama (5q) | -0.1936 | 5/13 |
| 14sig DS+Llama (5q) | -0.1902 | 7/13 |
| 11sig Llama (10q) | -0.2399 | 6/10 |
| 11sig DS+Llama (10q) | -0.1951 | 9/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | -0.6131 |
| 7sig DS (5q) | -0.6324 |
| 7sig DS+Llama (5q) | -0.6445 |
| 14sig Llama (5q) | -0.6260 |
| 14sig DS+Llama (5q) | -0.6110 |
| 11sig Llama (10q) | -0.5741 |
| 11sig DS+Llama (10q) | -0.5186 |

**Result.** Mean |r| with RAR = 0.1713. The relationship is weak and negative, consistent across all experiments. Range of r values: [-0.2399 to -0.0477].

**Impact.** Moderate predictor. Keep at standard weight in the policy.

---

### 2.2 top1_prob -- Draft Quality [ACTIVE]

**What it is.** The probability assigned to the top-1 draft token by the draft model. Ranges from 0 to 1.

**Hypothesis.** Higher top-1 probability means the draft model strongly favors one token, which should correlate with acceptance if target agrees.

**Collection.** Phase: Extend (carried from draft). Read from `spec_info.topk_p[:, 0]` in `_collect_signals()`. CUDA graph compatible (read from graph output). Negligible cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | 0.7743 | 0.2610 | 0.0444 | 1.0000 | 0.8982 |
| 7sig DS (5q) | 5569 | 0.4327 | 0.2824 | 0.0106 | 1.0000 | 0.3633 |
| 7sig DS+Llama (5q) | 4537 | 0.5668 | 0.3131 | 0.0053 | 1.0000 | 0.5541 |
| 14sig Llama (5q) | 1091 | 0.7703 | 0.2578 | 0.0459 | 1.0000 | 0.8919 |
| 14sig DS+Llama (5q) | 3803 | 0.5532 | 0.3011 | 0.0079 | 1.0000 | 0.5228 |
| 11sig Llama (10q) | 1970 | 0.7545 | 0.2690 | 0.0468 | 1.0000 | 0.8786 |
| 11sig DS+Llama (10q) | 7361 | 0.5441 | 0.3025 | 0.0072 | 1.0000 | 0.5109 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | 0.1669 | 2/6 |
| 7sig DS (5q) | 0.2503 | 3/6 |
| 7sig DS+Llama (5q) | 0.2813 | 3/6 |
| 14sig Llama (5q) | 0.1839 | 6/13 |
| 14sig DS+Llama (5q) | 0.2052 | 6/13 |
| 11sig Llama (10q) | 0.1541 | 8/10 |
| 11sig DS+Llama (10q) | 0.1977 | 8/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | 0.8063 |
| 7sig DS (5q) | 0.7782 |
| 7sig DS+Llama (5q) | 0.7823 |
| 14sig Llama (5q) | 0.5084 |
| 14sig DS+Llama (5q) | 0.5501 |
| 11sig Llama (10q) | 0.6719 |
| 11sig DS+Llama (10q) | 0.6786 |

**Result.** Mean |r| with RAR = 0.2056. The relationship is weak and positive, consistent across all experiments. Range of r values: [0.1541 to 0.2813].

**Impact.** Moderate predictor. Keep at standard weight in the policy.

---

### 2.3 top1_minus_top2 -- Draft Quality [ACTIVE]

**What it is.** Difference between the top-1 and top-2 draft token probabilities. Measures how decisively the draft model favors its top pick.

**Hypothesis.** A large gap suggests the draft model is decisive, not hedging between alternatives, which should predict acceptance.

**Collection.** Phase: Extend. Computed from `topk_p[:, 0] - topk_p[:, 1]` in `_collect_signals()`. Requires topk >= 2. CUDA graph compatible. Negligible cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | 0.6882 | 0.3428 | 0.0000 | 1.0000 | 0.8517 |
| 7sig DS (5q) | 5569 | 0.3237 | 0.3118 | 0.0000 | 1.0000 | 0.2051 |
| 7sig DS+Llama (5q) | 4537 | 0.4665 | 0.3613 | 0.0000 | 1.0000 | 0.4055 |
| 14sig Llama (5q) | 1091 | 0.6801 | 0.3422 | 0.0000 | 1.0000 | 0.8362 |
| 14sig DS+Llama (5q) | 3803 | 0.4406 | 0.3526 | 0.0000 | 1.0000 | 0.3581 |
| 11sig Llama (10q) | 1970 | 0.6617 | 0.3505 | 0.0000 | 1.0000 | 0.8141 |
| 11sig DS+Llama (10q) | 7361 | 0.4335 | 0.3507 | 0.0000 | 1.0000 | 0.3475 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | 0.1563 | 4/6 |
| 7sig DS (5q) | 0.2552 | 2/6 |
| 7sig DS+Llama (5q) | 0.3022 | 2/6 |
| 14sig Llama (5q) | 0.1785 | 7/13 |
| 14sig DS+Llama (5q) | 0.2152 | 5/13 |
| 11sig Llama (10q) | 0.1458 | 9/10 |
| 11sig DS+Llama (10q) | 0.2049 | 7/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | 0.8017 |
| 7sig DS (5q) | 0.7693 |
| 7sig DS+Llama (5q) | 0.7635 |
| 14sig Llama (5q) | 0.5001 |
| 14sig DS+Llama (5q) | 0.5437 |
| 11sig Llama (10q) | 0.6588 |
| 11sig DS+Llama (10q) | 0.6710 |

**Result.** Mean |r| with RAR = 0.2083. The relationship is weak and positive, consistent across all experiments. Range of r values: [0.1458 to 0.3022].

**Impact.** Moderate predictor. Keep at standard weight in the policy.

---

### 2.4 hidden_norm -- Representation [ACTIVE]

**What it is.** L2 norm of the draft model's hidden state vector, averaged across the sequence. Captures activation magnitude.

**Hypothesis.** Abnormally high or low hidden norms may indicate the model is in an unusual state, potentially leading to draft-target disagreement.

**Collection.** Phase: Extend. Computed from `torch.norm(hidden_states, dim=-1).mean()` in `_collect_signals()`. CUDA graph compatible. Low cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | 62.9338 | 10.1325 | 38.0000 | 108.0000 | 62.0000 |
| 7sig DS (5q) | 5569 | 96.2192 | 23.9267 | 44.7500 | 204.0000 | 92.0000 |
| 7sig DS+Llama (5q) | 4537 | 75.2486 | 12.0584 | 38.7500 | 142.0000 | 74.5000 |
| 14sig Llama (5q) | 1091 | 62.9574 | 10.2995 | 36.2500 | 108.5000 | 62.0000 |
| 14sig DS+Llama (5q) | 3803 | 77.4690 | 11.4622 | 38.2500 | 142.0000 | 76.5000 |
| 11sig Llama (10q) | 1970 | 63.4044 | 10.3828 | 33.5000 | 117.5000 | 62.5000 |
| 11sig DS+Llama (10q) | 7361 | 78.0163 | 11.5339 | 38.2500 | 142.0000 | 77.0000 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | -0.0458 | 6/6 |
| 7sig DS (5q) | -0.0689 | 5/6 |
| 7sig DS+Llama (5q) | -0.1165 | 4/6 |
| 14sig Llama (5q) | -0.1027 | 11/13 |
| 14sig DS+Llama (5q) | -0.0718 | 10/13 |
| 11sig Llama (10q) | -0.1018 | 10/10 |
| 11sig DS+Llama (10q) | -0.0042 | 10/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | -0.0673 |
| 7sig DS (5q) | -0.0580 |
| 7sig DS+Llama (5q) | 0.2060 |
| 14sig Llama (5q) | 0.0730 |
| 14sig DS+Llama (5q) | 0.2936 |
| 11sig Llama (10q) | -0.1530 |
| 11sig DS+Llama (10q) | 0.1201 |

**Result.** Mean |r| with RAR = 0.0731. The relationship is negligible and negative, consistent across all experiments. Range of r values: [-0.1165 to -0.0042].

**Impact.** Weak predictor. Consider downweighting or monitoring for improvement with more data.

---

### 2.5 hidden_cosine_sim -- Representation [ACTIVE]

**What it is.** Cosine similarity between consecutive hidden states from the draft model. Measures how much the representation changes between steps.

**Hypothesis.** High cosine similarity (stable representations) may indicate the model is in a predictable region, correlating with acceptance.

**Collection.** Phase: Extend. Computed from consecutive hidden states in `_collect_signals()`. CUDA graph compatible. Low cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.6165 | 0.1398 | 0.2310 | 1.0000 | 0.6074 |
| 14sig DS+Llama (5q) | 3803 | 0.7166 | 0.1357 | 0.2456 | 1.0000 | 0.6992 |
| 11sig Llama (10q) | 1970 | 0.6427 | 0.1666 | 0.0000 | 1.0000 | 0.6195 |
| 11sig DS+Llama (10q) | 7361 | 0.7549 | 0.1610 | 0.0000 | 1.0000 | 0.7227 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | -0.1475 | 8/13 |
| 14sig DS+Llama (5q) | -0.1807 | 8/13 |
| 11sig Llama (10q) | -0.2261 | 7/10 |
| 11sig DS+Llama (10q) | -0.2797 | 6/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | 0.0648 |
| 14sig DS+Llama (5q) | -0.0184 |
| 11sig Llama (10q) | -0.0453 |
| 11sig DS+Llama (10q) | -0.0588 |

**Result.** Mean |r| with RAR = 0.2085. The relationship is weak and negative, consistent across all experiments. Range of r values: [-0.2797 to -0.1475].

**Impact.** Moderate predictor. Keep at standard weight in the policy.

---

### 2.6 hidden_var -- Representation [REMOVED]

**What it is.** Variance of the draft model's hidden state activations across dimensions. Captures how spread out the activation values are.

**Hypothesis.** Higher variance might indicate a more informative hidden state, potentially correlating with prediction quality.

**Collection.** Phase: Extend. Computed from hidden state tensor variance. CUDA graph compatible. Low cost.

**Removal reason.** Removed: r=0.99 with hidden_norm (near-perfect redundancy). The variance and norm of a vector are algebraically related, so this signal adds no independent information.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.9937 | 0.3342 | 0.3223 | 2.8750 | 0.9336 |
| 14sig DS+Llama (5q) | 3803 | 1.4970 | 0.4514 | 0.3574 | 4.9375 | 1.4297 |
| 11sig Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | -0.0937 | 12/13 |
| 14sig DS+Llama (5q) | -0.0596 | 11/13 |
| 11sig Llama (10q) | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | 0.0852 |
| 14sig DS+Llama (5q) | 0.2928 |
| 11sig Llama (10q) | N/A |
| 11sig DS+Llama (10q) | N/A |

**Result.** Mean |r| with RAR = 0.0767. The relationship is negligible and negative, consistent across all experiments. Range of r values: [-0.0937 to -0.0596].

**Impact.** Removed from the signal set. r=0.99 with hidden_norm (near-perfect redundancy). The variance and norm of a vector are algebraically related, so this signal adds no independent information.

---

### 2.7 hidden_max -- Representation [REMOVED]

**What it is.** Maximum activation value in the draft model's hidden state vector. Captures peak activation magnitude.

**Hypothesis.** Extreme activation values might indicate confident internal representations.

**Collection.** Phase: Extend. Computed from max of hidden state tensor. CUDA graph compatible. Negligible cost.

**Removal reason.** Removed: r < 0.05 with acceptance rate (pure noise). The maximum single activation has no predictive relationship with draft quality.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 13.8669 | 5.2835 | 3.0625 | 33.2500 | 13.2500 |
| 14sig DS+Llama (5q) | 3803 | 16.2108 | 7.0478 | 3.6406 | 47.5000 | 15.0000 |
| 11sig Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | 0.0451 | 13/13 |
| 14sig DS+Llama (5q) | 0.0122 | 13/13 |
| 11sig Llama (10q) | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | 0.2257 |
| 14sig DS+Llama (5q) | 0.3454 |
| 11sig Llama (10q) | N/A |
| 11sig DS+Llama (10q) | N/A |

**Result.** Mean |r| with RAR = 0.0286. The relationship is negligible and positive, consistent across all experiments. Range of r values: [0.0122 to 0.0451].

**Impact.** Removed from the signal set. r < 0.05 with acceptance rate (pure noise). The maximum single activation has no predictive relationship with draft quality.

---

### 2.8 target_entropy -- Target Quality [ACTIVE]

**What it is.** Shannon entropy of the target model's output distribution at the verification step. Measures how uncertain the target model is.

**Hypothesis.** Low target entropy means the target model strongly favors one token. If the draft picked that token, acceptance is likely. High entropy means any draft token has a reasonable chance.

**Collection.** Phase: Verify. Computed from `softmax(logits) * log(softmax(logits))` on target logits in `eagle_info.py:verify()`. Eager mode (not in CUDA graph). Moderate cost (full softmax on target vocab).

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | 0.5678 | 0.4438 | 0.0001 | 2.8559 | 0.4565 |
| 7sig DS (5q) | 5569 | 1.2604 | 0.8034 | 0.0000 | 4.9608 | 1.1540 |
| 7sig DS+Llama (5q) | 4537 | 0.8171 | 0.6181 | 0.0000 | 4.3208 | 0.7029 |
| 14sig Llama (5q) | 1091 | 0.5315 | 0.4301 | 0.0000 | 2.8466 | 0.4321 |
| 14sig DS+Llama (5q) | 3803 | 0.8127 | 0.5841 | 0.0000 | 3.9742 | 0.7120 |
| 11sig Llama (10q) | 1970 | 0.5844 | 0.4580 | 0.0000 | 2.9982 | 0.4868 |
| 11sig DS+Llama (10q) | 7361 | 0.8335 | 0.5890 | 0.0001 | 4.6913 | 0.7359 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | -0.3724 | 1/6 |
| 7sig DS (5q) | -0.3827 | 1/6 |
| 7sig DS+Llama (5q) | -0.3644 | 1/6 |
| 14sig Llama (5q) | -0.3798 | 2/13 |
| 14sig DS+Llama (5q) | -0.3609 | 3/13 |
| 11sig Llama (10q) | -0.3929 | 3/10 |
| 11sig DS+Llama (10q) | -0.3614 | 3/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | -0.5872 |
| 7sig DS (5q) | -0.4242 |
| 7sig DS+Llama (5q) | -0.4711 |
| 14sig Llama (5q) | -0.7448 |
| 14sig DS+Llama (5q) | -0.6231 |
| 11sig Llama (10q) | -0.7246 |
| 11sig DS+Llama (10q) | -0.6616 |

**Result.** Mean |r| with RAR = 0.3735. The relationship is moderate and negative, consistent across all experiments. Range of r values: [-0.3929 to -0.3609].

**Impact.** Strong predictor. Recommended for upweighting in the policy confidence computation.

---

### 2.9 target_top1_gap -- Target Quality [ACTIVE]

**What it is.** Gap between the target model's top-1 and top-2 token probabilities. Analogous to top1_minus_top2 but on the target side.

**Hypothesis.** A large target gap means the target strongly prefers one token. If the draft model also picked it, acceptance follows.

**Collection.** Phase: Verify. Computed from target logits after softmax in `eagle_info.py:verify()`. Eager mode. Moderate cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.7417 | 0.1835 | 0.1869 | 1.0000 | 0.7614 |
| 14sig DS+Llama (5q) | 3803 | 0.6529 | 0.1980 | 0.0941 | 1.0000 | 0.6536 |
| 11sig Llama (10q) | 1970 | 0.7228 | 0.1902 | 0.0952 | 1.0000 | 0.7422 |
| 11sig DS+Llama (10q) | 7361 | 0.6475 | 0.2006 | 0.0405 | 1.0000 | 0.6524 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | 0.4354 | 1/13 |
| 14sig DS+Llama (5q) | 0.3976 | 1/13 |
| 11sig Llama (10q) | 0.4038 | 2/10 |
| 11sig DS+Llama (10q) | 0.3978 | 2/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | 0.7040 |
| 14sig DS+Llama (5q) | 0.5612 |
| 11sig Llama (10q) | 0.6937 |
| 11sig DS+Llama (10q) | 0.6263 |

**Result.** Mean |r| with RAR = 0.4086. The relationship is moderate and positive, consistent across all experiments. Range of r values: [0.3976 to 0.4354].

**Impact.** Strong predictor. Recommended for upweighting in the policy confidence computation.

---

### 2.10 target_varentropy -- Target Quality [ACTIVE]

**What it is.** Variance of the entropy across the target model's token-level distributions. Captures heterogeneity of target confidence across positions.

**Hypothesis.** Low varentropy indicates uniformly confident/uncertain predictions; high varentropy indicates mixed confidence, which may affect acceptance patterns.

**Collection.** Phase: Verify. Computed from target logits in `eagle_info.py:verify()`. Eager mode. Moderate cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.8232 | 0.7378 | 0.0002 | 6.3727 | 0.6321 |
| 14sig DS+Llama (5q) | 3803 | 1.6427 | 1.5649 | 0.0003 | 11.0542 | 1.1334 |
| 11sig Llama (10q) | 1970 | 0.9093 | 0.7747 | 0.0000 | 5.7746 | 0.7171 |
| 11sig DS+Llama (10q) | 7361 | 1.6749 | 1.5030 | 0.0008 | 12.4215 | 1.2071 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | -0.2854 | 4/13 |
| 14sig DS+Llama (5q) | -0.2871 | 4/13 |
| 11sig Llama (10q) | -0.3305 | 4/10 |
| 11sig DS+Llama (10q) | -0.2849 | 5/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | -0.6513 |
| 14sig DS+Llama (5q) | -0.5582 |
| 11sig Llama (10q) | -0.6446 |
| 11sig DS+Llama (10q) | -0.5732 |

**Result.** Mean |r| with RAR = 0.2970. The relationship is weak and negative, consistent across all experiments. Range of r values: [-0.3305 to -0.2849].

**Impact.** Moderate predictor. Keep at standard weight in the policy.

---

### 2.11 joint_entropy_gate -- Derived / Gate [ACTIVE]

**What it is.** Gating signal combining draft and target entropies. Computed as `sigmoid(target_entropy - draft_entropy) * min(draft_entropy, target_entropy) / max(draft_entropy, target_entropy)`.

**Hypothesis.** Captures the interaction between draft and target uncertainty. When both agree on confidence level, this should be higher and predict acceptance.

**Collection.** Phase: Derived. Computed in `_collect_signals()` from draft_entropy and target_entropy. No GPU cost beyond the constituent signals.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.3965 | 0.2394 | 0.0372 | 0.9998 | 0.3346 |
| 14sig DS+Llama (5q) | 3803 | 0.2146 | 0.1805 | 0.0254 | 0.9997 | 0.1528 |
| 11sig Llama (10q) | 1970 | 0.3741 | 0.2353 | 0.0374 | 0.9995 | 0.3156 |
| 11sig DS+Llama (10q) | 7361 | 0.2159 | 0.1802 | 0.0226 | 0.9995 | 0.1526 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | 0.3088 | 3/13 |
| 14sig DS+Llama (5q) | 0.3664 | 2/13 |
| 11sig Llama (10q) | 0.2942 | 5/10 |
| 11sig DS+Llama (10q) | 0.3567 | 4/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | 0.7315 |
| 14sig DS+Llama (5q) | 0.6752 |
| 11sig Llama (10q) | 0.7084 |
| 11sig DS+Llama (10q) | 0.6479 |

**Result.** Mean |r| with RAR = 0.3315. The relationship is moderate and positive, consistent across all experiments. Range of r values: [0.2942 to 0.3664].

**Impact.** Strong predictor. Recommended for upweighting in the policy confidence computation.

---

### 2.12 entropy_gap -- Derived [REMOVED (replaced)]

**What it is.** Simple difference: `target_entropy - draft_entropy`. Used in 7sig version.

**Hypothesis.** Negative values (draft more uncertain than target) should predict lower acceptance since the draft model is spreading probability over tokens the target does not consider.

**Collection.** Phase: Derived. Computed in `_collect_signals()`. Zero additional cost.

**Removal reason.** Replaced by entropy_gap_pos/entropy_gap_neg split in 14sig, then both halves removed. The signed gap conflates two different phenomena (draft overconfidence vs underconfidence).

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | -0.9978 | 1.2248 | -6.9416 | 1.3913 | -0.7370 |
| 7sig DS (5q) | 5569 | -3.0799 | 1.8732 | -8.8679 | 2.9566 | -3.0525 |
| 7sig DS+Llama (5q) | 4537 | -2.6455 | 2.2026 | -8.7626 | 2.4331 | -2.2573 |
| 14sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | 0.0496 | 5/6 |
| 7sig DS (5q) | 0.0100 | 6/6 |
| 7sig DS+Llama (5q) | -0.0525 | 5/6 |
| 14sig Llama (5q) | N/A | N/A |
| 14sig DS+Llama (5q) | N/A | N/A |
| 11sig Llama (10q) | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | 0.4820 |
| 7sig DS (5q) | 0.4672 |
| 7sig DS+Llama (5q) | 0.5405 |
| 14sig Llama (5q) | N/A |
| 14sig DS+Llama (5q) | N/A |
| 11sig Llama (10q) | N/A |
| 11sig DS+Llama (10q) | N/A |

**Result.** Mean |r| with RAR = 0.0374. The relationship is negligible and positive, inconsistent in sign across experiments. Range of r values: [-0.0525 to 0.0496].

**Impact.** Removed from the signal set. Replaced by entropy_gap_pos/entropy_gap_neg split in 14sig, then both halves removed. The signed gap conflates two different phenomena (draft overconfidence vs underconfidence).

---

### 2.13 entropy_gap_pos -- Derived [REMOVED]

**What it is.** Positive part of the entropy gap: `max(0, target_entropy - draft_entropy)`. Captures cases where the target is more uncertain than the draft.

**Hypothesis.** When the target is more uncertain than the draft (gap > 0), the draft's confident pick should often be accepted.

**Collection.** Phase: Derived. Computed in `_collect_signals()`. Zero cost.

**Removal reason.** Removed: too sparse. In practice, the draft model (smaller) is almost always more uncertain than the target, so this signal is zero for most steps.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.0708 | 0.1954 | 0.0000 | 1.4511 | 0.0000 |
| 14sig DS+Llama (5q) | 3803 | 0.0373 | 0.1607 | 0.0000 | 2.2366 | 0.0000 |
| 11sig Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | -0.1169 | 9/13 |
| 14sig DS+Llama (5q) | -0.0468 | 12/13 |
| 11sig Llama (10q) | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | -0.0928 |
| 14sig DS+Llama (5q) | -0.0179 |
| 11sig Llama (10q) | N/A |
| 11sig DS+Llama (10q) | N/A |

**Result.** Mean |r| with RAR = 0.0819. The relationship is negligible and negative, consistent across all experiments. Range of r values: [-0.1169 to -0.0468].

**Impact.** Removed from the signal set. too sparse. In practice, the draft model (smaller) is almost always more uncertain than the target, so this signal is zero for most steps.

---

### 2.14 entropy_gap_neg -- Derived [REMOVED]

**What it is.** Negative part of the entropy gap: `max(0, draft_entropy - target_entropy)`. Captures cases where the draft is more uncertain than the target.

**Hypothesis.** Larger values (draft much more uncertain) should predict lower acceptance.

**Collection.** Phase: Derived. Computed in `_collect_signals()`. Zero cost.

**Removal reason.** Removed: r=0.96 with draft_entropy. Since draft entropy is almost always higher than target entropy, `max(0, draft_entropy - target_entropy)` approximately equals `draft_entropy - constant`, making it a near-linear transform of draft_entropy.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | 1091 | 0.9262 | 1.0597 | 0.0000 | 5.8785 | 0.5943 |
| 14sig DS+Llama (5q) | 3803 | 2.2346 | 1.7283 | 0.0000 | 7.9588 | 1.9968 |
| 11sig Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A | N/A | N/A | N/A | N/A |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | -0.1062 | 10/13 |
| 14sig DS+Llama (5q) | -0.0914 | 9/13 |
| 11sig Llama (10q) | N/A | N/A |
| 11sig DS+Llama (10q) | N/A | N/A |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | -0.4870 |
| 14sig DS+Llama (5q) | -0.4625 |
| 11sig Llama (10q) | N/A |
| 11sig DS+Llama (10q) | N/A |

**Result.** Mean |r| with RAR = 0.0988. The relationship is negligible and negative, consistent across all experiments. Range of r values: [-0.1062 to -0.0914].

**Impact.** Removed from the signal set. r=0.96 with draft_entropy. Since draft entropy is almost always higher than target entropy, `max(0, draft_entropy - target_entropy)` approximately equals `draft_entropy - constant`, making it a near-linear transform of draft_entropy.

---

### 2.15 draft_oracle_gate -- Derived / Gate [ACTIVE]

**What it is.** Gating signal that combines draft confidence with the empirical acceptance rate: `top1_prob * rolling_accept_rate`. Captures when the draft model is confident AND historically accurate.

**Hypothesis.** Should be a strong predictor because it directly multiplies confidence with track record. High values indicate a confident draft model with a good acceptance history.

**Collection.** Phase: Derived. Computed in `_collect_signals()` from top1_prob and rolling_accept_rate. Zero cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 14sig DS+Llama (5q) | N/A | N/A | N/A | N/A | N/A | N/A |
| 11sig Llama (10q) | 1970 | 0.4163 | 0.2014 | 0.0199 | 0.8876 | 0.4206 |
| 11sig DS+Llama (10q) | 7361 | 0.1549 | 0.1542 | 0.0005 | 0.8221 | 0.1008 |

**Correlation with acceptance (rolling_accept_rate):**

| Experiment | r(signal, RAR) | Rank (of N signals) |
|------------|----------------|---------------------|
| 7sig Llama (5q) | N/A | N/A |
| 7sig DS (5q) | N/A | N/A |
| 7sig DS+Llama (5q) | N/A | N/A |
| 14sig Llama (5q) | N/A | N/A |
| 14sig DS+Llama (5q) | N/A | N/A |
| 11sig Llama (10q) | 0.6849 | 1/10 |
| 11sig DS+Llama (10q) | 0.7761 | 1/10 |

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | N/A |
| 7sig DS (5q) | N/A |
| 7sig DS+Llama (5q) | N/A |
| 14sig Llama (5q) | N/A |
| 14sig DS+Llama (5q) | N/A |
| 11sig Llama (10q) | 0.7587 |
| 11sig DS+Llama (10q) | 0.7281 |

**Result.** Mean |r| with RAR = 0.7305. The relationship is strong and positive, consistent across all experiments. Range of r values: [0.6849 to 0.7761].

**Impact.** Strong predictor. Recommended for upweighting in the policy confidence computation.

---

### 2.16 rolling_accept_rate -- Empirical / EMA [ACTIVE]

**What it is.** Exponential moving average of the per-step acceptance rate, updated after each verification: `alpha * step_rate + (1-alpha) * prev_rate`.

**Hypothesis.** Acts as an adaptive baseline. When acceptance rate is trending high, the policy should be more aggressive (larger trees); when trending low, more conservative.

**Collection.** Phase: Verify. Updated in `eagle_info.py` per-request loop via EMA. Eager mode. Negligible cost.

**Raw data across experiments:**

| Experiment | N steps | Mean | Std | Min | Max | Median |
|------------|---------|------|-----|-----|-----|--------|
| 7sig Llama (5q) | 903 | 0.5344 | 0.1633 | 0.0695 | 1.0000 | 0.5487 |
| 7sig DS (5q) | 5569 | 0.1631 | 0.1598 | 0.0000 | 1.0000 | 0.1167 |
| 7sig DS+Llama (5q) | 4537 | 0.2838 | 0.1842 | 0.0018 | 1.0000 | 0.2453 |
| 14sig Llama (5q) | 1091 | 0.5711 | 0.1612 | 0.0795 | 0.9200 | 0.5876 |
| 14sig DS+Llama (5q) | 3803 | 0.2701 | 0.1643 | 0.0018 | 0.9316 | 0.2415 |
| 11sig Llama (10q) | 1970 | 0.5433 | 0.1553 | 0.0773 | 0.8883 | 0.5594 |
| 11sig DS+Llama (10q) | 7361 | 0.2667 | 0.1629 | 0.0029 | 0.8225 | 0.2401 |

**Correlation with acceptance (rolling_accept_rate):**

N/A (this IS the acceptance rate signal).

**Correlation with confidence:**

| Experiment | r(signal, confidence) |
|------------|----------------------|
| 7sig Llama (5q) | 0.4535 |
| 7sig DS (5q) | 0.4831 |
| 7sig DS+Llama (5q) | 0.4148 |
| 14sig Llama (5q) | 0.4339 |
| 14sig DS+Llama (5q) | 0.4075 |
| 11sig Llama (10q) | 0.4832 |
| 11sig DS+Llama (10q) | 0.4988 |

**Result.** Mean |r| with RAR = 1.0000. The relationship is strong and positive, consistent across all experiments. Range of r values: [1.0000 to 1.0000].

**Impact.** Strong predictor. Recommended for upweighting in the policy confidence computation.

---

## 3. Signal Interaction Analysis

Inter-signal Pearson correlation matrices computed on the 11sig datasets (most signals available).

### Correlation Matrix: 11sig Llama (10q)

| Signal | draft_entrop | top1_prob | top1_minus_t | hidden_norm | hidden_cosin | target_entro | target_top1_ | target_varen | joint_entrop | draft_oracle | rolling_acce |
|--------|------|------|------|------|------|------|------|------|------|------|------|
| draft_entrop | +1.00 | -0.18 | -0.16 | +0.09 | +0.07 | +0.46 | -0.35 | +0.50 | -0.78 | -0.26 | -0.24 |
| top1_prob    | -0.18 | +1.00 | +0.97 | -0.31 | -0.08 | -0.28 | +0.24 | -0.26 | +0.22 | +0.80 | +0.15 |
| top1_minus_t | -0.16 | +0.97 | +1.00 | -0.29 | -0.07 | -0.25 | +0.23 | -0.22 | +0.21 | +0.78 | +0.15 |
| hidden_norm  | +0.09 | -0.31 | -0.29 | +1.00 | +0.05 | +0.20 | -0.17 | +0.19 | -0.12 | -0.29 | -0.10 |
| hidden_cosin | +0.07 | -0.08 | -0.07 | +0.05 | +1.00 | +0.24 | -0.22 | +0.22 | -0.07 | -0.17 | -0.23 |
| target_entro | +0.46 | -0.28 | -0.25 | +0.20 | +0.24 | +1.00 | -0.88 | +0.88 | -0.64 | -0.41 | -0.39 |
| target_top1_ | -0.35 | +0.24 | +0.23 | -0.17 | -0.22 | -0.88 | +1.00 | -0.63 | +0.62 | +0.39 | +0.40 |
| target_varen | +0.50 | -0.26 | -0.22 | +0.19 | +0.22 | +0.88 | -0.63 | +1.00 | -0.57 | -0.36 | -0.33 |
| joint_entrop | -0.78 | +0.22 | +0.21 | -0.12 | -0.07 | -0.64 | +0.62 | -0.57 | +1.00 | +0.33 | +0.29 |
| draft_oracle | -0.26 | +0.80 | +0.78 | -0.29 | -0.17 | -0.41 | +0.39 | -0.36 | +0.33 | +1.00 | +0.68 |
| rolling_acce | -0.24 | +0.15 | +0.15 | -0.10 | -0.23 | -0.39 | +0.40 | -0.33 | +0.29 | +0.68 | +1.00 |

**Redundancy pairs (|r| > 0.7) for 11sig Llama (10q):**

| Signal A | Signal B | r |
|----------|----------|---|
| top1_prob | top1_minus_top2 | 0.9732 |
| target_entropy | target_varentropy | 0.8819 |
| target_entropy | target_top1_gap | -0.8762 |
| top1_prob | draft_oracle_gate | 0.7957 |
| draft_entropy | joint_entropy_gate | -0.7806 |
| top1_minus_top2 | draft_oracle_gate | 0.7770 |

### Correlation Matrix: 11sig DS+Llama (10q)

| Signal | draft_entrop | top1_prob | top1_minus_t | hidden_norm | hidden_cosin | target_entro | target_top1_ | target_varen | joint_entrop | draft_oracle | rolling_acce |
|--------|------|------|------|------|------|------|------|------|------|------|------|
| draft_entrop | +1.00 | -0.17 | -0.15 | -0.04 | +0.15 | +0.34 | -0.24 | +0.38 | -0.75 | -0.24 | -0.20 |
| top1_prob    | -0.17 | +1.00 | +0.97 | -0.05 | -0.10 | -0.22 | +0.20 | -0.21 | +0.23 | +0.68 | +0.20 |
| top1_minus_t | -0.15 | +0.97 | +1.00 | -0.05 | -0.10 | -0.21 | +0.20 | -0.18 | +0.23 | +0.68 | +0.20 |
| hidden_norm  | -0.04 | -0.05 | -0.05 | +1.00 | -0.03 | +0.01 | -0.04 | -0.04 | -0.02 | -0.01 | -0.00 |
| hidden_cosin | +0.15 | -0.10 | -0.10 | -0.03 | +1.00 | +0.19 | -0.17 | +0.18 | -0.17 | -0.24 | -0.28 |
| target_entro | +0.34 | -0.22 | -0.21 | +0.01 | +0.19 | +1.00 | -0.82 | +0.88 | -0.53 | -0.37 | -0.36 |
| target_top1_ | -0.24 | +0.20 | +0.20 | -0.04 | -0.17 | -0.82 | +1.00 | -0.53 | +0.52 | +0.40 | +0.40 |
| target_varen | +0.38 | -0.21 | -0.18 | -0.04 | +0.18 | +0.88 | -0.53 | +1.00 | -0.46 | -0.31 | -0.28 |
| joint_entrop | -0.75 | +0.23 | +0.23 | -0.02 | -0.17 | -0.53 | +0.52 | -0.46 | +1.00 | +0.40 | +0.36 |
| draft_oracle | -0.24 | +0.68 | +0.68 | -0.01 | -0.24 | -0.37 | +0.40 | -0.31 | +0.40 | +1.00 | +0.78 |
| rolling_acce | -0.20 | +0.20 | +0.20 | -0.00 | -0.28 | -0.36 | +0.40 | -0.28 | +0.36 | +0.78 | +1.00 |

**Redundancy pairs (|r| > 0.7) for 11sig DS+Llama (10q):**

| Signal A | Signal B | r |
|----------|----------|---|
| top1_prob | top1_minus_top2 | 0.9697 |
| target_entropy | target_varentropy | 0.8771 |
| target_entropy | target_top1_gap | -0.8166 |
| draft_oracle_gate | rolling_accept_rate | 0.7761 |
| draft_entropy | joint_entropy_gate | -0.7510 |

## 4. Policy Performance Across Versions

### Config Distributions

| Experiment | Mean topk | Mean steps | Mean ndt | Vanilla config (1,3,4) % | Mean confidence |
|------------|-----------|------------|----------|--------------------------|-----------------|
| 7sig Llama (5q) | 2.06 | 3.64 | 5.32 | 23.03% | 0.6671 |
| 7sig DS (5q) | 1.15 | 2.78 | 3.87 | 55.83% | 0.4452 |
| 7sig DS+Llama (5q) | 1.48 | 3.18 | 4.48 | 51.29% | 0.5468 |
| 14sig Llama (5q) | 1.64 | 3.37 | 4.76 | 40.33% | 0.5905 |
| 14sig DS+Llama (5q) | 1.35 | 3.09 | 4.31 | 57.43% | 0.5236 |
| 11sig Llama (10q) | 1.68 | 3.32 | 4.76 | 36.29% | 0.5802 |
| 11sig DS+Llama (10q) | 1.33 | 2.96 | 4.17 | 49.34% | 0.4916 |

### Signal Dilution Effect

As the number of signals increases from 7 to 14 to 11, the policy's confidence
computation averages over more normalized signals. This creates a **dilution effect**:

- **7sig**: Fewer signals mean each signal has higher individual weight in the
  confidence score. The policy can respond more strongly to any single signal change.
- **14sig**: With 14 signals (many weakly correlated or redundant), the confidence
  score becomes an average of many noisy inputs. Strong signals like `rolling_accept_rate`
  get diluted by noise signals like `hidden_max` (r < 0.05). This reduces policy
  responsiveness and can cause it to default to vanilla configs more often.
- **11sig**: After pruning the 3 weakest/most redundant signals (hidden_var, hidden_max,
  entropy_gap_neg) and adding the targeted `draft_oracle_gate`, the signal set is leaner.
  Each remaining signal carries more weight, and the gate signals provide multiplicative
  (not just additive) information.

### Throughput Comparison (Steady-State)

| Config | SS Throughput (tok/s) | SS Accept Len | vs Vanilla Baseline |
|--------|---------------------|---------------|---------------------|
| Vanilla Llama (5q) | 179.18 | 2.90 | +0.0% |
| Vanilla DS (5q) | 89.69 | 1.44 | +0.0% |
| Vanilla DS+Llama (5q) | 117.35 | 1.88 | +0.0% |
| 7sig Llama (5q) | 184.27 | 3.32 | +2.8% |
| 7sig DS (5q) | 91.69 | 1.52 | +2.2% |
| 7sig DS+Llama (5q) | 122.63 | 2.09 | +4.5% |
| 14sig Llama (5q) | 179.87 | 3.17 | +0.4% |
| 14sig DS+Llama (5q) | 112.11 | 1.93 | -4.5% |
| 11sig Llama (10q) | 173.68 | 3.11 | -3.1% |
| 11sig DS+Llama (10q) | 113.28 | 1.93 | -3.5% |

## 5. Recommended Signal Configuration

### Final Signal Ranking by Consensus |r| with Rolling Accept Rate

Consensus |r| is the mean of |r| across all experiments where the signal is present.

| Rank | Signal | Consensus |r| | # Experiments | Status | Tier |
|------|--------|----------------|---------------|--------|------|
| 1 | draft_oracle_gate | 0.7305 | 2 | ACTIVE | Tier 1 (high weight) |
| 2 | target_top1_gap | 0.4086 | 4 | ACTIVE | Tier 1 (high weight) |
| 3 | target_entropy | 0.3735 | 7 | ACTIVE | Tier 2 (standard weight) |
| 4 | joint_entropy_gate | 0.3315 | 4 | ACTIVE | Tier 2 (standard weight) |
| 5 | target_varentropy | 0.2970 | 4 | ACTIVE | Tier 2 (standard weight) |
| 6 | hidden_cosine_sim | 0.2085 | 4 | ACTIVE | Tier 2 (standard weight) |
| 7 | top1_minus_top2 | 0.2083 | 7 | ACTIVE | Tier 2 (standard weight) |
| 8 | top1_prob | 0.2056 | 7 | ACTIVE | Tier 2 (standard weight) |
| 9 | draft_entropy | 0.1713 | 7 | ACTIVE | Tier 3 (low weight) |
| 10 | entropy_gap_neg | 0.0988 | 2 | REMOVED | Tier 4 (candidate for removal) |
| 11 | entropy_gap_pos | 0.0819 | 2 | REMOVED | Tier 4 (candidate for removal) |
| 12 | hidden_var | 0.0767 | 2 | REMOVED | Tier 4 (candidate for removal) |
| 13 | hidden_norm | 0.0731 | 7 | ACTIVE | Tier 4 (candidate for removal) |
| 14 | entropy_gap | 0.0374 | 3 | REMOVED (replaced) | Tier 4 (candidate for removal) |
| 15 | hidden_max | 0.0286 | 2 | REMOVED | Tier 4 (candidate for removal) |

### Proposed Weight Tiers

Based on consensus |r| and cross-experiment consistency:

**Tier 1 (High Weight, 1.5x-2x):** Signals with consensus |r| > 0.4. These are the
strongest predictors and should dominate the confidence computation.

- `draft_oracle_gate`
- `target_top1_gap`

**Tier 2 (Standard Weight, 1.0x):** Signals with consensus |r| between 0.2 and 0.4.
Solid contributors that add independent information.

- `target_entropy`
- `joint_entropy_gate`
- `target_varentropy`
- `hidden_cosine_sim`
- `top1_minus_top2`
- `top1_prob`

**Tier 3 (Low Weight, 0.5x):** Signals with consensus |r| between 0.1 and 0.2.
Marginal contributors; keep for diversity but downweight.

- `draft_entropy`

**Tier 4 (Candidate for Removal):** Signals with consensus |r| < 0.1 or already removed.

- `entropy_gap_neg`
- `entropy_gap_pos`
- `hidden_var`
- `hidden_norm`
- `entropy_gap`
- `hidden_max`

### rolling_accept_rate (Special)

`rolling_accept_rate` is both a signal and the primary target variable. It serves as
the adaptive baseline and should always be included with Tier 1 weight. Its EMA nature
provides temporal smoothing that complements the instantaneous signals above.

