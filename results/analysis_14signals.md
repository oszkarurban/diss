# 14-Signal Dynamic Speculative Decoding: Comprehensive Analysis

**Setup**: 5 MT-Bench questions, 2 turns each (10 turns total), bs=1, temperature=0, A100 GPU

**14-signal runs** add 7 new signals on top of the original 7: hidden_cosine_sim, hidden_var, hidden_max, target_top1_gap, target_varentropy, joint_entropy_gate, entropy_gap_pos, entropy_gap_neg (entropy_gap split into positive/negative components).

## Section 1: Accept Length and Throughput Comparison

### 1.1 Accept Length (all batches)

| Config | N batches | Mean | Std | Min | Max | Median |
|--------|-----------|------|-----|-----|-----|--------|
| Vanilla Llama-8B | 28 | 2.9046 | 0.2502 | 2.2300 | 3.3500 | 2.9650 |
| 7-sig Dynamic Llama-8B | 22 | 3.3123 | 0.4147 | 2.5200 | 4.0800 | 3.3250 |
| 14-sig Dynamic Llama-8B | 27 | 3.1693 | 0.3763 | 2.3300 | 4.2800 | 3.1700 |
| Vanilla DeepSeek-8B | 101 | 1.8823 | 0.4514 | 1.2300 | 3.6000 | 1.7500 |
| 7-sig Dynamic DeepSeek-8B | 113 | 2.0852 | 0.6458 | 1.3000 | 4.7000 | 1.9000 |
| 14-sig Dynamic DeepSeek-8B | 95 | 1.9496 | 0.4577 | 1.3500 | 3.9500 | 1.8200 |

### 1.2 Accept Length (steady-state, excluding cold-start batches)

| Config | N batches | Mean | Std | Median |
|--------|-----------|------|-----|--------|
| Vanilla Llama-8B | 27 | 2.9019 | 0.2544 | 2.9500 |
| 7-sig Dynamic Llama-8B | 21 | 3.3214 | 0.4223 | 3.4200 |
| 14-sig Dynamic Llama-8B | 26 | 3.1669 | 0.3833 | 3.1450 |
| Vanilla DeepSeek-8B | 100 | 1.8821 | 0.4536 | 1.7400 |
| 7-sig Dynamic DeepSeek-8B | 112 | 2.0851 | 0.6487 | 1.9000 |
| 14-sig Dynamic DeepSeek-8B | 93 | 1.9259 | 0.4122 | 1.8200 |

### 1.3 Throughput (steady-state tok/s)

| Config | N batches | Mean | Std | Median | Max |
|--------|-----------|------|-----|--------|-----|
| Vanilla Llama-8B | 27 | 179.18 | 16.41 | 182.29 | 209.29 |
| 7-sig Dynamic Llama-8B | 21 | 184.27 | 30.65 | 190.19 | 232.63 |
| 14-sig Dynamic Llama-8B | 26 | 179.87 | 20.57 | 181.68 | 232.37 |
| Vanilla DeepSeek-8B | 100 | 117.35 | 27.97 | 109.11 | 224.83 |
| 7-sig Dynamic DeepSeek-8B | 112 | 122.63 | 35.58 | 113.57 | 264.79 |
| 14-sig Dynamic DeepSeek-8B | 93 | 112.11 | 22.86 | 106.64 | 200.59 |

### 1.4 Delta Summary

**Llama-8B (matched draft model)**:

| Metric | Vanilla | 7-sig Dynamic | 14-sig Dynamic | 7sig vs Vanilla | 14sig vs Vanilla | 14sig vs 7sig |
|--------|---------|---------------|----------------|-----------------|------------------|---------------|
| Accept len (ss) | 2.9019 | 3.3214 | 3.1669 | +14.5% | +9.1% | -4.7% |
| Throughput (ss) | 179.1778 | 184.2743 | 179.8665 | +2.8% | +0.4% | -2.4% |
| Decode steps | 28 | 22 | 27 | — | — | — |
| Signal steps | 0 | 1091 (7sig: from prev analysis) | 1091 | — | — | — |

**DeepSeek-8B + LlamaDraft (mismatched draft model)**:

| Metric | Vanilla | 7-sig Dynamic | 14-sig Dynamic | 7sig vs Vanilla | 14sig vs Vanilla | 14sig vs 7sig |
|--------|---------|---------------|----------------|-----------------|------------------|---------------|
| Accept len (ss) | 1.8821 | 2.0851 | 1.9259 | +10.8% | +2.3% | -7.6% |
| Throughput (ss) | 117.3459 | 122.6268 | 112.1142 | +4.5% | -4.5% | -8.6% |
| Decode steps | 101 | 113 | 95 | — | — | — |

## Section 2: Per-Signal Correlation Rankings

### 2.1 14-sig Llama-8B — Signal Statistics (1091 steps)

| Signal | N | Min | Max | Mean | Std | Median |
|--------|---|-----|-----|------|-----|--------|
| draft_entropy | 1091 | 0.0000 | 7.2275 | 1.3869 | 1.3065 | 1.0404 |
| top1_prob | 1091 | 0.0459 | 1.0000 | 0.7703 | 0.2577 | 0.8919 |
| top1_minus_top2 | 1091 | 0.0000 | 1.0000 | 0.6801 | 0.3420 | 0.8362 |
| hidden_norm | 1091 | 36.2500 | 108.5000 | 62.9574 | 10.2948 | 62.0000 |
| hidden_cosine_sim | 1091 | 0.2310 | 1.0000 | 0.6165 | 0.1397 | 0.6074 |
| hidden_var | 1091 | 0.3223 | 2.8750 | 0.9937 | 0.3340 | 0.9336 |
| hidden_max | 1091 | 3.0625 | 33.2500 | 13.8669 | 5.2811 | 13.2500 |
| target_entropy | 1091 | 0.0000 | 2.8466 | 0.5315 | 0.4299 | 0.4321 |
| target_top1_gap | 1091 | 0.1869 | 1.0000 | 0.7417 | 0.1835 | 0.7614 |
| target_varentropy | 1091 | 0.0002 | 6.3727 | 0.8232 | 0.7375 | 0.6321 |
| joint_entropy_gate | 1091 | 0.0372 | 0.9998 | 0.3965 | 0.2393 | 0.3346 |
| entropy_gap_pos | 1091 | 0.0000 | 1.4511 | 0.0708 | 0.1953 | 0.0000 |
| entropy_gap_neg | 1091 | 0.0000 | 5.8785 | 0.9262 | 1.0592 | 0.5943 |
| rolling_accept_rate | 1091 | 0.0795 | 0.9200 | 0.5711 | 0.1611 | 0.5876 |

### 2.2 14-sig Llama-8B — Correlation with rolling_accept_rate (ranked by |r|)

| Rank | Signal | r(signal, RAR) | |r| | Signal type |
|------|--------|----------------|-----|-------------|
| 1 | rolling_accept_rate | +1.0000 | 1.0000 | SELF |
| 2 | target_top1_gap | +0.4354 | 0.4354 | NEW |
| 3 | target_entropy | -0.3798 | 0.3798 | OLD |
| 4 | joint_entropy_gate | +0.3088 | 0.3088 | NEW |
| 5 | target_varentropy | -0.2854 | 0.2854 | NEW |
| 6 | draft_entropy | -0.1936 | 0.1936 | OLD |
| 7 | top1_prob | +0.1839 | 0.1839 | OLD |
| 8 | top1_minus_top2 | +0.1785 | 0.1785 | OLD |
| 9 | hidden_cosine_sim | -0.1475 | 0.1475 | NEW |
| 10 | entropy_gap_pos | -0.1169 | 0.1169 | NEW |
| 11 | entropy_gap_neg | -0.1062 | 0.1062 | NEW |
| 12 | hidden_norm | -0.1027 | 0.1027 | OLD |
| 13 | hidden_var | -0.0937 | 0.0937 | NEW |
| 14 | hidden_max | +0.0451 | 0.0451 | NEW |

### 2.3 14-sig Llama-8B — Correlation with confidence (ranked by |r|)

| Rank | Signal | r(signal, confidence) | |r| | Signal type |
|------|--------|----------------------|-----|-------------|
| 1 | target_entropy | -0.7448 | 0.7448 | OLD |
| 2 | joint_entropy_gate | +0.7315 | 0.7315 | NEW |
| 3 | target_top1_gap | +0.7040 | 0.7040 | NEW |
| 4 | target_varentropy | -0.6513 | 0.6513 | NEW |
| 5 | draft_entropy | -0.6260 | 0.6260 | OLD |
| 6 | top1_prob | +0.5084 | 0.5084 | OLD |
| 7 | top1_minus_top2 | +0.5001 | 0.5001 | OLD |
| 8 | entropy_gap_neg | -0.4870 | 0.4870 | NEW |
| 9 | rolling_accept_rate | +0.4339 | 0.4339 | OLD |
| 10 | hidden_max | +0.2257 | 0.2257 | NEW |
| 11 | entropy_gap_pos | -0.0928 | 0.0928 | NEW |
| 12 | hidden_var | +0.0852 | 0.0852 | NEW |
| 13 | hidden_norm | +0.0730 | 0.0730 | OLD |
| 14 | hidden_cosine_sim | +0.0648 | 0.0648 | NEW |

### 2.4 14-sig Llama-8B — Config Distribution

**chosen_topk**:
| Value | Count | Pct |
|-------|-------|-----|
| 1 | 500 | 45.8% |
| 2 | 482 | 44.2% |
| 3 | 109 | 10.0% |

**chosen_num_steps**:
| Value | Count | Pct |
|-------|-------|-----|
| 2 | 60 | 5.5% |
| 3 | 564 | 51.7% |
| 4 | 465 | 42.6% |
| 5 | 2 | 0.2% |

**chosen_num_draft_tokens**:
| Value | Count | Pct |
|-------|-------|-----|
| 3 | 60 | 5.5% |
| 4 | 440 | 40.3% |
| 5 | 321 | 29.4% |
| 6 | 239 | 21.9% |
| 7 | 31 | 2.8% |

**Config tuple (topk, steps, ndt) distribution**:
| (topk, steps, ndt) | Count | Pct |
|--------------------|-------|-----|
| (1, 3, 4) | 440 | 40.3% |
| (2, 4, 5) | 197 | 18.1% |
| (2, 4, 6) | 161 | 14.8% |
| (2, 3, 5) | 124 | 11.4% |
| (3, 4, 6) | 78 | 7.1% |
| (1, 2, 3) | 60 | 5.5% |
| (3, 4, 7) | 29 | 2.7% |
| (3, 5, 7) | 2 | 0.2% |

### 2.5 14-sig Llama-8B — Confidence Distribution

| Bucket | Count | Pct |
|--------|-------|-----|
| [0.0, 0.1) | 0 | 0.0% |
| [0.1, 0.2) | 1 | 0.1% |
| [0.2, 0.3) | 18 | 1.6% |
| [0.3, 0.4) | 67 | 6.1% |
| [0.4, 0.5) | 184 | 16.9% |
| [0.5, 0.6) | 276 | 25.3% |
| [0.6, 0.7) | 309 | 28.3% |
| [0.7, 0.8) | 194 | 17.8% |
| [0.8, 0.9) | 42 | 3.8% |
| [0.9, 1.0) | 0 | 0.0% |

### 2.1 14-sig DeepSeek-8B+LlamaDraft — Signal Statistics (3803 steps)

| Signal | N | Min | Max | Mean | Std | Median |
|--------|---|-----|-----|------|-----|--------|
| draft_entropy | 3803 | 0.0001 | 8.1605 | 3.0100 | 1.8988 | 2.8457 |
| top1_prob | 3803 | 0.0079 | 1.0000 | 0.5532 | 0.3011 | 0.5228 |
| top1_minus_top2 | 3803 | 0.0000 | 1.0000 | 0.4406 | 0.3526 | 0.3581 |
| hidden_norm | 3803 | 38.2500 | 142.0000 | 77.4690 | 11.4607 | 76.5000 |
| hidden_cosine_sim | 3803 | 0.2456 | 1.0000 | 0.7166 | 0.1357 | 0.6992 |
| hidden_var | 3803 | 0.3574 | 4.9375 | 1.4970 | 0.4514 | 1.4297 |
| hidden_max | 3803 | 3.6406 | 47.5000 | 16.2108 | 7.0468 | 15.0000 |
| target_entropy | 3803 | 0.0000 | 3.9742 | 0.8127 | 0.5840 | 0.7120 |
| target_top1_gap | 3803 | 0.0941 | 1.0000 | 0.6529 | 0.1980 | 0.6536 |
| target_varentropy | 3803 | 0.0003 | 11.0542 | 1.6427 | 1.5647 | 1.1334 |
| joint_entropy_gate | 3803 | 0.0254 | 0.9997 | 0.2146 | 0.1805 | 0.1528 |
| entropy_gap_pos | 3803 | 0.0000 | 2.2366 | 0.0373 | 0.1607 | 0.0000 |
| entropy_gap_neg | 3803 | 0.0000 | 7.9588 | 2.2346 | 1.7281 | 1.9968 |
| rolling_accept_rate | 3803 | 0.0018 | 0.9316 | 0.2701 | 0.1642 | 0.2415 |

### 2.2 14-sig DeepSeek-8B+LlamaDraft — Correlation with rolling_accept_rate (ranked by |r|)

| Rank | Signal | r(signal, RAR) | |r| | Signal type |
|------|--------|----------------|-----|-------------|
| 1 | rolling_accept_rate | +1.0000 | 1.0000 | SELF |
| 2 | target_top1_gap | +0.3976 | 0.3976 | NEW |
| 3 | joint_entropy_gate | +0.3664 | 0.3664 | NEW |
| 4 | target_entropy | -0.3609 | 0.3609 | OLD |
| 5 | target_varentropy | -0.2871 | 0.2871 | NEW |
| 6 | top1_minus_top2 | +0.2152 | 0.2152 | OLD |
| 7 | top1_prob | +0.2052 | 0.2052 | OLD |
| 8 | draft_entropy | -0.1902 | 0.1902 | OLD |
| 9 | hidden_cosine_sim | -0.1807 | 0.1807 | NEW |
| 10 | entropy_gap_neg | -0.0914 | 0.0914 | NEW |
| 11 | hidden_norm | -0.0718 | 0.0718 | OLD |
| 12 | hidden_var | -0.0596 | 0.0596 | NEW |
| 13 | entropy_gap_pos | -0.0468 | 0.0468 | NEW |
| 14 | hidden_max | +0.0122 | 0.0122 | NEW |

### 2.3 14-sig DeepSeek-8B+LlamaDraft — Correlation with confidence (ranked by |r|)

| Rank | Signal | r(signal, confidence) | |r| | Signal type |
|------|--------|----------------------|-----|-------------|
| 1 | joint_entropy_gate | +0.6752 | 0.6752 | NEW |
| 2 | target_entropy | -0.6231 | 0.6231 | OLD |
| 3 | draft_entropy | -0.6110 | 0.6110 | OLD |
| 4 | target_top1_gap | +0.5612 | 0.5612 | NEW |
| 5 | target_varentropy | -0.5582 | 0.5582 | NEW |
| 6 | top1_prob | +0.5501 | 0.5501 | OLD |
| 7 | top1_minus_top2 | +0.5437 | 0.5437 | OLD |
| 8 | entropy_gap_neg | -0.4625 | 0.4625 | NEW |
| 9 | rolling_accept_rate | +0.4075 | 0.4075 | OLD |
| 10 | hidden_max | +0.3454 | 0.3454 | NEW |
| 11 | hidden_norm | +0.2936 | 0.2936 | OLD |
| 12 | hidden_var | +0.2928 | 0.2928 | NEW |
| 13 | hidden_cosine_sim | -0.0184 | 0.0184 | NEW |
| 14 | entropy_gap_pos | -0.0179 | 0.0179 | NEW |

### 2.4 14-sig DeepSeek-8B+LlamaDraft — Config Distribution

**chosen_topk**:
| Value | Count | Pct |
|-------|-------|-----|
| 1 | 2644 | 69.5% |
| 2 | 978 | 25.7% |
| 3 | 178 | 4.7% |
| 4 | 3 | 0.1% |

**chosen_num_steps**:
| Value | Count | Pct |
|-------|-------|-----|
| 2 | 460 | 12.1% |
| 3 | 2537 | 66.7% |
| 4 | 793 | 20.9% |
| 5 | 13 | 0.3% |

**chosen_num_draft_tokens**:
| Value | Count | Pct |
|-------|-------|-----|
| 3 | 460 | 12.1% |
| 4 | 2184 | 57.4% |
| 5 | 747 | 19.6% |
| 6 | 356 | 9.4% |
| 7 | 55 | 1.4% |
| 8 | 1 | 0.0% |

**Config tuple (topk, steps, ndt) distribution**:
| (topk, steps, ndt) | Count | Pct |
|--------------------|-------|-----|
| (1, 3, 4) | 2184 | 57.4% |
| (1, 2, 3) | 460 | 12.1% |
| (2, 4, 5) | 394 | 10.4% |
| (2, 3, 5) | 353 | 9.3% |
| (2, 4, 6) | 231 | 6.1% |
| (3, 4, 6) | 125 | 3.3% |
| (3, 4, 7) | 43 | 1.1% |
| (3, 5, 7) | 10 | 0.3% |
| (4, 5, 7) | 2 | 0.1% |
| (4, 5, 8) | 1 | 0.0% |

### 2.5 14-sig DeepSeek-8B+LlamaDraft — Confidence Distribution

| Bucket | Count | Pct |
|--------|-------|-----|
| [0.0, 0.1) | 0 | 0.0% |
| [0.1, 0.2) | 3 | 0.1% |
| [0.2, 0.3) | 119 | 3.1% |
| [0.3, 0.4) | 507 | 13.3% |
| [0.4, 0.5) | 1059 | 27.8% |
| [0.5, 0.6) | 1096 | 28.8% |
| [0.6, 0.7) | 668 | 17.6% |
| [0.7, 0.8) | 276 | 7.3% |
| [0.8, 0.9) | 70 | 1.8% |
| [0.9, 1.0) | 5 | 0.1% |

## Section 3: New Signals Assessment

This section evaluates whether the 7 new signals (hidden_cosine_sim, hidden_var, hidden_max, target_top1_gap, target_varentropy, joint_entropy_gate, entropy_gap_pos, entropy_gap_neg) improve prediction of acceptance rate compared to the original 7 signals.

### 3.1 Llama-8B — Signal Ranking by |r(signal, RAR)| (excluding RAR itself)

| Rank | Signal | r | |r| | Old/New | Top-3? |
|------|--------|---|-----|---------|--------|
| 1 | target_top1_gap | +0.4354 | 0.4354 | NEW | YES |
| 2 | target_entropy | -0.3798 | 0.3798 | OLD | YES |
| 3 | joint_entropy_gate | +0.3088 | 0.3088 | NEW | YES |
| 4 | target_varentropy | -0.2854 | 0.2854 | NEW |  |
| 5 | draft_entropy | -0.1936 | 0.1936 | OLD |  |
| 6 | top1_prob | +0.1839 | 0.1839 | OLD |  |
| 7 | top1_minus_top2 | +0.1785 | 0.1785 | OLD |  |
| 8 | hidden_cosine_sim | -0.1475 | 0.1475 | NEW |  |
| 9 | entropy_gap_pos | -0.1169 | 0.1169 | NEW |  |
| 10 | entropy_gap_neg | -0.1062 | 0.1062 | NEW |  |
| 11 | hidden_norm | -0.1027 | 0.1027 | OLD |  |
| 12 | hidden_var | -0.0937 | 0.0937 | NEW |  |
| 13 | hidden_max | +0.0451 | 0.0451 | NEW |  |

**New signals in top half**: 3/8 new signals rank in top 6
**Old signals in top half**: 3/6 old signals rank in top 6

**Average |r| for old signals**: 0.2077
**Average |r| for new signals**: 0.1924

### 3.1 DeepSeek-8B+LlamaDraft — Signal Ranking by |r(signal, RAR)| (excluding RAR itself)

| Rank | Signal | r | |r| | Old/New | Top-3? |
|------|--------|---|-----|---------|--------|
| 1 | target_top1_gap | +0.3976 | 0.3976 | NEW | YES |
| 2 | joint_entropy_gate | +0.3664 | 0.3664 | NEW | YES |
| 3 | target_entropy | -0.3609 | 0.3609 | OLD | YES |
| 4 | target_varentropy | -0.2871 | 0.2871 | NEW |  |
| 5 | top1_minus_top2 | +0.2152 | 0.2152 | OLD |  |
| 6 | top1_prob | +0.2052 | 0.2052 | OLD |  |
| 7 | draft_entropy | -0.1902 | 0.1902 | OLD |  |
| 8 | hidden_cosine_sim | -0.1807 | 0.1807 | NEW |  |
| 9 | entropy_gap_neg | -0.0914 | 0.0914 | NEW |  |
| 10 | hidden_norm | -0.0718 | 0.0718 | OLD |  |
| 11 | hidden_var | -0.0596 | 0.0596 | NEW |  |
| 12 | entropy_gap_pos | -0.0468 | 0.0468 | NEW |  |
| 13 | hidden_max | +0.0122 | 0.0122 | NEW |  |

**New signals in top half**: 3/8 new signals rank in top 6
**Old signals in top half**: 3/6 old signals rank in top 6

**Average |r| for old signals**: 0.2087
**Average |r| for new signals**: 0.1802

### 3.2 Comparison with Previous 7-Signal Analysis

From the previous analysis (analysis_4configs.md), the 7-signal correlations with RAR were:

| Signal | 7sig Llama r(RAR) | 7sig DeepSeek r(RAR) | 14sig Llama r(RAR) | 14sig DeepSeek r(RAR) |
|--------|-------------------|----------------------|--------------------|-----------------------|
| draft_entropy | -0.1629 | -0.1696 | -0.1936 | -0.1902 |
| top1_prob | +0.1669 | +0.2503 | +0.1839 | +0.2052 |
| top1_minus_top2 | +0.1563 | +0.2552 | +0.1785 | +0.2152 |
| hidden_norm | -0.0458 | -0.0689 | -0.1027 | -0.0718 |
| target_entropy | -0.3724 | -0.3827 | -0.3798 | -0.3609 |
| entropy_gap (old) | +0.0496 | +0.0100 | split below | split below |
| entropy_gap_pos (new) | — | — | -0.1169 | -0.0468 |
| entropy_gap_neg (new) | — | — | -0.1062 | -0.0914 |

### 3.3 Key Findings on New Signals

**Llama-8B**: New signals ranked by |r(signal, RAR)|:

- target_top1_gap: r=+0.4354 (overall rank 1/13)
- joint_entropy_gate: r=+0.3088 (overall rank 3/13)
- target_varentropy: r=-0.2854 (overall rank 4/13)
- hidden_cosine_sim: r=-0.1475 (overall rank 8/13)
- entropy_gap_pos: r=-0.1169 (overall rank 9/13)
- entropy_gap_neg: r=-0.1062 (overall rank 10/13)
- hidden_var: r=-0.0937 (overall rank 12/13)
- hidden_max: r=+0.0451 (overall rank 13/13)

**DeepSeek-8B**: New signals ranked by |r(signal, RAR)|:

- target_top1_gap: r=+0.3976 (overall rank 1/13)
- joint_entropy_gate: r=+0.3664 (overall rank 2/13)
- target_varentropy: r=-0.2871 (overall rank 4/13)
- hidden_cosine_sim: r=-0.1807 (overall rank 8/13)
- entropy_gap_neg: r=-0.0914 (overall rank 9/13)
- hidden_var: r=-0.0596 (overall rank 11/13)
- entropy_gap_pos: r=-0.0468 (overall rank 12/13)
- hidden_max: r=+0.0122 (overall rank 13/13)

## Section 4: Recommended Signal Weights Based on Correlation Strength

The current policy uses equal weights (1/N for N signals). Below we propose weights proportional to |r(signal, RAR)|, normalized to sum to 1.

### 4.1 Llama-8B — Proposed Weights

| Signal | |r(RAR)| | Equal weight (1/13) | Proposed weight | Ratio vs equal |
|--------|---------|---------------------|-----------------|----------------|
| target_top1_gap | 0.4354 | 0.0769 | 0.1689 | 2.20x |
| target_entropy | 0.3798 | 0.0769 | 0.1474 | 1.92x |
| joint_entropy_gate | 0.3088 | 0.0769 | 0.1198 | 1.56x |
| target_varentropy | 0.2854 | 0.0769 | 0.1107 | 1.44x |
| draft_entropy | 0.1936 | 0.0769 | 0.0751 | 0.98x |
| top1_prob | 0.1839 | 0.0769 | 0.0713 | 0.93x |
| top1_minus_top2 | 0.1785 | 0.0769 | 0.0692 | 0.90x |
| hidden_cosine_sim | 0.1475 | 0.0769 | 0.0572 | 0.74x |
| entropy_gap_pos | 0.1169 | 0.0769 | 0.0454 | 0.59x |
| entropy_gap_neg | 0.1062 | 0.0769 | 0.0412 | 0.54x |
| hidden_norm | 0.1027 | 0.0769 | 0.0398 | 0.52x |
| hidden_var | 0.0937 | 0.0769 | 0.0364 | 0.47x |
| hidden_max | 0.0451 | 0.0769 | 0.0175 | 0.23x |

**Tier breakdown for Llama-8B**:
- HIGH (|r| >= 0.2): target_entropy, target_top1_gap, target_varentropy, joint_entropy_gate
- MEDIUM (0.1 <= |r| < 0.2): draft_entropy, top1_prob, top1_minus_top2, hidden_norm, hidden_cosine_sim, entropy_gap_pos, entropy_gap_neg
- LOW (|r| < 0.1): hidden_var, hidden_max

### 4.1 DeepSeek-8B+LlamaDraft — Proposed Weights

| Signal | |r(RAR)| | Equal weight (1/13) | Proposed weight | Ratio vs equal |
|--------|---------|---------------------|-----------------|----------------|
| target_top1_gap | 0.3976 | 0.0769 | 0.1600 | 2.08x |
| joint_entropy_gate | 0.3664 | 0.0769 | 0.1474 | 1.92x |
| target_entropy | 0.3609 | 0.0769 | 0.1452 | 1.89x |
| target_varentropy | 0.2871 | 0.0769 | 0.1155 | 1.50x |
| top1_minus_top2 | 0.2152 | 0.0769 | 0.0866 | 1.13x |
| top1_prob | 0.2052 | 0.0769 | 0.0825 | 1.07x |
| draft_entropy | 0.1902 | 0.0769 | 0.0765 | 0.99x |
| hidden_cosine_sim | 0.1807 | 0.0769 | 0.0727 | 0.95x |
| entropy_gap_neg | 0.0914 | 0.0769 | 0.0368 | 0.48x |
| hidden_norm | 0.0718 | 0.0769 | 0.0289 | 0.38x |
| hidden_var | 0.0596 | 0.0769 | 0.0240 | 0.31x |
| entropy_gap_pos | 0.0468 | 0.0769 | 0.0188 | 0.25x |
| hidden_max | 0.0122 | 0.0769 | 0.0049 | 0.06x |

**Tier breakdown for DeepSeek-8B+LlamaDraft**:
- HIGH (|r| >= 0.2): top1_prob, top1_minus_top2, target_entropy, target_top1_gap, target_varentropy, joint_entropy_gate
- MEDIUM (0.1 <= |r| < 0.2): draft_entropy, hidden_cosine_sim
- LOW (|r| < 0.1): hidden_norm, hidden_var, hidden_max, entropy_gap_pos, entropy_gap_neg

### 4.2 Cross-Model Consensus Weights

For a model-agnostic policy, we average |r| across both datasets:

| Signal | |r| Llama | |r| DeepSeek | Avg |r| | Consensus weight | Old/New |
|--------|-----------|--------------|---------|------------------|---------|
| target_top1_gap | 0.4354 | 0.3976 | 0.4165 | 0.1645 | NEW |
| target_entropy | 0.3798 | 0.3609 | 0.3704 | 0.1463 | OLD |
| joint_entropy_gate | 0.3088 | 0.3664 | 0.3376 | 0.1334 | NEW |
| target_varentropy | 0.2854 | 0.2871 | 0.2863 | 0.1131 | NEW |
| top1_minus_top2 | 0.1785 | 0.2152 | 0.1968 | 0.0778 | OLD |
| top1_prob | 0.1839 | 0.2052 | 0.1945 | 0.0768 | OLD |
| draft_entropy | 0.1936 | 0.1902 | 0.1919 | 0.0758 | OLD |
| hidden_cosine_sim | 0.1475 | 0.1807 | 0.1641 | 0.0648 | NEW |
| entropy_gap_neg | 0.1062 | 0.0914 | 0.0988 | 0.0390 | NEW |
| hidden_norm | 0.1027 | 0.0718 | 0.0873 | 0.0345 | OLD |
| entropy_gap_pos | 0.1169 | 0.0468 | 0.0819 | 0.0324 | NEW |
| hidden_var | 0.0937 | 0.0596 | 0.0767 | 0.0303 | NEW |
| hidden_max | 0.0451 | 0.0122 | 0.0286 | 0.0113 | NEW |

## Section 5: Signal Dilution Analysis

This section investigates why the 14-signal policy may achieve slightly different accept_len than the 7-signal policy.

### 5.1 Config Distribution Comparison (Llama-8B)

| Config Param | 7-sig Mean (from prev analysis) | 14-sig Mean |
|-------------|--------------------------------|-------------|
| chosen_topk | 2.0587 | 1.6416 |
| chosen_num_steps | 3.6368 | 3.3749 |
| chosen_num_draft_tokens | 5.3189 | 4.7626 |

### 5.2 Config Distribution Comparison (DeepSeek-8B)

| Config Param | 7-sig Mean (from prev analysis) | 14-sig Mean |
|-------------|--------------------------------|-------------|
| chosen_topk | 1.1514 | 1.3531 |
| chosen_num_steps | 2.7800 | 3.0944 |
| chosen_num_draft_tokens | 3.8738 | 4.3071 |

### 5.3 Confidence Distribution Comparison

| Metric | 7-sig Llama | 14-sig Llama | 7-sig DeepSeek | 14-sig DeepSeek |
|--------|-------------|-------------|----------------|-----------------|
| Confidence mean | 0.6671 | 0.5905 | 0.4452 | 0.5236 |
| Confidence std | 0.1239 | 0.1282 | 0.1218 | 0.1271 |
| Confidence median | 0.6889 | 0.5995 | 0.4317 | 0.5173 |

### 5.4 Signal Dilution Mechanism

When expanding from 7 to 14 signals with equal weights, each signal's influence on the
confidence score is reduced from 1/7 (0.143) to 1/14 (0.071) — a 50% reduction per signal.
This has several consequences:

1. **Reduced sensitivity to strong predictors**: target_entropy (the strongest predictor of RAR
   in the 7-signal analysis) now has half the weight. If new signals are weaker predictors,
   the overall confidence becomes a noisier estimate of true acceptance probability.

2. **Regression to the mean**: Adding weak signals (those with |r| < 0.1) pulls confidence
   toward the center of its range, reducing the system's ability to discriminate between
   easy (high-acceptance) and hard (low-acceptance) regions.

**Llama-8B**:
- Old signals (5 excl RAR, entropy_gap): total |r| = 1.0384 (40.3% of total)
- New signals (8): total |r| = 1.5389 (59.7% of total)
- But equal weighting gives new signals 8/13 = 62% of total weight
- Information-weight mismatch: new signals get 62% of weight but contribute only 59.7% of predictive power

**DeepSeek-8B**:
- Old signals (5 excl RAR, entropy_gap): total |r| = 1.0433 (42.0% of total)
- New signals (8): total |r| = 1.4419 (58.0% of total)
- But equal weighting gives new signals 8/13 = 62% of total weight
- Information-weight mismatch: new signals get 62% of weight but contribute only 58.0% of predictive power

### 5.5 Recommendations to Fix Signal Dilution

1. **Use correlation-weighted signals**: Replace equal 1/N weights with weights proportional
   to |r(signal, RAR)| as shown in Section 4. This preserves the influence of strong
   predictors while allowing new signals to contribute where informative.

2. **Prune uninformative signals**: Remove signals with |r| < 0.05 across both datasets.
   Candidates for removal:
   - hidden_max (|r|=0.0451 Llama, |r|=0.0122 DeepSeek)

3. **Use a learned weighting**: Train a small linear model (or logistic regression) to predict
   RAR from the 14 signals. The learned coefficients directly give optimal weights. This can
   be done offline on collected signal data without modifying the serving infrastructure.

4. **Consider feature selection**: Use forward stepwise selection starting from the strongest
   signal (target_entropy) and adding signals only if they improve cross-validated R-squared.
   This would likely yield a 4-6 signal subset that outperforms both the 7 and 14 signal sets.

## Section 6: Inter-Signal Correlation Highlights

Highly correlated signals are redundant. Below are signal pairs with |r| > 0.7:

**Llama-8B** (|r| > 0.7):

| Signal A | Signal B | r |
|----------|----------|---|
| draft_entropy | joint_entropy_gate | -0.7909 |
| draft_entropy | entropy_gap_neg | +0.9556 |
| top1_prob | top1_minus_top2 | +0.9741 |
| hidden_norm | hidden_var | +0.9925 |
| target_entropy | target_top1_gap | -0.8805 |
| target_entropy | target_varentropy | +0.8850 |

**DeepSeek-8B+LlamaDraft** (|r| > 0.7):

| Signal A | Signal B | r |
|----------|----------|---|
| draft_entropy | joint_entropy_gate | -0.7483 |
| draft_entropy | entropy_gap_neg | +0.9533 |
| top1_prob | top1_minus_top2 | +0.9699 |
| hidden_norm | hidden_var | +0.9936 |
| target_entropy | target_top1_gap | -0.8080 |
| target_entropy | target_varentropy | +0.8784 |

## Summary

### Key Findings

- **Llama-8B** top-3 predictors of RAR: target_top1_gap, target_entropy, joint_entropy_gate
  - NEW signal(s) in top 3: target_top1_gap, joint_entropy_gate
- **DeepSeek-8B** top-3 predictors of RAR: target_top1_gap, joint_entropy_gate, target_entropy
  - NEW signal(s) in top 3: target_top1_gap, joint_entropy_gate

### Accept Length Summary

| Model Pair | Vanilla | 7-sig Dynamic | 14-sig Dynamic |
|------------|---------|---------------|----------------|
| Llama-8B (ss) | 2.9019 | 3.3214 | 3.1669 |
| DeepSeek-8B (ss) | 1.8821 | 2.0851 | 1.9259 |

### Verdict on 14-Signal Expansion

The expansion from 7 to 14 signals introduces several signals with non-trivial correlation
to acceptance rate, but the equal-weight averaging dilutes the influence of the strongest
predictors. The recommended path forward is:

1. Adopt correlation-weighted signals (Section 4)
2. Prune signals with consistently low |r| across both model pairs
3. Consider a learned weighting for production deployment
4. The split of entropy_gap into pos/neg components should be evaluated — if one component
   is consistently stronger, keep only that one

