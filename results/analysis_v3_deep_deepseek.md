# Dynamic Speculative Decoding V3 — Deep Analysis

*Generated from 14709 dynamic steps and 13825 static steps.*

*All AccLen values are SpecForge-compatible (accepted_draft + 1 bonus token).*

*Throughput = total_output_tokens / elapsed_seconds (same as SpecForge).*


## DeepSeek-R1 8B

### Dynamic vs Static Comparison — DeepSeek-R1 8B

| Metric | Static 7,4,8 | Dynamic V3 |
|--------|-------------|------------|
| Total steps | 13825 | 14709 |
| Total tokens | 30309 | 31754 |
| AccLen | 2.19 | 2.16 |
| Efficiency | 0.274 | 0.454 |
| Total rejections (AccLen=1) | 5144 (37.2%) | 5559 (37.8%) |
| Throughput (tok/s) | 105.2 | 119.4 |

**AccLen distribution:**

| AccLen | Static | Dynamic |
|--------|--------|---------|
| 1 | 5144 (37.2%) | 5559 (37.8%) |
| 2 | 4539 (32.8%) | 4874 (33.1%) |
| 3 | 2302 (16.7%) | 2280 (15.5%) |
| 4 | 930 (6.7%) | 1086 (7.4%) |
| 5 | 440 (3.2%) | 536 (3.6%) |
| 6 | 201 (1.5%) | 167 (1.1%) |
| 7 | 97 (0.7%) | 75 (0.5%) |
| 8 | 172 (1.2%) | 132 (0.9%) |


### A. Signal-Acceptance Correlations — DeepSeek-R1 8B

**Same-step: signal(i) → accept_length(i)**

| Signal | r | Interpretation |
|--------|---|----------------|
| draft_oracle_gate | +0.536 | strong |
| joint_confidence_product_fixed | +0.493 | strong |
| top1_prob | +0.442 | strong |
| target_oracle_gate_fixed | +0.414 | strong |
| rolling_accept_length | +0.391 | moderate |
| rolling_accept_rate | +0.380 | moderate |
| confidence_agreement | +0.324 | moderate |
| target_top1_prob | +0.266 | moderate |

**Next-step: signal(i) → accept_length(i+1)**

| Signal | r |
|--------|---|
| top1_prob | +0.167 |
| target_top1_prob | +0.211 |
| rolling_accept_rate | +0.357 |
| rolling_accept_length | +0.365 |
| draft_oracle_gate | +0.374 |
| target_oracle_gate_fixed | +0.380 |
| joint_confidence_product_fixed | +0.220 |
| confidence_agreement | +0.113 |


### B. Config Efficiency — DeepSeek-R1 8B (Dynamic)

**Per-config breakdown:**

| Config (topk,steps,ndt) | Count | % | AccLen | Eff | Zero% |
|-------------------------|-------|---|--------|-----|-------|
| (4, 3, 8) | 6959 | 47.3% | 1.72 | 0.430 | 50% |
| (3, 4, 8) | 4127 | 28.1% | 2.39 | 0.479 | 26% |
| (4, 4, 8) | 1710 | 11.6% | 2.09 | 0.417 | 33% |
| (3, 5, 8) | 797 | 5.4% | 2.77 | 0.462 | 21% |
| (2, 5, 8) | 336 | 2.3% | 3.18 | 0.530 | 18% |
| (1, 7, 8) | 283 | 1.9% | 5.80 | 0.725 | 5% |
| (2, 6, 8) | 235 | 1.6% | 3.94 | 0.562 | 10% |
| (1, 3, 4) | 230 | 1.6% | 1.51 | 0.378 | 62% |
| (1, 6, 7) | 32 | 0.2% | 4.66 | 0.665 | 6% |

**Overall**: AccLen=2.16, Eff=0.454

**By DOG bucket:**

| DOG Range | Count | AccLen | Mean topk | Mean steps | Eff | Zero% |
|-----------|-------|--------|-----------|------------|-----|-------|
| [0.0, 0.1) | 9962 | 1.83 | 3.8 | 3.3 | 0.429 | 45% |
| [0.1, 0.2) | 3280 | 2.47 | 3.0 | 4.1 | 0.487 | 25% |
| [0.2, 0.3) | 871 | 2.90 | 2.7 | 5.0 | 0.484 | 20% |
| [0.3, 0.4) | 255 | 3.71 | 2.0 | 5.8 | 0.544 | 12% |
| [0.4, 0.5) | 107 | 4.67 | 1.2 | 6.5 | 0.630 | 7% |
| [0.5, 0.7) | 122 | 5.75 | 1.0 | 7.0 | 0.718 | 5% |
| [0.7, 1.0) | 112 | 6.40 | 1.0 | 7.0 | 0.800 | 3% |


### B. Config Efficiency — DeepSeek-R1 8B (Static)

**Per-config breakdown:**

| Config (topk,steps,ndt) | Count | % | AccLen | Eff | Zero% |
|-------------------------|-------|---|--------|-----|-------|
| (4, 7, 8) | 13825 | 100.0% | 2.19 | 0.274 | 37% |

**Overall**: AccLen=2.19, Eff=0.274

**By DOG bucket:**

| DOG Range | Count | AccLen | Mean topk | Mean steps | Eff | Zero% |
|-----------|-------|--------|-----------|------------|-----|-------|
| [0.0, 0.1) | 9318 | 1.87 | 4.0 | 7.0 | 0.233 | 45% |
| [0.1, 0.2) | 3045 | 2.51 | 4.0 | 7.0 | 0.314 | 24% |
| [0.2, 0.3) | 894 | 3.01 | 4.0 | 7.0 | 0.376 | 16% |
| [0.3, 0.4) | 279 | 3.87 | 4.0 | 7.0 | 0.484 | 9% |
| [0.4, 0.5) | 125 | 4.17 | 4.0 | 7.0 | 0.521 | 10% |
| [0.5, 0.7) | 92 | 5.40 | 4.0 | 7.0 | 0.675 | 5% |
| [0.7, 1.0) | 72 | 6.72 | 4.0 | 7.0 | 0.840 | 3% |


### C. Token-Level Analysis — DeepSeek-R1 8B

- Total steps with token data: 14709
- Total rejections (acc=0): 0 (0.0%)
- First-token rejections: 0
- Tree steps (topk>1): 14164
- Non-top-1 branch accepted: 8763/14164 (61.9%)


### D. Failure Modes — DeepSeek-R1 8B

**Total rejections**: 5559/14709 (37.8%)

**Config at total rejection:**

| Config | Count | % of rejections |
|--------|-------|-----------------|
| (4, 3, 8) | 3495 | 63% |
| (3, 4, 8) | 1092 | 20% |
| (4, 4, 8) | 563 | 10% |
| (3, 5, 8) | 166 | 3% |
| (1, 3, 4) | 142 | 3% |
| (2, 5, 8) | 61 | 1% |
| (2, 6, 8) | 24 | 0% |
| (1, 7, 8) | 14 | 0% |
| (1, 6, 7) | 2 | 0% |

**Signal values: rejection vs success:**

| Signal | Mean@AccLen=1 | Mean@AccLen≥4 | Δ |
|--------|--------------|--------------|---|
| top1_prob | 0.371 | 0.761 | +0.391 |
| target_top1_prob | 0.730 | 0.815 | +0.085 |
| rolling_accept_rate | 0.153 | 0.251 | +0.098 |
| rolling_accept_length | 1.042 | 1.739 | +0.696 |
| draft_oracle_gate | 0.060 | 0.204 | +0.143 |
| target_oracle_gate_fixed | 0.115 | 0.218 | +0.103 |
| joint_confidence_product_fixed | 0.274 | 0.628 | +0.354 |
| confidence_agreement | 0.605 | 0.806 | +0.200 |

**Expensive failures** (acc=0, steps≥5): 267/5559 (5% of rejections)

**Consecutive failure streaks:**
- Total streaks: 3271
- Mean length: 1.7
- Max length: 16
- Distribution: {1: 1953, 2: 787, 3: 329, 4: 111, 5: 44, 6: 14, 7: 9, 8: 6, 9: 8, 10: 4, 11: 3, 12: 1, 13: 1, 16: 1}
- Streaks ≥5 (circuit breaker): 91


### E. Temporal Patterns — DeepSeek-R1 8B

**Accept_length autocorrelation:**

| Lag | r |
|-----|---|
| 1 | +0.276 |
| 2 | +0.246 |
| 3 | +0.223 |
| 5 | +0.225 |

**DOG stability:**
- Mean |ΔDOG| between steps: 0.053
- DOG std: 0.111
- DOG range: [0.000, 0.975]

**Per-turn summary** (40 turns):

| Turn | Steps | AccLen | Mean DOG | Zero% |
|------|-------|--------|----------|-------|
| 0 | 535 | 1.97 | 0.072 | 39% |
| 1 | 251 | 4.28 | 0.386 | 18% |
| 2 | 331 | 2.11 | 0.087 | 40% |
| 3 | 236 | 2.40 | 0.118 | 31% |
| 4 | 367 | 2.11 | 0.079 | 41% |
| 5 | 171 | 2.04 | 0.085 | 39% |
| 6 | 307 | 1.93 | 0.071 | 42% |
| 7 | 269 | 3.23 | 0.243 | 31% |
| 8 | 257 | 2.02 | 0.076 | 37% |
| 9 | 328 | 2.93 | 0.176 | 25% |
| 10 | 398 | 2.10 | 0.087 | 36% |
| 11 | 308 | 2.04 | 0.080 | 36% |
| 12 | 470 | 2.05 | 0.080 | 35% |
| 13 | 356 | 2.77 | 0.172 | 27% |
| 14 | 313 | 1.94 | 0.069 | 41% |
| 15 | 204 | 2.08 | 0.084 | 35% |
| 16 | 469 | 2.17 | 0.092 | 41% |
| 17 | 123 | 2.33 | 0.103 | 33% |
| 18 | 182 | 2.96 | 0.201 | 29% |
| 19 | 235 | 2.59 | 0.142 | 35% |
| 20 | 748 | 2.01 | 0.076 | 41% |
| 21 | 237 | 2.08 | 0.089 | 35% |
| 22 | 375 | 2.00 | 0.077 | 42% |
| 23 | 201 | 1.98 | 0.072 | 39% |
| 24 | 428 | 1.94 | 0.066 | 44% |
| 25 | 383 | 2.29 | 0.106 | 33% |
| 26 | 659 | 2.05 | 0.087 | 41% |
| 27 | 427 | 2.31 | 0.112 | 29% |
| 28 | 175 | 1.78 | 0.053 | 54% |
| 29 | 187 | 2.11 | 0.090 | 34% |
| 30 | 265 | 2.05 | 0.082 | 40% |
| 31 | 281 | 2.10 | 0.084 | 34% |
| 32 | 1007 | 2.03 | 0.083 | 38% |
| 33 | 1089 | 1.57 | 0.023 | 52% |
| 34 | 153 | 1.97 | 0.065 | 38% |
| 35 | 199 | 1.89 | 0.059 | 45% |
| 36 | 391 | 2.21 | 0.100 | 37% |
| 37 | 734 | 2.24 | 0.105 | 32% |
| 38 | 339 | 1.98 | 0.076 | 40% |
| 39 | 321 | 2.38 | 0.123 | 33% |


### F. Opportunity Analysis — DeepSeek-R1 8B

**Baseline**: 31754 tokens in 14709 steps, AccLen=2.16

**Oracle (skip all AccLen=1 steps)**: 26195 tokens in 9150 steps, AccLen=2.86 (+33%)

**Counterfactual chain-only (topk=1, vary steps)**:
- Efficiency: 0.485 (current: 0.454, Δ=+0.032)
- Note: same AccLen assumed — actual may differ with topk=1

**Efficiency by topk group:**
- topk=1: eff=0.575, AccLen=3.92, n=545 (4%)
- topk=2: eff=0.543, AccLen=3.49, n=571 (4%)
- topk≥3: eff=0.445, AccLen=2.03, n=13593 (92%)


### G. Topk vs Chain — DeepSeek-R1 8B

**Per-topk breakdown:**

| topk | Count | % | AccLen | Mean Steps | Eff | Zero% | Mean DOG |
|------|-------|---|--------|------------|-----|-------|----------|
| 1 | 545 | 4% | 3.92 | 5.3 | 0.575 | 29% | 0.372 |
| 2 | 571 | 4% | 3.49 | 5.4 | 0.543 | 15% | 0.310 |
| 3 | 4924 | 33% | 2.46 | 4.2 | 0.476 | 26% | 0.139 |
| 4 | 8669 | 59% | 1.79 | 3.2 | 0.427 | 47% | 0.039 |

**Topk=1 vs topk>1 at similar DOG levels:**

| DOG Range | topk=1 AccLen (n) | topk>1 AccLen (n) | Δ |
|-----------|-------------------|-------------------|---|
| [0.0, 0.2) | 1.51 (230) | 2.00 (13012) | +0.49 |
| [0.2, 0.3) | — | 2.97 (1011) | — |
| [0.3, 0.5) | 4.59 (81) | 4.26 (141) | -0.33 |
| [0.5, 1.0) | 6.06 (234) | — | — |


