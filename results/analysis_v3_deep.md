# Dynamic Speculative Decoding V3 — Deep Analysis

*Generated from 7617 dynamic steps and 1524 static steps.*


## Llama 3.1 8B

### Dynamic vs Static Comparison — Llama 3.1 8B

| Metric | Static 7,4,8 | Dynamic V3 |
|--------|-------------|------------|
| Total steps | 1524 | 1507 |
| Total tokens | 4470 | 4447 |
| Mean accept_length | 2.93 | 2.95 |
| Efficiency (acc+1)/(steps+1) | 0.492 | 0.607 |
| Total rejections | 164 (10.8%) | 166 (11.0%) |

**Accept length distribution:**

| Accept | Static | Dynamic |
|--------|--------|---------|
| 0 | 164 (10.8%) | 166 (11.0%) |
| 1 | 295 (19.4%) | 304 (20.2%) |
| 2 | 293 (19.2%) | 244 (16.2%) |
| 3 | 231 (15.2%) | 222 (14.7%) |
| 4 | 197 (12.9%) | 195 (12.9%) |
| 5 | 108 (7.1%) | 157 (10.4%) |
| 6 | 84 (5.5%) | 109 (7.2%) |
| 7 | 152 (10.0%) | 110 (7.3%) |


### A. Signal-Acceptance Correlations — Llama 3.1 8B

**Same-step: signal(i) → accept_length(i)**

| Signal | r | Interpretation |
|--------|---|----------------|
| joint_confidence_product_fixed | +0.473 | strong |
| top1_prob | +0.436 | strong |
| draft_oracle_gate | +0.415 | strong |
| confidence_agreement | +0.367 | moderate |
| target_top1_prob | +0.268 | moderate |
| target_oracle_gate_fixed | +0.248 | moderate |
| rolling_accept_rate | +0.198 | weak |
| rolling_accept_length | +0.192 | weak |

**Next-step: signal(i) → accept_length(i+1)**

| Signal | r |
|--------|---|
| top1_prob | +0.091 |
| target_top1_prob | +0.163 |
| rolling_accept_rate | +0.136 |
| rolling_accept_length | +0.124 |
| draft_oracle_gate | +0.150 |
| target_oracle_gate_fixed | +0.158 |
| joint_confidence_product_fixed | +0.135 |
| confidence_agreement | +0.084 |


### B. Config Efficiency — Llama 3.1 8B (Dynamic)

**Per-config breakdown:**

| Config (topk,steps,ndt) | Count | % | Mean Acc | Eff | Zero% |
|-------------------------|-------|---|----------|-----|-------|
| (1, 7, 8) | 378 | 25.1% | 3.94 | 0.618 | 3% |
| (2, 6, 8) | 322 | 21.4% | 3.41 | 0.630 | 4% |
| (3, 4, 8) | 278 | 18.4% | 1.90 | 0.579 | 22% |
| (3, 5, 8) | 185 | 12.3% | 2.65 | 0.609 | 10% |
| (2, 5, 8) | 185 | 12.3% | 2.78 | 0.631 | 8% |
| (4, 3, 8) | 61 | 4.0% | 0.92 | 0.480 | 44% |
| (1, 6, 7) | 58 | 3.8% | 3.57 | 0.653 | 7% |
| (4, 4, 8) | 40 | 2.7% | 1.52 | 0.505 | 38% |

**Overall dynamic efficiency**: 0.607

**By DOG bucket:**

| DOG Range | Count | Mean Acc | Mean topk | Mean steps | Eff | Zero% |
|-----------|-------|----------|-----------|------------|-----|-------|
| [0.0, 0.1) | 143 | 1.24 | 3.7 | 3.6 | 0.488 | 40% |
| [0.1, 0.2) | 265 | 2.05 | 3.0 | 4.1 | 0.597 | 18% |
| [0.2, 0.3) | 306 | 2.71 | 2.5 | 5.0 | 0.618 | 9% |
| [0.3, 0.4) | 320 | 3.32 | 2.0 | 5.9 | 0.627 | 3% |
| [0.4, 0.5) | 222 | 3.54 | 1.2 | 6.6 | 0.603 | 5% |
| [0.5, 0.7) | 196 | 4.03 | 1.0 | 7.0 | 0.628 | 4% |
| [0.7, 1.0) | 55 | 4.75 | 1.0 | 7.0 | 0.718 | 2% |


### B. Config Efficiency — Llama 3.1 8B (Static)

**Per-config breakdown:**

| Config (topk,steps,ndt) | Count | % | Mean Acc | Eff | Zero% |
|-------------------------|-------|---|----------|-----|-------|
| (4, 7, 8) | 1524 | 100.0% | 2.93 | 0.492 | 11% |

**Overall dynamic efficiency**: 0.492

**By DOG bucket:**

| DOG Range | Count | Mean Acc | Mean topk | Mean steps | Eff | Zero% |
|-----------|-------|----------|-----------|------------|-----|-------|
| [0.0, 0.1) | 154 | 1.16 | 4.0 | 7.0 | 0.269 | 41% |
| [0.1, 0.2) | 304 | 2.02 | 4.0 | 7.0 | 0.378 | 17% |
| [0.2, 0.3) | 296 | 2.94 | 4.0 | 7.0 | 0.493 | 7% |
| [0.3, 0.4) | 312 | 3.13 | 4.0 | 7.0 | 0.516 | 4% |
| [0.4, 0.5) | 209 | 3.48 | 4.0 | 7.0 | 0.560 | 4% |
| [0.5, 0.7) | 199 | 4.47 | 4.0 | 7.0 | 0.684 | 1% |
| [0.7, 1.0) | 50 | 4.26 | 4.0 | 7.0 | 0.657 | 6% |


### C. Token-Level Analysis — Llama 3.1 8B

*No token-level data available. Re-collect with updated test_signal_collection.py.*


### D. Failure Modes — Llama 3.1 8B

**Total rejections**: 166/1507 (11.0%)

**Config at total rejection:**

| Config | Count | % of rejections |
|--------|-------|-----------------|
| (3, 4, 8) | 60 | 36% |
| (4, 3, 8) | 27 | 16% |
| (3, 5, 8) | 18 | 11% |
| (2, 5, 8) | 15 | 9% |
| (4, 4, 8) | 15 | 9% |
| (2, 6, 8) | 14 | 8% |
| (1, 7, 8) | 13 | 8% |
| (1, 6, 7) | 4 | 2% |

**Signal values: rejection vs success:**

| Signal | Mean@acc=0 | Mean@acc≥3 | Δ |
|--------|-----------|-----------|---|
| top1_prob | 0.439 | 0.831 | +0.391 |
| target_top1_prob | 0.789 | 0.830 | +0.041 |
| rolling_accept_rate | 0.425 | 0.440 | +0.015 |
| rolling_accept_length | 2.879 | 2.987 | +0.107 |
| draft_oracle_gate | 0.195 | 0.371 | +0.176 |
| target_oracle_gate_fixed | 0.345 | 0.374 | +0.029 |
| joint_confidence_product_fixed | 0.355 | 0.696 | +0.341 |
| confidence_agreement | 0.628 | 0.834 | +0.206 |

**Expensive failures** (acc=0, steps≥5): 64/166 (39% of rejections)

**Consecutive failure streaks:**
- Total streaks: 147
- Mean length: 1.1
- Max length: 4
- Distribution: {1: 131, 2: 14, 3: 1, 4: 1}
- Streaks ≥5 (circuit breaker): 0


### E. Temporal Patterns — Llama 3.1 8B

**Accept_length autocorrelation:**

| Lag | r |
|-----|---|
| 1 | +0.167 |
| 2 | +0.119 |
| 3 | +0.044 |
| 5 | +0.047 |

**DOG stability:**
- Mean |ΔDOG| between steps: 0.128
- DOG std: 0.181
- DOG range: [0.013, 0.955]

**Per-turn summary** (20 turns):

| Turn | Steps | Mean Acc | Mean DOG | Zero% |
|------|-------|----------|----------|-------|
| 0 | 202 | 3.19 | 0.371 | 9% |
| 1 | 127 | 3.38 | 0.393 | 15% |
| 2 | 39 | 3.03 | 0.338 | 18% |
| 3 | 74 | 2.85 | 0.314 | 11% |
| 4 | 47 | 3.21 | 0.376 | 11% |
| 5 | 16 | 1.94 | 0.230 | 6% |
| 6 | 96 | 2.57 | 0.268 | 8% |
| 7 | 81 | 3.81 | 0.453 | 9% |
| 8 | 60 | 2.65 | 0.279 | 3% |
| 9 | 63 | 3.86 | 0.462 | 8% |
| 10 | 63 | 2.86 | 0.307 | 11% |
| 11 | 73 | 2.73 | 0.278 | 16% |
| 12 | 174 | 2.96 | 0.319 | 8% |
| 13 | 60 | 1.73 | 0.164 | 18% |
| 14 | 61 | 2.57 | 0.263 | 11% |
| 15 | 8 | 2.12 | 0.266 | 12% |
| 16 | 114 | 2.39 | 0.243 | 12% |
| 17 | 21 | 2.52 | 0.254 | 19% |
| 18 | 77 | 3.13 | 0.369 | 10% |
| 19 | 51 | 3.24 | 0.388 | 14% |


### F. Opportunity Analysis — Llama 3.1 8B

**Baseline**: 4447 tokens in 1507 steps = 2.95 tokens/step

**Oracle (skip all acc=0)**: 4447 tokens in 1341 steps = 3.32 tokens/step (+12%)

**Counterfactual chain-only (topk=1, vary steps)**:
- Efficiency: 0.612 (current: 0.607, Δ=+0.005)
- Note: same accept_length assumed — actual may differ with topk=1

**Efficiency by topk group:**
- topk=1: eff=0.623, acc=3.89, n=436 (29%)
- topk=2: eff=0.631, acc=3.18, n=507 (34%)
- topk≥3: eff=0.573, acc=2.01, n=564 (37%)


### G. Topk vs Chain — Llama 3.1 8B

**Per-topk breakdown:**

| topk | Count | % | Mean Acc | Mean Steps | Eff | Zero% | Mean DOG |
|------|-------|---|----------|------------|-----|-------|----------|
| 1 | 436 | 29% | 3.89 | 6.9 | 0.623 | 4% | 0.553 |
| 2 | 507 | 34% | 3.18 | 5.6 | 0.631 | 6% | 0.333 |
| 3 | 463 | 31% | 2.20 | 4.4 | 0.591 | 17% | 0.171 |
| 4 | 101 | 7% | 1.16 | 3.4 | 0.490 | 42% | 0.055 |

**Topk=1 vs topk>1 at similar DOG levels:**

| DOG Range | topk=1 acc (n) | topk>1 acc (n) | Δ |
|-----------|---------------|----------------|---|
| [0.0, 0.2) | — | 1.76 (408) | — |
| [0.2, 0.3) | — | 2.92 (456) | — |
| [0.3, 0.5) | 3.50 (185) | 3.38 (207) | -0.13 |
| [0.5, 1.0) | 4.18 (251) | — | — |



## DeepSeek-R1 8B

### A. Signal-Acceptance Correlations — DeepSeek-R1 8B

**Same-step: signal(i) → accept_length(i)**

| Signal | r | Interpretation |
|--------|---|----------------|
| draft_oracle_gate | +0.611 | strong |
| joint_confidence_product_fixed | +0.534 | strong |
| target_oracle_gate_fixed | +0.514 | strong |
| rolling_accept_length | +0.492 | strong |
| rolling_accept_rate | +0.483 | strong |
| top1_prob | +0.460 | strong |
| target_top1_prob | +0.339 | moderate |
| confidence_agreement | +0.334 | moderate |

**Next-step: signal(i) → accept_length(i+1)**

| Signal | r |
|--------|---|
| top1_prob | +0.205 |
| target_top1_prob | +0.297 |
| rolling_accept_rate | +0.457 |
| rolling_accept_length | +0.463 |
| draft_oracle_gate | +0.471 |
| target_oracle_gate_fixed | +0.481 |
| joint_confidence_product_fixed | +0.287 |
| confidence_agreement | +0.121 |


### B. Config Efficiency — DeepSeek-R1 8B (Dynamic)

**Per-config breakdown:**

| Config (topk,steps,ndt) | Count | % | Mean Acc | Eff | Zero% |
|-------------------------|-------|---|----------|-----|-------|
| (4, 3, 8) | 2626 | 43.0% | 0.77 | 0.442 | 49% |
| (3, 4, 8) | 1764 | 28.9% | 1.39 | 0.479 | 26% |
| (4, 4, 8) | 687 | 11.2% | 1.16 | 0.431 | 31% |
| (3, 5, 8) | 365 | 6.0% | 1.79 | 0.466 | 21% |
| (1, 7, 8) | 260 | 4.3% | 4.96 | 0.745 | 5% |
| (2, 5, 8) | 166 | 2.7% | 2.35 | 0.558 | 19% |
| (2, 6, 8) | 147 | 2.4% | 3.32 | 0.617 | 10% |
| (1, 3, 4) | 72 | 1.2% | 0.67 | 0.417 | 57% |
| (1, 6, 7) | 23 | 0.4% | 4.22 | 0.745 | 4% |

**Overall dynamic efficiency**: 0.474

**By DOG bucket:**

| DOG Range | Count | Mean Acc | Mean topk | Mean steps | Eff | Zero% |
|-----------|-------|----------|-----------|------------|-----|-------|
| [0.0, 0.1) | 3836 | 0.89 | 3.8 | 3.3 | 0.440 | 44% |
| [0.1, 0.2) | 1417 | 1.49 | 3.0 | 4.1 | 0.491 | 24% |
| [0.2, 0.3) | 402 | 1.95 | 2.6 | 5.0 | 0.492 | 22% |
| [0.3, 0.4) | 151 | 3.06 | 2.0 | 5.8 | 0.594 | 11% |
| [0.4, 0.5) | 78 | 4.06 | 1.3 | 6.4 | 0.686 | 6% |
| [0.5, 0.7) | 114 | 4.89 | 1.0 | 7.0 | 0.736 | 5% |
| [0.7, 1.0) | 112 | 5.40 | 1.0 | 7.0 | 0.800 | 3% |


### C. Token-Level Analysis — DeepSeek-R1 8B

*No token-level data available. Re-collect with updated test_signal_collection.py.*


### D. Failure Modes — DeepSeek-R1 8B

**Total rejections**: 2145/6110 (35.1%)

**Config at total rejection:**

| Config | Count | % of rejections |
|--------|-------|-----------------|
| (4, 3, 8) | 1292 | 60% |
| (3, 4, 8) | 465 | 22% |
| (4, 4, 8) | 211 | 10% |
| (3, 5, 8) | 76 | 4% |
| (1, 3, 4) | 41 | 2% |
| (2, 5, 8) | 32 | 1% |
| (2, 6, 8) | 14 | 1% |
| (1, 7, 8) | 13 | 1% |
| (1, 6, 7) | 1 | 0% |

**Signal values: rejection vs success:**

| Signal | Mean@acc=0 | Mean@acc≥3 | Δ |
|--------|-----------|-----------|---|
| top1_prob | 0.395 | 0.779 | +0.385 |
| target_top1_prob | 0.727 | 0.826 | +0.099 |
| rolling_accept_rate | 0.170 | 0.312 | +0.142 |
| rolling_accept_length | 1.155 | 2.157 | +1.002 |
| draft_oracle_gate | 0.068 | 0.262 | +0.194 |
| target_oracle_gate_fixed | 0.127 | 0.277 | +0.150 |
| joint_confidence_product_fixed | 0.290 | 0.655 | +0.365 |
| confidence_agreement | 0.626 | 0.819 | +0.193 |

**Expensive failures** (acc=0, steps≥5): 136/2145 (6% of rejections)

**Consecutive failure streaks:**
- Total streaks: 1313
- Mean length: 1.6
- Max length: 11
- Distribution: {1: 816, 2: 306, 3: 122, 4: 37, 5: 16, 6: 4, 7: 4, 8: 4, 9: 2, 10: 1, 11: 1}
- Streaks ≥5 (circuit breaker): 32


### E. Temporal Patterns — DeepSeek-R1 8B

**Accept_length autocorrelation:**

| Lag | r |
|-----|---|
| 1 | +0.374 |
| 2 | +0.358 |
| 3 | +0.306 |
| 5 | +0.290 |

**DOG stability:**
- Mean |ΔDOG| between steps: 0.061
- DOG std: 0.146
- DOG range: [0.000, 0.975]

**Per-turn summary** (20 turns):

| Turn | Steps | Mean Acc | Mean DOG | Zero% |
|------|-------|----------|----------|-------|
| 0 | 535 | 0.97 | 0.072 | 39% |
| 1 | 251 | 3.28 | 0.386 | 18% |
| 2 | 331 | 1.11 | 0.087 | 40% |
| 3 | 236 | 1.40 | 0.118 | 31% |
| 4 | 367 | 1.11 | 0.079 | 41% |
| 5 | 171 | 1.04 | 0.085 | 39% |
| 6 | 307 | 0.93 | 0.071 | 42% |
| 7 | 269 | 2.23 | 0.243 | 31% |
| 8 | 257 | 1.02 | 0.076 | 37% |
| 9 | 328 | 1.93 | 0.176 | 25% |
| 10 | 398 | 1.10 | 0.087 | 36% |
| 11 | 308 | 1.04 | 0.080 | 36% |
| 12 | 470 | 1.05 | 0.080 | 35% |
| 13 | 356 | 1.77 | 0.172 | 27% |
| 14 | 313 | 0.94 | 0.069 | 41% |
| 15 | 204 | 1.08 | 0.084 | 35% |
| 16 | 469 | 1.17 | 0.092 | 41% |
| 17 | 123 | 1.33 | 0.103 | 33% |
| 18 | 182 | 1.96 | 0.201 | 29% |
| 19 | 235 | 1.59 | 0.142 | 35% |


### F. Opportunity Analysis — DeepSeek-R1 8B

**Baseline**: 8241 tokens in 6110 steps = 1.35 tokens/step

**Oracle (skip all acc=0)**: 8241 tokens in 3965 steps = 2.08 tokens/step (+54%)

**Counterfactual chain-only (topk=1, vary steps)**:
- Efficiency: 0.505 (current: 0.474, Δ=+0.031)
- Note: same accept_length assumed — actual may differ with topk=1

**Efficiency by topk group:**
- topk=1: eff=0.679, acc=4.04, n=355 (6%)
- topk=2: eff=0.586, acc=2.81, n=313 (5%)
- topk≥3: eff=0.454, acc=1.09, n=5442 (89%)


### G. Topk vs Chain — DeepSeek-R1 8B

**Per-topk breakdown:**

| topk | Count | % | Mean Acc | Mean Steps | Eff | Zero% | Mean DOG |
|------|-------|---|----------|------------|-----|-------|----------|
| 1 | 355 | 6% | 4.04 | 6.1 | 0.679 | 15% | 0.526 |
| 2 | 313 | 5% | 2.81 | 5.5 | 0.586 | 15% | 0.318 |
| 3 | 2129 | 35% | 1.46 | 4.2 | 0.477 | 25% | 0.140 |
| 4 | 3313 | 54% | 0.85 | 3.2 | 0.440 | 45% | 0.041 |

**Topk=1 vs topk>1 at similar DOG levels:**

| DOG Range | topk=1 acc (n) | topk>1 acc (n) | Δ |
|-----------|---------------|----------------|---|
| [0.0, 0.2) | 0.67 (72) | 1.06 (5181) | +0.39 |
| [0.2, 0.3) | — | 2.07 (477) | — |
| [0.3, 0.5) | 3.95 (57) | 3.63 (97) | -0.32 |
| [0.5, 1.0) | 5.14 (226) | — | — |



## Cross-Model Comparison

| Model | Steps | Mean Acc | Eff | Zero% | Mean DOG |
|-------|-------|----------|-----|-------|----------|
| Llama 3.1 8B | 1507 | 2.95 | 0.607 | 11% | 0.329 |
| DeepSeek-R1 8B | 6110 | 1.35 | 0.474 | 35% | 0.118 |
