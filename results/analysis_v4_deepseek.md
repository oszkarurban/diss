# Dynamic Speculative Decoding V3 — Deep Analysis

*Generated from 14279 dynamic steps and 13825 static steps.*

*All AccLen values are SpecForge-compatible (accepted_draft + 1 bonus token).*

*Throughput = total_output_tokens / elapsed_seconds (same as SpecForge).*


## DeepSeek-R1 8B (V4)

### Dynamic vs Static Comparison — DeepSeek-R1 8B (V4)

| Metric | Static 7,4,8 | Dynamic V3 |
|--------|-------------|------------|
| Total steps | 13825 | 14279 |
| Total tokens | 30309 | 30136 |
| AccLen | 2.19 | 2.11 |
| Efficiency | 0.274 | 0.492 |
| Total rejections (AccLen=1) | 5144 (37.2%) | 5521 (38.7%) |
| Throughput (tok/s) | 105.2 | 114.8 |

**AccLen distribution:**

| AccLen | Static | Dynamic |
|--------|--------|---------|
| 1 | 5144 (37.2%) | 5521 (38.7%) |
| 2 | 4539 (32.8%) | 4599 (32.2%) |
| 3 | 2302 (16.7%) | 2508 (17.6%) |
| 4 | 930 (6.7%) | 952 (6.7%) |
| 5 | 440 (3.2%) | 395 (2.8%) |
| 6 | 201 (1.5%) | 133 (0.9%) |
| 7 | 97 (0.7%) | 56 (0.4%) |
| 8 | 172 (1.2%) | 115 (0.8%) |


### A. Signal-Acceptance Correlations — DeepSeek-R1 8B (V4)

**Same-step: signal(i) → accept_length(i)**

| Signal | r | Interpretation |
|--------|---|----------------|
| rolling_accept_rate | +0.370 | moderate |
| top1_prob | +0.000 | weak |
| target_top1_prob | +0.000 | weak |
| rolling_accept_length | +0.000 | weak |
| draft_oracle_gate | +0.000 | weak |
| target_oracle_gate_fixed | +0.000 | weak |
| joint_confidence_product_fixed | +0.000 | weak |
| confidence_agreement | +0.000 | weak |

**Next-step: signal(i) → accept_length(i+1)**

| Signal | r |
|--------|---|
| top1_prob | +0.000 |
| target_top1_prob | +0.000 |
| rolling_accept_rate | +0.329 |
| rolling_accept_length | +0.000 |
| draft_oracle_gate | +0.000 |
| target_oracle_gate_fixed | +0.000 |
| joint_confidence_product_fixed | +0.000 |
| confidence_agreement | +0.000 |


### B. Config Efficiency — DeepSeek-R1 8B (V4) (Dynamic)

**Per-config breakdown:**

| Config (topk,steps,ndt) | Count | % | AccLen | Eff | Zero% |
|-------------------------|-------|---|--------|-----|-------|
| (4, 3, 8) | 4125 | 28.9% | 2.00 | 0.501 | 38% |
| (4, 2, 8) | 3501 | 24.5% | 1.80 | 0.599 | 42% |
| (3, 4, 8) | 3329 | 23.3% | 2.16 | 0.432 | 38% |
| (4, 4, 8) | 1157 | 8.1% | 2.15 | 0.429 | 35% |
| (3, 5, 8) | 714 | 5.0% | 2.41 | 0.402 | 35% |
| (2, 5, 8) | 455 | 3.2% | 2.52 | 0.420 | 35% |
| (1, 7, 8) | 365 | 2.6% | 4.42 | 0.552 | 26% |
| (1, 3, 4) | 315 | 2.2% | 1.44 | 0.361 | 67% |
| (2, 6, 8) | 278 | 1.9% | 3.00 | 0.428 | 35% |
| (1, 6, 7) | 40 | 0.3% | 3.45 | 0.493 | 28% |

**Overall**: AccLen=2.11, Eff=0.492

**By DOG bucket:**

| DOG Range | Count | AccLen | Mean topk | Mean steps | Eff | Zero% |
|-----------|-------|--------|-----------|------------|-----|-------|
| [0.0, 0.1) | 14279 | 2.11 | 3.5 | 3.4 | 0.492 | 39% |


### B. Config Efficiency — DeepSeek-R1 8B (V4) (Static)

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


### C. Token-Level Analysis — DeepSeek-R1 8B (V4)

- Total steps with token data: 14279
- Total rejections (acc=0): 0 (0.0%)
- First-token rejections: 0
- Tree steps (topk>1): 13559
- Non-top-1 branch accepted: 8353/13559 (61.6%)


### D. Failure Modes — DeepSeek-R1 8B (V4)

**Total rejections**: 5521/14279 (38.7%)

**Config at total rejection:**

| Config | Count | % of rejections |
|--------|-------|-----------------|
| (4, 3, 8) | 1579 | 29% |
| (4, 2, 8) | 1459 | 26% |
| (3, 4, 8) | 1263 | 23% |
| (4, 4, 8) | 405 | 7% |
| (3, 5, 8) | 247 | 4% |
| (1, 3, 4) | 210 | 4% |
| (2, 5, 8) | 157 | 3% |
| (2, 6, 8) | 96 | 2% |
| (1, 7, 8) | 94 | 2% |
| (1, 6, 7) | 11 | 0% |

**Signal values: rejection vs success:**

| Signal | Mean@AccLen=1 | Mean@AccLen≥4 | Δ |
|--------|--------------|--------------|---|
| top1_prob | 0.000 | 0.000 | +0.000 |
| target_top1_prob | 0.000 | 0.000 | +0.000 |
| rolling_accept_rate | 0.151 | 0.260 | +0.110 |
| rolling_accept_length | 0.000 | 0.000 | +0.000 |
| draft_oracle_gate | 0.000 | 0.000 | +0.000 |
| target_oracle_gate_fixed | 0.000 | 0.000 | +0.000 |
| joint_confidence_product_fixed | 0.000 | 0.000 | +0.000 |
| confidence_agreement | 0.000 | 0.000 | +0.000 |

**Expensive failures** (acc=0, steps≥5): 605/5521 (11% of rejections)

**Consecutive failure streaks:**
- Total streaks: 3197
- Mean length: 1.7
- Max length: 20
- Distribution: {1: 1927, 2: 766, 3: 277, 4: 118, 5: 43, 6: 26, 7: 10, 8: 10, 9: 6, 10: 4, 11: 2, 12: 1, 13: 3, 16: 1, 17: 1, 18: 1, 20: 1}
- Streaks ≥5 (circuit breaker): 109


### E. Temporal Patterns — DeepSeek-R1 8B (V4)

**Accept_length autocorrelation:**

| Lag | r |
|-----|---|
| 1 | +0.295 |
| 2 | +0.240 |
| 3 | +0.206 |
| 5 | +0.192 |

**Per-turn summary** (40 turns):

| Turn | Steps | AccLen | Mean DOG | Zero% |
|------|-------|--------|----------|-------|
| 0 | 563 | 1.90 | 0.000 | 42% |
| 1 | 202 | 2.29 | 0.000 | 35% |
| 2 | 324 | 2.24 | 0.000 | 38% |
| 3 | 305 | 2.53 | 0.000 | 29% |
| 4 | 368 | 2.10 | 0.000 | 40% |
| 5 | 291 | 2.43 | 0.000 | 34% |
| 6 | 305 | 1.94 | 0.000 | 40% |
| 7 | 322 | 3.41 | 0.000 | 26% |
| 8 | 258 | 2.01 | 0.000 | 38% |
| 9 | 337 | 3.09 | 0.000 | 25% |
| 10 | 411 | 2.03 | 0.000 | 38% |
| 11 | 291 | 2.10 | 0.000 | 35% |
| 12 | 500 | 1.92 | 0.000 | 41% |
| 13 | 175 | 3.50 | 0.000 | 22% |
| 14 | 331 | 1.89 | 0.000 | 41% |
| 15 | 225 | 2.20 | 0.000 | 39% |
| 16 | 250 | 2.20 | 0.000 | 39% |
| 17 | 87 | 2.29 | 0.000 | 32% |
| 18 | 184 | 2.93 | 0.000 | 32% |
| 19 | 176 | 2.38 | 0.000 | 35% |
| 20 | 634 | 1.96 | 0.000 | 41% |
| 21 | 247 | 2.09 | 0.000 | 32% |
| 22 | 306 | 1.90 | 0.000 | 44% |
| 23 | 230 | 1.94 | 0.000 | 39% |
| 24 | 442 | 1.87 | 0.000 | 46% |
| 25 | 278 | 2.29 | 0.000 | 36% |
| 26 | 978 | 2.09 | 0.000 | 37% |
| 27 | 730 | 1.58 | 0.000 | 50% |
| 28 | 177 | 1.76 | 0.000 | 51% |
| 29 | 257 | 2.07 | 0.000 | 39% |
| 30 | 267 | 2.07 | 0.000 | 39% |
| 31 | 116 | 2.14 | 0.000 | 35% |
| 32 | 700 | 1.96 | 0.000 | 40% |
| 33 | 1123 | 1.82 | 0.000 | 42% |
| 34 | 151 | 1.99 | 0.000 | 38% |
| 35 | 204 | 1.91 | 0.000 | 44% |
| 36 | 348 | 1.91 | 0.000 | 43% |
| 37 | 356 | 2.35 | 0.000 | 33% |
| 38 | 337 | 2.01 | 0.000 | 38% |
| 39 | 493 | 2.31 | 0.000 | 35% |


### F. Opportunity Analysis — DeepSeek-R1 8B (V4)

**Baseline**: 30136 tokens in 14279 steps, AccLen=2.11

**Oracle (skip all AccLen=1 steps)**: 24615 tokens in 8758 steps, AccLen=2.81 (+33%)

**Counterfactual chain-only (topk=1, vary steps)**:
- Efficiency: 0.528 (current: 0.492, Δ=+0.035)
- Note: same AccLen assumed — actual may differ with topk=1

**Efficiency by topk group:**
- topk=1: eff=0.465, AccLen=3.06, n=720 (5%)
- topk=2: eff=0.423, AccLen=2.70, n=733 (5%)
- topk≥3: eff=0.498, AccLen=2.02, n=12826 (90%)


### G. Topk vs Chain — DeepSeek-R1 8B (V4)

**Per-topk breakdown:**

| topk | Count | % | AccLen | Mean Steps | Eff | Zero% | Mean DOG |
|------|-------|---|--------|------------|-----|-------|----------|
| 1 | 720 | 5% | 3.06 | 5.2 | 0.465 | 44% | 0.000 |
| 2 | 733 | 5% | 2.70 | 5.4 | 0.423 | 35% | 0.000 |
| 3 | 4043 | 28% | 2.20 | 4.2 | 0.427 | 37% | 0.000 |
| 4 | 8783 | 62% | 1.94 | 2.7 | 0.531 | 39% | 0.000 |

**Topk=1 vs topk>1 at similar DOG levels:**

| DOG Range | topk=1 AccLen (n) | topk>1 AccLen (n) | Δ |
|-----------|-------------------|-------------------|---|
| [0.0, 0.2) | 3.06 (720) | 2.06 (13559) | -1.00 |
| [0.2, 0.3) | — | — | — |
| [0.3, 0.5) | — | — | — |
| [0.5, 1.0) | — | — | — |


