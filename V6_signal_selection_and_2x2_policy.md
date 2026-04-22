# Dynamic Speculative Decoding — Signal Selection & 2×2 Policy Design

## 1. Background

EAGLE3 speculative decoding has three tree hyperparameters: `num_steps` (depth), `topk` (branching factor per node), and `num_draft_tokens` (pruning budget). SGLang's dynamic speculative decoding adjusts these per step based on runtime signals.

**Starting points tested:**

- **V4 (incumbent):** single signal `rolling_accept_rate` (RAR), zero GPU sync, piecewise-linear mapping across three zones.
- **V5 (this project's extension):** 5 signals jointly averaged into a scalar `confidence ∈ [0,1]`, which is interpolated to `(topk, num_steps, ndt)` between start and max bounds.

Bounds for all experiments: start `(3, 1, 4)`, max `(7, 4, 8)` — same config space as the V3 era, close to the latest static SOTA on Llama `(7, 4, 8)` @ 209 tok/s.

## 2. Signals considered

Five signals were instrumented and logged per decode step in V5:

| Signal | Source | Definition | Initial expectation |
|---|---|---|---|
| `rolling_accept_rate` (RAR) | CPU EMA per-request (α=0.3, init 0.5) | Exponentially smoothed acceptance rate of the draft | Historical memory of how well the draft has been doing |
| `top1_prob` | GPU — `topk_p[:, 0].mean()` | Draft model's instantaneous top-1 probability | Draft's own confidence in what it's proposing |
| `draft_oracle_gate` (DOG) | CPU — `top1_prob × RAR` | Product of instantaneous draft confidence and historical acceptance | V3's champion signal (R²=0.38 on accept_length) |
| `target_top1_prob` | GPU — `F.softmax(target_logits).topk(2)[:,0].mean()` | Target model's top-1 probability | Target's decisiveness at each verified position |
| `target_entropy` | GPU — `-(p * log(p)).sum()`, mean-over-positions | Entropy of target's softmax distribution | Target distribution sharpness (low = easy) |

Each carries a GPU-sync cost only when explicitly read. V5 packed all GPU-side signals into one batched `.tolist()` call per step, keeping the overhead at 2 GPU→CPU stalls per decode step.

## 3. Llama MT-Bench 80q V5 trace — key statistics

14,474 decode steps observed, bs=1, temperature=0, A100.

| Signal | Mean | Std | Min → Max |
|---|---:|---:|---|
| `rolling_accept_rate` | 0.455 | 0.183 | 0.02 → 1.00 |
| `draft_oracle_gate` | 0.362 | 0.215 | 0.00 → 1.00 |
| `top1_prob` | 0.769 | 0.264 | 0.01 → 1.00 |
| `target_top1_prob` | 0.838 | 0.114 | 0.33 → 1.00 |
| `target_entropy` | 0.511 (raw, [0,5]) | 0.384 | 0.00 → 3.50 |
| V5 `confidence` (unweighted mean) | 0.664 | 0.137 | 0.20 → 1.00 |

V5 throughput: **206.7 tok/s, accept 4.15** — below static `(7,4,8)` at 209.4 tok/s.

## 4. Signal selection — the four empirical tests

### Test 1 — Direct correlation with per-step throughput

| Signal | Pearson | Spearman |
|---|---:|---:|
| `rolling_accept_rate` | 0.347 | 0.293 |
| `draft_oracle_gate` | 0.512 | 0.488 |
| **`top1_prob`** | **0.476** | **0.496** |
| `target_top1_prob` | 0.333 | 0.341 |
| `target_entropy` | −0.331 | −0.349 |

### Test 2 — Marginal information over RAR (Spearman on throughput residuals after regressing out RAR)

Answers "which signal adds the most *new* information once RAR is already known":

| Signal | Spearman on residuals |
|---|---:|
| **`top1_prob`** | **+0.406** ⭐ |
| `draft_oracle_gate` | +0.243 |
| `target_top1_prob` | +0.169 |
| `target_entropy` | −0.174 |
| `rolling_accept_rate` | 0.000 (baseline, by construction) |

`top1_prob` carries ~2× the independent predictive value of any other signal.

### Test 3 — Signal redundancy (Pearson between signal pairs)

| Pair | Pearson | Interpretation |
|---|---:|---|
| `target_top1_prob` vs `target_entropy` | **−0.955** | Near-identical — 97% agreement on "target-high" classification |
| `rolling_accept_rate` vs `draft_oracle_gate` | **+0.845** | DOG is dominated by RAR (it's top1×RAR) |
| `top1_prob` vs `draft_oracle_gate` | +0.693 | Large overlap via top1 factor |
| `top1_prob` vs `rolling_accept_rate` | +0.254 | Loosely related — instantaneous vs historical |
| `top1_prob` vs `target_top1_prob` | +0.307 | Partly independent — useful as 2D axes |

### Test 4 — Step-to-step jitter (avg |Δ|)

| Signal | Jitter | Character |
|---|---:|---|
| `top1_prob` | 0.250 | Highly responsive to content difficulty |
| `target_entropy` | 0.284 (raw scale) | Comparable |
| `draft_oracle_gate` | 0.118 | Smoothed via RAR's EMA |
| `target_top1_prob` | 0.089 | Narrow range, slow-changing |
| `rolling_accept_rate` | 0.073 | EMA — slow-changing |

Jitter looks like a downside, but config-swap in sglang is a cheap pointer reassignment, so per-step responsiveness is a feature. High-jitter signals shouldn't be preferentially smoothed out.

## 5. Signal decisions

Based on the four tests:

| Signal | Decision | Reason |
|---|---|---|
| **`top1_prob`** | **Keep — draft axis of 2×2** | Highest marginal information over RAR; best single per-step predictor |
| **`target_top1_prob`** | **Keep — target axis of 2×2** | Captures independent target-side dimension; natively [0,1] |
| `target_entropy` | Drop from policy; stop computing | Pearson −0.955 with target_top1_prob; 97% partition agreement; redundant |
| `draft_oracle_gate` | Drop from policy | Pearson +0.84 with RAR and +0.69 with top1_prob; its +0.24 marginal is half of top1_prob's +0.41; redundant |
| `rolling_accept_rate` | Keep logged, not a policy axis | Free CPU signal; available for the 5-consecutive-failures circuit breaker; contextual diagnostic |

**Final signal set driving policy: 2 signals (`top1_prob`, `target_top1_prob`). GPU cost: 2 batched syncs/step (unchanged from V5, but `eagle_info.py` entropy compute dropped — small speedup).**

## 6. The 2×2 policy

### Rationale

A linear combination like V5's unweighted mean systematically biased confidence upward because the target-side signals saturate near their maximum on Llama (`target_top1_prob` mean 0.84, ~constant). This pushed V5 into a narrow mid-config corridor (`(2, 6, 8)` in 51% of steps) and prevented it from firing either the deep chain at the high end or the minimal chain at the low end.

A 2×2 matrix on `(top1_prob, target_top1_prob)` gives four distinct regimes, each with a qualitatively different correct response:

| Cell | Meaning | Right response |
|---|---|---|
| **CC — Certain × Certain** | Both models agree this is easy | Deep chain — high payoff, cheap per-step |
| **CU — Certain × Uncertain** | Draft commits; target has many plausible tokens | Medium chain — some commits land |
| **UC — Uncertain × Certain** | Draft is scattered; target has a clear winner | Wide tree — try multiple candidates so one hits target's winner |
| **UU — Uncertain × Uncertain** | Both lost | Minimal config — don't burn compute |

Width (`topk > 1`) helps only in UC, where multiple candidate tokens stand a chance of matching a decisive target. In the other three cells, a **chain (`topk = 1`)** is more efficient because there's no benefit from spreading draft tokens across branches.

### Llama MT-Bench thresholds (median-split)

| Axis | Threshold (median of the V5 Llama trace) |
|---|---|
| `top1_prob` ≥ **0.894** | Draft is confident |
| `target_top1_prob` ≥ **0.852** | Target is confident |

Thresholds are the observed medians; tuning (per-model, or adaptive) is a follow-up once DeepSeek and Qwen traces are available.

### Cell shares on Llama MT-Bench 80q

| Cell | N | % of steps | Accept | Zero-rate | Big-win (≥5) | Throughput proxy | V5's choice today |
|---|---:|---:|---:|---:|---:|---:|---|
| **CC** (both ≥ threshold) | 4,587 | **31.7%** | 4.49 | 1.8% | 56% | **227 tok/s** | (2, 6, 8) — suboptimal |
| **CU** (draft high, target low) | 2,650 | 18.3% | 3.30 | 3.0% | 28% | 181 tok/s | (2, 6, 8) |
| **UC** (draft low, target high) | 2,650 | 18.3% | 2.50 | 21% | 21% | 149 tok/s | (2, 6, 8) |
| **UU** (both < threshold) | 4,587 | **31.7%** | 2.04 | 22% | 12% | **131 tok/s** | (2, 5, 8) |

Note the throughput gradient: **CC is 73% more productive per step than UU**. The policy's job is to exploit that gradient by picking cheap configs in UU (where productivity is low regardless of config) and aggressive configs in CC (where every extra accepted token is essentially free).

V5 currently misses both ends — it picks `(2, 6, 8)` in three of four cells — which is why it barely matches static `(7, 4, 8)` despite having the flexibility to do more.

### V6 config per cell

| Cell | V6 config | Justification |
|---|---|---|
| CC | `(topk=1, num_steps=num_steps_max=7, ndt=8)` | Deep chain — (1,7,8) already achieves ~263 tok/s when V5 fires it on a subset of CC steps |
| CU | `(topk=1, num_steps=(start+max)//2=5, ndt=6)` | Medium chain — draft commits, but target is hedging; don't over-commit depth |
| UC | `(topk=topk_max=4, num_steps=num_steps_start+1=2, ndt=min(ndt_max, topk*steps+1)=8)` | Wide shallow tree — only cell where width is justified |
| UU | `(topk=1, num_steps=2, ndt=3)` | Minimal chain — don't waste draft compute when neither model is decisive |

### Expected behaviour change from V5 to V6 on Llama

- **CC (32% of trace)**: (2,6,8) → (1,7,8). V5 observed `(1,7,8)` throughput ≈ 263 tok/s. Using it on all CC steps (vs V5's 7% coverage) should lift CC-cell throughput from 227 → ~255 tok/s.
- **UU (32% of trace)**: (2,5,8) → (1,2,3). Same accept rate (~2 tok), but step_time drops by roughly 2× → per-step throughput nearly doubles on this chunk.
- **CU (18%)**: minor shift, ~neutral.
- **UC (18%)**: widening may or may not help. Separate experiment once V6 is live.

The biggest gains come from the two corner cells (CC and UU), which together cover **63% of the trace**.

## 7. Summary of decisions

1. **Signal count: 5 → 2 for policy decisions (3 logged total, including RAR).** Target entropy and `draft_oracle_gate` dropped as redundant; RAR kept only for logging and circuit-breaker use.
2. **Selection criterion: marginal Spearman on throughput residuals after RAR** — not raw correlation, because raw correlation over-credits signals that partly encode RAR.
3. **Policy topology: 2×2 hard thresholds** on `(top1_prob, target_top1_prob)`, not a linear confidence scalar — because the cell-to-throughput gradient is large (131 → 227 tok/s) and the correct config per cell is qualitatively different (chain in three cells, tree only in one).
4. **Thresholds: observed medians on Llama** (0.894 / 0.852) — to be recalibrated when DeepSeek and Qwen traces arrive; per-model static thresholds are the default, adaptive quantile thresholds a possible follow-up.
5. **Config assignment per cell: derived from V5 cell statistics** — deep chain in CC (where (1,7,8) already wins), wide tree only in UC (where width's theoretical rationale holds), minimal chain in UU (where any config is roughly equally unproductive, so pick the cheapest).
