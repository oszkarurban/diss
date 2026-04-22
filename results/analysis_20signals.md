# 20-Signal Deep Analysis for Dynamic Speculative Decoding

**Date**: Generated from analysis_20signals.py
**Datasets**: Llama-8B, DS+LlamaDraft

---

## Section 1: Signal Encyclopedia

### 1.1 `draft_entropy` — Draft

- **Formula**: `-sum(p log p) of draft softmax`
- **Collected from**: Draft phase
- **High value means**: Draft model uncertain about prediction
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 1.6034 | 1.5301 | 0.0000 | 8.8007 |
| DS+LlamaDraft | 6863 | 2.9052 | 1.9084 | 0.0000 | 8.4195 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.2395 | -0.2308 | -0.2685 | -0.5159 |
| DS+LlamaDraft | -0.2313 | -0.2762 | -0.3027 | -0.5087 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 1.9690 | 1.8734 | 1.8103 | 1.4562 | 0.6611 |
| DS+LlamaDraft | 3.1784 | 2.9068 | 2.8927 | 2.3480 | 0.8060 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.525 | Moderate separator |
| DS+LlamaDraft | -0.752 | Moderate separator |

---

### 1.2 `top1_prob` — Draft

- **Formula**: `topk_p[:, 0].mean()`
- **Collected from**: Extend phase
- **High value means**: Draft strongly favors one token
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.7352 | 0.2729 | 0.0470 | 1.0000 |
| DS+LlamaDraft | 6863 | 0.5443 | 0.2981 | 0.0072 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5598 | +0.1322 | +0.4125 | +0.5926 |
| DS+LlamaDraft | +0.5487 | +0.2096 | +0.4459 | +0.5992 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.4450 | 0.7571 | 0.7723 | 0.8266 | 0.9523 |
| DS+LlamaDraft | 0.3926 | 0.6437 | 0.6776 | 0.7966 | 0.9488 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +2.001 | **Strong separator** |
| DS+LlamaDraft | +2.047 | **Strong separator** |

---

### 1.3 `top1_minus_top2` — Draft

- **Formula**: `(topk_p[:,0] - topk_p[:,1]).mean()`
- **Collected from**: Extend phase
- **High value means**: Large gap between top-2 draft predictions
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.6358 | 0.3542 | 0.0000 | 1.0000 |
| DS+LlamaDraft | 6863 | 0.4301 | 0.3469 | 0.0000 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5514 | +0.1266 | +0.3996 | +0.5773 |
| DS+LlamaDraft | +0.5537 | +0.2198 | +0.4518 | +0.5926 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2711 | 0.6586 | 0.6812 | 0.7458 | 0.9284 |
| DS+LlamaDraft | 0.2560 | 0.5371 | 0.5792 | 0.7258 | 0.9290 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +2.029 | **Strong separator** |
| DS+LlamaDraft | +2.048 | **Strong separator** |

---

### 1.4 `hidden_norm` — Draft hidden

- **Formula**: `torch.norm(hidden_states, dim=-1).mean()`
- **Collected from**: Extend phase
- **High value means**: High activation magnitude in draft representation
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 64.0152 | 10.4547 | 35.5000 | 117.5000 |
| DS+LlamaDraft | 6863 | 78.3585 | 11.4552 | 38.0000 | 142.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.1719 | -0.0929 | -0.1595 | -0.1574 |
| DS+LlamaDraft | -0.0009 | +0.0191 | +0.0328 | +0.1161 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 67.0384 | 64.1759 | 63.9549 | 62.9228 | 61.1042 |
| DS+LlamaDraft | 78.6639 | 77.6412 | 78.1111 | 77.6405 | 80.5954 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.448 | Weak separator |
| DS+LlamaDraft | -0.001 | Not useful |

---

### 1.5 `hidden_cosine_sim` — Draft hidden

- **Formula**: `cosine_sim(h[step_i], h[step_i-1])`
- **Collected from**: Draft phase
- **High value means**: Draft representation stable between consecutive steps
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.6509 | 0.1698 | 0.2095 | 1.0000 |
| DS+LlamaDraft | 6863 | 0.7770 | 0.1652 | 0.0000 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.1015 | -0.2665 | -0.2030 | -0.0943 |
| DS+LlamaDraft | -0.1463 | -0.3215 | -0.2448 | -0.1516 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6797 | 0.6565 | 0.6405 | 0.6438 | 0.6218 |
| DS+LlamaDraft | 0.7927 | 0.7823 | 0.7557 | 0.7438 | 0.6872 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.251 | Weak separator |
| DS+LlamaDraft | -0.437 | Weak separator |

---

### 1.6 `hidden_projection_score` — Draft hidden

- **Formula**: `h · W_lm[predicted_token] / ||W||`
- **Collected from**: Draft phase
- **High value means**: Draft hidden strongly commits to its top prediction
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 1.2478 | 4.8614 | -13.6875 | 22.1875 |
| DS+LlamaDraft | 6863 | 0.5000 | 4.9797 | -17.7500 | 30.3750 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0698 | +0.1281 | +0.1176 | +0.1923 |
| DS+LlamaDraft | +0.0461 | +0.0985 | +0.0803 | +0.1403 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.8447 | 1.2484 | 0.9557 | 1.2159 | 2.3770 |
| DS+LlamaDraft | 0.4056 | 0.3623 | 0.5404 | 0.6501 | 1.9257 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.149 | Not useful |
| DS+LlamaDraft | +0.131 | Not useful |

---

### 1.7 `hidden_entropy` — Draft hidden

- **Formula**: `entropy(h²/sum(h²))`
- **Collected from**: Draft phase
- **High value means**: Draft activation energy spread across many dimensions
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 4.7023 | 0.1299 | 3.8594 | 4.8594 |
| DS+LlamaDraft | 6863 | 4.7846 | 0.1105 | 0.0000 | 4.8750 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0717 | +0.0775 | +0.0707 | +0.0407 |
| DS+LlamaDraft | +0.0061 | -0.0073 | +0.0042 | -0.0591 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 4.6939 | 4.6874 | 4.7011 | 4.7142 | 4.7128 |
| DS+LlamaDraft | 4.7848 | 4.7831 | 4.7864 | 4.7776 | 4.7988 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.155 | Not useful |
| DS+LlamaDraft | +0.000 | Not useful |

---

### 1.8 `target_entropy` — Target

- **Formula**: `-sum(p log p) of target softmax`
- **Collected from**: Verify phase
- **High value means**: Target model uncertain — harder to match
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.6013 | 0.4491 | 0.0000 | 2.9906 |
| DS+LlamaDraft | 6863 | 0.8105 | 0.5597 | 0.0000 | 4.7706 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.3437 | -0.3917 | -0.4900 | -0.7184 |
| DS+LlamaDraft | -0.2897 | -0.3702 | -0.4182 | -0.6882 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.7602 | 0.7077 | 0.6799 | 0.5476 | 0.1984 |
| DS+LlamaDraft | 0.9101 | 0.8216 | 0.8047 | 0.5408 | 0.1206 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.757 | Moderate separator |
| DS+LlamaDraft | -1.058 | **Strong separator** |

---

### 1.9 `target_top1_gap` — Target

- **Formula**: `target top-1 minus top-2 probability`
- **Collected from**: Verify phase
- **High value means**: Target has one clear winner — easy to match
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.7128 | 0.1910 | 0.0841 | 1.0000 |
| DS+LlamaDraft | 6863 | 0.6494 | 0.2037 | 0.0195 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.3294 | +0.3993 | +0.4839 | +0.7008 |
| DS+LlamaDraft | +0.3144 | +0.4062 | +0.4514 | +0.6374 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6478 | 0.6741 | 0.6847 | 0.7224 | 0.8938 |
| DS+LlamaDraft | 0.6127 | 0.6370 | 0.6567 | 0.7447 | 0.9416 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.697 | Moderate separator |
| DS+LlamaDraft | +1.068 | **Strong separator** |

---

### 1.10 `target_varentropy` — Target

- **Formula**: `Var(-log p) of target distribution`
- **Collected from**: Verify phase
- **High value means**: Target uncertainty is spread (uniform-ish) — harder
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.9268 | 0.7506 | 0.0002 | 6.3580 |
| DS+LlamaDraft | 6863 | 1.5977 | 1.4160 | 0.0004 | 12.1061 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.2914 | -0.3195 | -0.4083 | -0.6130 |
| DS+LlamaDraft | -0.2278 | -0.2916 | -0.3345 | -0.6042 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 1.1334 | 1.1062 | 1.0531 | 0.8365 | 0.3601 |
| DS+LlamaDraft | 1.8034 | 1.5976 | 1.5948 | 1.0356 | 0.2797 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.634 | Moderate separator |
| DS+LlamaDraft | -0.846 | **Strong separator** |

---

### 1.11 `target_hidden_cosine_sim` — Target hidden

- **Formula**: `cosine_sim between consecutive target positions`
- **Collected from**: Verify phase
- **High value means**: Target representation stable across draft positions
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.6059 | 0.0790 | 0.3184 | 0.8359 |
| DS+LlamaDraft | 6863 | 0.4980 | 0.0887 | 0.1602 | 0.8516 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0724 | +0.0531 | +0.0849 | +0.1669 |
| DS+LlamaDraft | +0.0518 | +0.0648 | +0.0764 | +0.2419 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6039 | 0.6024 | 0.5940 | 0.6051 | 0.6310 |
| DS+LlamaDraft | 0.4947 | 0.5009 | 0.4955 | 0.5004 | 0.5246 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.116 | Not useful |
| DS+LlamaDraft | +0.168 | Not useful |

---

### 1.12 `target_projection_score` — Target hidden

- **Formula**: `h · W_target_lm[predicted] / ||W||`
- **Collected from**: Verify phase
- **High value means**: Target hidden commits to its prediction
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.0088 | 0.0157 | -0.0483 | 0.0703 |
| DS+LlamaDraft | 6863 | 0.0129 | 0.0179 | -0.0549 | 0.0859 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0290 | +0.0846 | +0.0677 | +0.1692 |
| DS+LlamaDraft | -0.0027 | -0.0071 | -0.0020 | +0.1199 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.0092 | 0.0082 | 0.0072 | 0.0084 | 0.0120 |
| DS+LlamaDraft | 0.0129 | 0.0133 | 0.0120 | 0.0123 | 0.0141 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.018 | Not useful |
| DS+LlamaDraft | +0.000 | Not useful |

---

### 1.13 `target_hidden_entropy` — Target hidden

- **Formula**: `entropy(h²/sum(h²)) of target hidden`
- **Collected from**: Verify phase
- **High value means**: Target activation energy spread
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 4.1260 | 0.2158 | 3.2969 | 4.8125 |
| DS+LlamaDraft | 6863 | 4.6657 | 0.1475 | 3.6719 | 4.8750 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.1341 | -0.0908 | -0.1816 | -0.2946 |
| DS+LlamaDraft | -0.2542 | -0.2714 | -0.3492 | -0.4614 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 4.1508 | 4.1355 | 4.1559 | 4.1344 | 4.0135 |
| DS+LlamaDraft | 4.6870 | 4.6727 | 4.6627 | 4.6143 | 4.4869 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.246 | Weak separator |
| DS+LlamaDraft | -0.752 | Moderate separator |

---

### 1.14 `kl_approx_target_draft` — Divergence

- **Formula**: `F.cross_entropy(target_logits, draft_tokens)`
- **Collected from**: Verify phase
- **High value means**: Target assigns low probability to draft's choices
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 3.0076 | 3.5413 | 0.0000 | 22.6992 |
| DS+LlamaDraft | 6863 | 5.8657 | 4.5298 | 0.0000 | 24.4724 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.0154 | -0.0365 | -0.0411 | -0.2259 |
| DS+LlamaDraft | -0.1328 | -0.2743 | -0.2663 | -0.5190 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 3.0311 | 2.9588 | 2.8973 | 3.4053 | 2.2760 |
| DS+LlamaDraft | 6.3044 | 5.6334 | 5.9125 | 5.2247 | 3.0544 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.008 | Not useful |
| DS+LlamaDraft | -0.427 | Weak separator |

---

### 1.15 `target_draft_agree` — Agreement

- **Formula**: `fraction argmax(target)==draft_token`
- **Collected from**: Verify phase
- **High value means**: Target and draft agree on token choices
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.6504 | 0.2867 | 0.0000 | 1.0000 |
| DS+LlamaDraft | 6863 | 0.3867 | 0.3237 | 0.0000 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.2112 | +0.5448 | +0.4620 | +0.6268 |
| DS+LlamaDraft | +0.2631 | +0.5855 | +0.5317 | +0.6916 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.5829 | 0.6253 | 0.6351 | 0.6384 | 0.8488 |
| DS+LlamaDraft | 0.3342 | 0.3837 | 0.4014 | 0.4739 | 0.8075 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.430 | Weak separator |
| DS+LlamaDraft | +0.824 | **Strong separator** |

---

### 1.16 `joint_entropy_gate` — Joint

- **Formula**: `(1/(1+draft_ent)) * (1/(1+target_ent))`
- **Collected from**: Derived
- **High value means**: Both models are certain
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.3657 | 0.2369 | 0.0322 | 0.9982 |
| DS+LlamaDraft | 6863 | 0.2242 | 0.1915 | 0.0226 | 0.9993 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.3109 | +0.2868 | +0.3834 | +0.6558 |
| DS+LlamaDraft | +0.3839 | +0.4468 | +0.5057 | +0.6569 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2948 | 0.3244 | 0.3246 | 0.3623 | 0.6060 |
| DS+LlamaDraft | 0.1845 | 0.2112 | 0.2227 | 0.2928 | 0.6305 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.632 | Moderate separator |
| DS+LlamaDraft | +1.007 | **Strong separator** |

---

### 1.17 `draft_oracle_gate` — Joint

- **Formula**: `top1_prob * rolling_accept_rate`
- **Collected from**: Derived
- **High value means**: Draft confident AND target has been accepting
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.4083 | 0.2105 | 0.0156 | 0.8746 |
| DS+LlamaDraft | 6863 | 0.1646 | 0.1674 | 0.0002 | 0.9202 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5540 | +0.6934 | +0.6441 | +0.7315 |
| DS+LlamaDraft | +0.5916 | +0.7916 | +0.7391 | +0.7007 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2288 | 0.3826 | 0.4175 | 0.4511 | 0.6470 |
| DS+LlamaDraft | 0.0976 | 0.1716 | 0.1958 | 0.2872 | 0.5902 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +1.689 | **Strong separator** |
| DS+LlamaDraft | +1.602 | **Strong separator** |

---

### 1.18 `target_oracle_gate` — Joint

- **Formula**: `target_top1_gap * rolling_accept_rate`
- **Collected from**: Derived
- **High value means**: Target confident AND has been accepting
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.4029 | 0.1882 | 0.0244 | 0.8640 |
| DS+LlamaDraft | 6863 | 0.1978 | 0.1686 | 0.0004 | 0.9191 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.3423 | +0.8561 | +0.6354 | +0.7155 |
| DS+LlamaDraft | +0.4325 | +0.9295 | +0.7224 | +0.6675 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.3438 | 0.3542 | 0.3792 | 0.4015 | 0.6074 |
| DS+LlamaDraft | 0.1602 | 0.1778 | 0.1961 | 0.2807 | 0.5888 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.677 | Moderate separator |
| DS+LlamaDraft | +1.168 | **Strong separator** |

---

### 1.19 `joint_confidence_product` — Joint

- **Formula**: `top1_prob * target_top1_gap`
- **Collected from**: Derived
- **High value means**: Both models confident in their top prediction
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.5353 | 0.2615 | 0.0119 | 1.0000 |
| DS+LlamaDraft | 6863 | 0.3660 | 0.2571 | 0.0041 | 0.9999 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5922 | +0.3108 | +0.5678 | +0.8038 |
| DS+LlamaDraft | +0.6125 | +0.3865 | +0.6000 | +0.7629 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2949 | 0.5100 | 0.5362 | 0.5946 | 0.8519 |
| DS+LlamaDraft | 0.2416 | 0.4131 | 0.4537 | 0.5952 | 0.8949 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +1.854 | **Strong separator** |
| DS+LlamaDraft | +2.130 | **Strong separator** |

---

### 1.20 `rolling_accept_rate` — Historical

- **Formula**: `alpha*step_rate + (1-alpha)*prev, alpha=0.3`
- **Collected from**: Verify EMA
- **High value means**: Recent acceptance has been high
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2021 | 0.5470 | 0.1706 | 0.0530 | 0.9420 |
| DS+LlamaDraft | 6863 | 0.2822 | 0.1761 | 0.0010 | 0.9202 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.2477 | +1.0000 | +0.5696 | +0.5196 |
| DS+LlamaDraft | +0.3571 | +1.0000 | +0.6597 | +0.5564 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.5113 | 0.5077 | 0.5337 | 0.5480 | 0.6790 |
| DS+LlamaDraft | 0.2502 | 0.2638 | 0.2794 | 0.3561 | 0.6193 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.467 | Weak separator |
| DS+LlamaDraft | +1.011 | **Strong separator** |

---

## Section 2: Correlation Rankings

### Ranked by \|r(signal, accept_length)|

| Rank | Signal | r (Llama-8B) | r (DS+LlamaDraft) | Avg \|r\| | Consistent? |
|------|--------|------|------|---------|-------------|
| 1 | joint_confidence_product | +0.5922 | +0.6125 | 0.6023 | YES |
| 2 | draft_oracle_gate | +0.5540 | +0.5916 | 0.5728 | YES |
| 3 | top1_prob | +0.5598 | +0.5487 | 0.5542 | YES |
| 4 | top1_minus_top2 | +0.5514 | +0.5537 | 0.5525 | YES |
| 5 | target_oracle_gate | +0.3423 | +0.4325 | 0.3874 | YES |
| 6 | joint_entropy_gate | +0.3109 | +0.3839 | 0.3474 | YES |
| 7 | target_top1_gap | +0.3294 | +0.3144 | 0.3219 | YES |
| 8 | target_entropy | -0.3437 | -0.2897 | 0.3167 | YES |
| 9 | rolling_accept_rate | +0.2477 | +0.3571 | 0.3024 | YES |
| 10 | target_varentropy | -0.2914 | -0.2278 | 0.2596 | YES |
| 11 | target_draft_agree | +0.2112 | +0.2631 | 0.2372 | YES |
| 12 | draft_entropy | -0.2395 | -0.2313 | 0.2354 | YES |
| 13 | target_hidden_entropy | -0.1341 | -0.2542 | 0.1941 | YES |
| 14 | hidden_cosine_sim | -0.1015 | -0.1463 | 0.1239 | YES |
| 15 | hidden_norm | -0.1719 | -0.0009 | 0.0864 | YES |
| 16 | kl_approx_target_draft | -0.0154 | -0.1328 | 0.0741 | YES |
| 17 | target_hidden_cosine_sim | +0.0724 | +0.0518 | 0.0621 | YES |
| 18 | hidden_projection_score | +0.0698 | +0.0461 | 0.0579 | YES |
| 19 | hidden_entropy | +0.0717 | +0.0061 | 0.0389 | YES |
| 20 | target_projection_score | +0.0290 | -0.0027 | 0.0158 | NO |

### Ranked by \|r(signal, rolling_accept_rate)|

| Rank | Signal | r (Llama-8B) | r (DS+LlamaDraft) | Avg \|r\| | Consistent? |
|------|--------|------|------|---------|-------------|
| 1 | rolling_accept_rate | +1.0000 | +1.0000 | 1.0000 | YES |
| 2 | target_oracle_gate | +0.8561 | +0.9295 | 0.8928 | YES |
| 3 | draft_oracle_gate | +0.6934 | +0.7916 | 0.7425 | YES |
| 4 | target_draft_agree | +0.5448 | +0.5855 | 0.5652 | YES |
| 5 | target_top1_gap | +0.3993 | +0.4062 | 0.4027 | YES |
| 6 | target_entropy | -0.3917 | -0.3702 | 0.3810 | YES |
| 7 | joint_entropy_gate | +0.2868 | +0.4468 | 0.3668 | YES |
| 8 | joint_confidence_product | +0.3108 | +0.3865 | 0.3486 | YES |
| 9 | target_varentropy | -0.3195 | -0.2916 | 0.3056 | YES |
| 10 | hidden_cosine_sim | -0.2665 | -0.3215 | 0.2940 | YES |
| 11 | draft_entropy | -0.2308 | -0.2762 | 0.2535 | YES |
| 12 | target_hidden_entropy | -0.0908 | -0.2714 | 0.1811 | YES |
| 13 | top1_minus_top2 | +0.1266 | +0.2198 | 0.1732 | YES |
| 14 | top1_prob | +0.1322 | +0.2096 | 0.1709 | YES |
| 15 | kl_approx_target_draft | -0.0365 | -0.2743 | 0.1554 | YES |
| 16 | hidden_projection_score | +0.1281 | +0.0985 | 0.1133 | YES |
| 17 | target_hidden_cosine_sim | +0.0531 | +0.0648 | 0.0589 | YES |
| 18 | hidden_norm | -0.0929 | +0.0191 | 0.0560 | NO |
| 19 | target_projection_score | +0.0846 | -0.0071 | 0.0458 | NO |
| 20 | hidden_entropy | +0.0775 | -0.0073 | 0.0424 | NO |

### Ranked by \|r(signal, windowed_AL_3)|

| Rank | Signal | r (Llama-8B) | r (DS+LlamaDraft) | Avg \|r\| | Consistent? |
|------|--------|------|------|---------|-------------|
| 1 | draft_oracle_gate | +0.6441 | +0.7391 | 0.6916 | YES |
| 2 | target_oracle_gate | +0.6354 | +0.7224 | 0.6789 | YES |
| 3 | rolling_accept_rate | +0.5696 | +0.6597 | 0.6147 | YES |
| 4 | joint_confidence_product | +0.5678 | +0.6000 | 0.5839 | YES |
| 5 | target_draft_agree | +0.4620 | +0.5317 | 0.4969 | YES |
| 6 | target_top1_gap | +0.4839 | +0.4514 | 0.4677 | YES |
| 7 | target_entropy | -0.4900 | -0.4182 | 0.4541 | YES |
| 8 | joint_entropy_gate | +0.3834 | +0.5057 | 0.4446 | YES |
| 9 | top1_prob | +0.4125 | +0.4459 | 0.4292 | YES |
| 10 | top1_minus_top2 | +0.3996 | +0.4518 | 0.4257 | YES |
| 11 | target_varentropy | -0.4083 | -0.3345 | 0.3714 | YES |
| 12 | draft_entropy | -0.2685 | -0.3027 | 0.2856 | YES |
| 13 | target_hidden_entropy | -0.1816 | -0.3492 | 0.2654 | YES |
| 14 | hidden_cosine_sim | -0.2030 | -0.2448 | 0.2239 | YES |
| 15 | kl_approx_target_draft | -0.0411 | -0.2663 | 0.1537 | YES |
| 16 | hidden_projection_score | +0.1176 | +0.0803 | 0.0990 | YES |
| 17 | hidden_norm | -0.1595 | +0.0328 | 0.0962 | NO |
| 18 | target_hidden_cosine_sim | +0.0849 | +0.0764 | 0.0807 | YES |
| 19 | hidden_entropy | +0.0707 | +0.0042 | 0.0374 | YES |
| 20 | target_projection_score | +0.0677 | -0.0020 | 0.0348 | NO |

## Section 3: Failure Pattern Analysis

### Llama-8B (2021 steps)

**Accept length distribution:**

| accept_length | Count | Pct |
|---------------|-------|-----|
| 0 | 456 | 22.6% |
| 1 | 378 | 18.7% |
| 2 | 327 | 16.2% |
| 3 | 596 | 29.5% |
| 4 | 264 | 13.1% |

**Wrong aggressive** (conf>0.55, accept=0): 90 steps (4.5%)

Mean signals in wrong-aggressive steps vs overall:

| Signal | Wrong-Agg Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.3673 | 0.4083 | -0.0409 |
| joint_confidence_product | 0.4974 | 0.5353 | -0.0379 |
| top1_prob | 0.6065 | 0.7352 | -0.1287 |
| target_draft_agree | 0.8326 | 0.6504 | +0.1822 |
| target_entropy | 0.3487 | 0.6013 | -0.2526 |
| kl_approx_target_draft | 2.0722 | 3.0076 | -0.9355 |
| rolling_accept_rate | 0.6217 | 0.5470 | +0.0747 |
| draft_entropy | 1.1153 | 1.6034 | -0.4881 |

**Wrong conservative** (conf<0.4, accept≥3): 17 steps (0.8%)

| Signal | Wrong-Con Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.3056 | 0.4083 | -0.1027 |
| joint_confidence_product | 0.4028 | 0.5353 | -0.1326 |
| top1_prob | 0.7254 | 0.7352 | -0.0098 |
| target_draft_agree | 0.3745 | 0.6504 | -0.2759 |
| target_entropy | 0.9845 | 0.6013 | +0.3832 |
| rolling_accept_rate | 0.4360 | 0.5470 | -0.1110 |

**Signature comparison: total failure (al=0) vs perfect (al≥4)**

| Signal | Fail Mean | Perfect Mean | Gap | Cohen's d |
|--------|-----------|-------------|-----|-----------|
| joint_confidence_product | 0.2949 | 0.8519 | +0.5571 | +3.445 |
| top1_minus_top2 | 0.2711 | 0.9284 | +0.6572 | +3.248 |
| draft_oracle_gate | 0.2288 | 0.6470 | +0.4181 | +3.015 |
| top1_prob | 0.4450 | 0.9523 | +0.5073 | +2.928 |
| target_oracle_gate | 0.3438 | 0.6074 | +0.2637 | +1.691 |
| target_top1_gap | 0.6478 | 0.8938 | +0.2460 | +1.569 |
| target_entropy | 0.7602 | 0.1984 | -0.5618 | -1.537 |
| joint_entropy_gate | 0.2948 | 0.6060 | +0.3112 | +1.426 |
| target_varentropy | 1.1334 | 0.3601 | -0.7733 | -1.257 |
| rolling_accept_rate | 0.5113 | 0.6790 | +0.1677 | +1.146 |
| target_draft_agree | 0.5829 | 0.8488 | +0.2659 | +1.065 |
| draft_entropy | 1.9690 | 0.6611 | -1.3080 | -1.013 |
| target_hidden_entropy | 4.1508 | 4.0135 | -0.1373 | -0.644 |
| hidden_norm | 67.0384 | 61.1042 | -5.9342 | -0.560 |
| target_hidden_cosine_sim | 0.6039 | 0.6310 | +0.0271 | +0.365 |
| hidden_cosine_sim | 0.6797 | 0.6218 | -0.0580 | -0.359 |
| hidden_projection_score | 0.8447 | 2.3770 | +1.5323 | +0.305 |
| kl_approx_target_draft | 3.0311 | 2.2760 | -0.7551 | -0.229 |
| target_projection_score | 0.0092 | 0.0120 | +0.0028 | +0.177 |
| hidden_entropy | 4.6939 | 4.7128 | +0.0189 | +0.149 |

### DS+LlamaDraft (6863 steps)

**Accept length distribution:**

| accept_length | Count | Pct |
|---------------|-------|-----|
| 0 | 3552 | 51.8% |
| 1 | 1590 | 23.2% |
| 2 | 835 | 12.2% |
| 3 | 582 | 8.5% |
| 4 | 304 | 4.4% |

**Wrong aggressive** (conf>0.55, accept=0): 376 steps (5.5%)

Mean signals in wrong-aggressive steps vs overall:

| Signal | Wrong-Agg Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.2189 | 0.1646 | +0.0543 |
| joint_confidence_product | 0.4657 | 0.3660 | +0.0997 |
| top1_prob | 0.5978 | 0.5443 | +0.0535 |
| target_draft_agree | 0.7246 | 0.3867 | +0.3379 |
| target_entropy | 0.3865 | 0.8105 | -0.4240 |
| kl_approx_target_draft | 2.3257 | 5.8657 | -3.5400 |
| rolling_accept_rate | 0.3856 | 0.2822 | +0.1034 |
| draft_entropy | 1.7208 | 2.9052 | -1.1845 |

**Wrong conservative** (conf<0.4, accept≥3): 24 steps (0.3%)

| Signal | Wrong-Con Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.1599 | 0.1646 | -0.0047 |
| joint_confidence_product | 0.3486 | 0.3660 | -0.0175 |
| top1_prob | 0.7108 | 0.5443 | +0.1665 |
| target_draft_agree | 0.2576 | 0.3867 | -0.1290 |
| target_entropy | 1.1669 | 0.8105 | +0.3564 |
| rolling_accept_rate | 0.2613 | 0.2822 | -0.0209 |

**Signature comparison: total failure (al=0) vs perfect (al≥4)**

| Signal | Fail Mean | Perfect Mean | Gap | Cohen's d |
|--------|-----------|-------------|-----|-----------|
| joint_confidence_product | 0.2416 | 0.8949 | +0.6533 | +4.217 |
| draft_oracle_gate | 0.0976 | 0.5902 | +0.4926 | +3.400 |
| top1_minus_top2 | 0.2560 | 0.9290 | +0.6730 | +3.265 |
| top1_prob | 0.3926 | 0.9488 | +0.5563 | +3.030 |
| target_oracle_gate | 0.1602 | 0.5888 | +0.4286 | +2.715 |
| rolling_accept_rate | 0.2502 | 0.6193 | +0.3692 | +2.286 |
| target_top1_gap | 0.6127 | 0.9416 | +0.3290 | +2.177 |
| joint_entropy_gate | 0.1845 | 0.6305 | +0.4460 | +2.122 |
| target_entropy | 0.9101 | 0.1206 | -0.7895 | -1.896 |
| target_draft_agree | 0.3342 | 0.8075 | +0.4732 | +1.852 |
| draft_entropy | 3.1784 | 0.8060 | -2.3724 | -1.560 |
| target_varentropy | 1.8034 | 0.2797 | -1.5237 | -1.417 |
| target_hidden_entropy | 4.6870 | 4.4869 | -0.2000 | -1.269 |
| kl_approx_target_draft | 6.3044 | 3.0544 | -3.2500 | -0.830 |
| hidden_cosine_sim | 0.7927 | 0.6872 | -0.1055 | -0.738 |
| target_hidden_cosine_sim | 0.4947 | 0.5246 | +0.0299 | +0.372 |
| hidden_projection_score | 0.4056 | 1.9257 | +1.5202 | +0.269 |
| hidden_norm | 78.6639 | 80.5954 | +1.9315 | +0.157 |
| hidden_entropy | 4.7848 | 4.7988 | +0.0140 | +0.152 |
| target_projection_score | 0.0129 | 0.0141 | +0.0011 | +0.067 |

## Section 4: Signal Interaction Analysis

### Llama-8B: Highly correlated pairs (|r| > 0.7)

| Signal A | Signal B | r |
|----------|----------|---|
| top1_prob | top1_minus_top2 | +0.973 |
| target_entropy | target_top1_gap | -0.874 |
| target_entropy | target_varentropy | +0.866 |
| target_oracle_gate | rolling_accept_rate | +0.856 |
| top1_prob | joint_confidence_product | +0.831 |
| top1_minus_top2 | joint_confidence_product | +0.810 |
| target_top1_gap | target_oracle_gate | +0.790 |
| draft_oracle_gate | joint_confidence_product | +0.781 |
| draft_entropy | joint_entropy_gate | -0.780 |
| top1_prob | draft_oracle_gate | +0.773 |
| top1_minus_top2 | draft_oracle_gate | +0.756 |
| target_entropy | target_oracle_gate | -0.712 |

### Llama-8B: Most independent signals (max |r| with others < 0.3)

- `hidden_norm` (max |r| with others = 0.280)
- `hidden_cosine_sim` (max |r| with others = 0.266)
- `hidden_projection_score` (max |r| with others = 0.143)
- `hidden_entropy` (max |r| with others = 0.244)
- `target_projection_score` (max |r| with others = 0.149)

### DS+LlamaDraft: Highly correlated pairs (|r| > 0.7)

| Signal A | Signal B | r |
|----------|----------|---|
| top1_prob | top1_minus_top2 | +0.968 |
| target_oracle_gate | rolling_accept_rate | +0.929 |
| top1_prob | joint_confidence_product | +0.869 |
| target_entropy | target_varentropy | +0.862 |
| top1_minus_top2 | joint_confidence_product | +0.852 |
| target_entropy | target_top1_gap | -0.817 |
| draft_oracle_gate | target_oracle_gate | +0.805 |
| draft_oracle_gate | rolling_accept_rate | +0.792 |
| draft_oracle_gate | joint_confidence_product | +0.763 |
| draft_entropy | joint_entropy_gate | -0.749 |

### DS+LlamaDraft: Most independent signals (max |r| with others < 0.3)

- `hidden_norm` (max |r| with others = 0.122)
- `hidden_projection_score` (max |r| with others = 0.130)
- `hidden_entropy` (max |r| with others = 0.105)
- `target_projection_score` (max |r| with others = 0.074)

## Section 5: Top-5 Signal Selection for Cross-Validation

**Selected 5 signals** (greedy: highest |r(accept_length)| with \|inter-r\| < 0.7):

1. **`joint_confidence_product`** — Joint, consensus |r|=0.6023
2. **`target_oracle_gate`** — Joint, consensus |r|=0.3874
3. **`joint_entropy_gate`** — Joint, consensus |r|=0.3474
4. **`target_varentropy`** — Target, consensus |r|=0.2596
5. **`target_draft_agree`** — Agreement, consensus |r|=0.2372

### Inter-correlation matrix of selected 5:

| Signal | joint_confid | target_oracl | joint_entrop | target_varen | target_draft |
|--------|------|------|------|------|------|
| joint_confid | +1.000 | +0.574 | +0.480 | -0.477 | +0.316 |
| target_oracl | +0.574 | +1.000 | +0.524 | -0.528 | +0.578 |
| joint_entrop | +0.480 | +0.524 | +1.000 | -0.575 | +0.435 |
| target_varen | -0.477 | -0.528 | -0.575 | +1.000 | -0.386 |
| target_draft | +0.316 | +0.578 | +0.435 | -0.386 | +1.000 |

## Section 6: Recommendations

### Final signal ranking by consensus |r(accept_length)|

| Rank | Signal | Consensus \|r\| | Tier |
|------|--------|----------------|------|
| 1 | joint_confidence_product | 0.6023 | **KEEP (high)** |
| 2 | draft_oracle_gate | 0.5728 | **KEEP (high)** |
| 3 | top1_prob | 0.5542 | **KEEP (high)** |
| 4 | top1_minus_top2 | 0.5525 | **KEEP (high)** |
| 5 | target_oracle_gate | 0.3874 | KEEP (medium) |
| 6 | joint_entropy_gate | 0.3474 | KEEP (medium) |
| 7 | target_top1_gap | 0.3219 | KEEP (medium) |
| 8 | target_entropy | 0.3167 | KEEP (medium) |
| 9 | rolling_accept_rate | 0.3024 | KEEP (medium) |
| 10 | target_varentropy | 0.2596 | KEEP (medium) |
| 11 | target_draft_agree | 0.2372 | Maybe |
| 12 | draft_entropy | 0.2354 | Maybe |
| 13 | target_hidden_entropy | 0.1941 | Maybe |
| 14 | hidden_cosine_sim | 0.1239 | Maybe |
| 15 | hidden_norm | 0.0864 | DROP |
| 16 | kl_approx_target_draft | 0.0741 | DROP |
| 17 | target_hidden_cosine_sim | 0.0621 | DROP |
| 18 | hidden_projection_score | 0.0579 | DROP |
| 19 | hidden_entropy | 0.0389 | DROP |
| 20 | target_projection_score | 0.0158 | DROP |

### Summary

- **KEEP (high, |r|>0.4)**: `joint_confidence_product`, `draft_oracle_gate`, `top1_prob`, `top1_minus_top2`
- **KEEP (medium, |r|>0.25)**: `target_oracle_gate`, `joint_entropy_gate`, `target_top1_gap`, `target_entropy`, `rolling_accept_rate`, `target_varentropy`
- **Maybe (|r|>0.1)**: `target_draft_agree`, `draft_entropy`, `target_hidden_entropy`, `hidden_cosine_sim`
- **DROP (|r|≤0.1)**: `hidden_norm`, `kl_approx_target_draft`, `target_hidden_cosine_sim`, `hidden_projection_score`, `hidden_entropy`, `target_projection_score`
