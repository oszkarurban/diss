# 26-Signal Deep Analysis for Dynamic Speculative Decoding

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
| Llama-8B | 2027 | 1.5546 | 1.4928 | 0.0000 | 8.0946 |
| DS+LlamaDraft | 7895 | 2.9273 | 1.9735 | 0.0000 | 8.5941 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.2328 | -0.2251 | -0.2782 | -0.4641 |
| DS+LlamaDraft | -0.2225 | -0.2313 | -0.2694 | -0.4513 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 1.8616 | 1.9321 | 1.6963 | 1.4839 | 0.7316 |
| DS+LlamaDraft | 3.1537 | 3.1554 | 2.9005 | 2.3877 | 1.2301 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.454 | Weak separator |
| DS+LlamaDraft | -0.683 | Moderate separator |

---

### 1.2 `top1_prob` — Draft

- **Formula**: `topk_p[:, 0].mean()`
- **Collected from**: Extend phase
- **High value means**: Draft strongly favors one token
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.7463 | 0.2722 | 0.0394 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.5620 | 0.3056 | 0.0070 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5928 | +0.1186 | +0.4160 | +0.6551 |
| DS+LlamaDraft | +0.5833 | +0.2856 | +0.4847 | +0.6838 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.4467 | 0.7678 | 0.7723 | 0.8384 | 0.9587 |
| DS+LlamaDraft | 0.3900 | 0.6621 | 0.6995 | 0.7921 | 0.9635 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +2.130 | **Strong separator** |
| DS+LlamaDraft | +2.162 | **Strong separator** |

---

### 1.3 `top1_minus_top2` — Draft

- **Formula**: `(topk_p[:,0] - topk_p[:,1]).mean()`
- **Collected from**: Extend phase
- **High value means**: Large gap between top-2 draft predictions
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.6514 | 0.3542 | 0.0000 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.4536 | 0.3575 | 0.0000 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5862 | +0.1164 | +0.4061 | +0.6386 |
| DS+LlamaDraft | +0.5884 | +0.2994 | +0.4913 | +0.6711 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2726 | 0.6720 | 0.6838 | 0.7612 | 0.9380 |
| DS+LlamaDraft | 0.2557 | 0.5612 | 0.6037 | 0.7228 | 0.9495 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +2.157 | **Strong separator** |
| DS+LlamaDraft | +2.182 | **Strong separator** |

---

### 1.4 `hidden_norm` — Draft hidden

- **Formula**: `torch.norm(hidden_states, dim=-1).mean()`
- **Collected from**: Extend phase
- **High value means**: High activation magnitude in draft representation
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 63.0348 | 10.3357 | 33.7500 | 109.0000 |
| DS+LlamaDraft | 7895 | 77.5964 | 11.3809 | 38.7500 | 142.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.1798 | -0.0483 | -0.1671 | -0.2060 |
| DS+LlamaDraft | -0.0467 | -0.0517 | -0.0282 | +0.0310 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 66.2192 | 62.5908 | 63.3946 | 62.4230 | 59.7403 |
| DS+LlamaDraft | 78.4686 | 76.5674 | 76.1153 | 77.3846 | 77.5033 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.469 | Weak separator |
| DS+LlamaDraft | -0.090 | Not useful |

---

### 1.5 `hidden_cosine_sim` — Draft hidden

- **Formula**: `cosine_sim(h[step_i], h[step_i-1])`
- **Collected from**: Draft phase
- **High value means**: Draft representation stable between consecutive steps
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.6418 | 0.1654 | 0.0000 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.7767 | 0.1680 | 0.0000 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.0699 | -0.2735 | -0.1848 | -0.1028 |
| DS+LlamaDraft | -0.1929 | -0.3602 | -0.3003 | -0.2398 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6567 | 0.6427 | 0.6521 | 0.6380 | 0.6164 |
| DS+LlamaDraft | 0.7980 | 0.7816 | 0.7722 | 0.7329 | 0.6671 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.164 | Not useful |
| DS+LlamaDraft | -0.582 | Moderate separator |

---

### 1.6 `hidden_projection_score` — Draft hidden

- **Formula**: `h · W_lm[predicted_token] / ||W||`
- **Collected from**: Draft phase
- **High value means**: Draft hidden strongly commits to its top prediction
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 1.4234 | 4.7241 | -11.3125 | 22.5625 |
| DS+LlamaDraft | 7895 | 0.8672 | 5.1112 | -20.5000 | 30.0625 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0629 | +0.0905 | +0.0915 | +0.1638 |
| DS+LlamaDraft | +0.1086 | +0.1616 | +0.1540 | +0.1650 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 1.1850 | 1.1503 | 1.2456 | 1.3844 | 2.2688 |
| DS+LlamaDraft | 0.4866 | 0.8938 | 0.9242 | 1.2334 | 3.0499 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.112 | Not useful |
| DS+LlamaDraft | +0.286 | Weak separator |

---

### 1.7 `hidden_entropy` — Draft hidden

- **Formula**: `entropy(h²/sum(h²))`
- **Collected from**: Draft phase
- **High value means**: Draft activation energy spread across many dimensions
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 4.7048 | 0.1625 | 0.0000 | 4.8594 |
| DS+LlamaDraft | 7895 | 4.7864 | 0.1178 | 0.0000 | 4.8750 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0668 | +0.0971 | +0.1000 | +0.0608 |
| DS+LlamaDraft | -0.0026 | +0.0287 | +0.0158 | -0.0211 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 4.7000 | 4.6905 | 4.6849 | 4.7138 | 4.7301 |
| DS+LlamaDraft | 4.7864 | 4.7882 | 4.7856 | 4.7801 | 4.7911 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.159 | Not useful |
| DS+LlamaDraft | -0.009 | Not useful |

---

### 1.8 `target_entropy` — Target

- **Formula**: `-sum(p log p) of target softmax`
- **Collected from**: Verify phase
- **High value means**: Target model uncertain — harder to match
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.5830 | 0.4496 | 0.0000 | 3.0278 |
| DS+LlamaDraft | 7895 | 0.7987 | 0.5882 | 0.0000 | 4.3172 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.3170 | -0.4123 | -0.4762 | -0.7179 |
| DS+LlamaDraft | -0.3350 | -0.4155 | -0.4446 | -0.6691 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.7018 | 0.7048 | 0.7108 | 0.5539 | 0.2192 |
| DS+LlamaDraft | 0.9248 | 0.8378 | 0.7674 | 0.5318 | 0.1313 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.651 | Moderate separator |
| DS+LlamaDraft | -1.138 | **Strong separator** |

---

### 1.9 `target_top1_prob` — Target

- **Formula**: `topk(target_probs, 1).values.mean()`
- **Collected from**: Verify phase
- **High value means**: Target strongly favors one token
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.8187 | 0.1317 | 0.3167 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.7721 | 0.1474 | 0.2104 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.3102 | +0.4189 | +0.4708 | +0.7240 |
| DS+LlamaDraft | +0.3617 | +0.4596 | +0.4849 | +0.6725 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.7849 | 0.7878 | 0.7820 | 0.8194 | 0.9318 |
| DS+LlamaDraft | 0.7391 | 0.7600 | 0.7767 | 0.8402 | 0.9615 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.612 | Moderate separator |
| DS+LlamaDraft | +1.195 | **Strong separator** |

---

### 1.10 `target_top1_gap` — Target

- **Formula**: `target top-1 minus top-2 probability`
- **Collected from**: Verify phase
- **High value means**: Target has one clear winner — easy to match
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.7214 | 0.1903 | 0.0686 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.6585 | 0.2079 | 0.0458 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.2997 | +0.4089 | +0.4566 | +0.7046 |
| DS+LlamaDraft | +0.3688 | +0.4713 | +0.4948 | +0.6527 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6734 | 0.6835 | 0.6700 | 0.7155 | 0.8867 |
| DS+LlamaDraft | 0.6118 | 0.6404 | 0.6630 | 0.7531 | 0.9372 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.580 | Moderate separator |
| DS+LlamaDraft | +1.196 | **Strong separator** |

---

### 1.11 `target_varentropy` — Target

- **Formula**: `Var(-log p) of target distribution`
- **Collected from**: Verify phase
- **High value means**: Target uncertainty is spread (uniform-ish) — harder
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.9006 | 0.7557 | 0.0003 | 6.4830 |
| DS+LlamaDraft | 7895 | 1.6045 | 1.4973 | 0.0004 | 12.2422 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.2885 | -0.3453 | -0.4251 | -0.6162 |
| DS+LlamaDraft | -0.2704 | -0.3219 | -0.3557 | -0.5786 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 1.0923 | 1.0957 | 1.0851 | 0.8208 | 0.3920 |
| DS+LlamaDraft | 1.8724 | 1.6745 | 1.5090 | 1.0236 | 0.3023 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.614 | Moderate separator |
| DS+LlamaDraft | -0.926 | **Strong separator** |

---

### 1.12 `target_hidden_norm` — Target hidden

- **Formula**: `torch.norm(target_hs, dim=-1).mean()`
- **Collected from**: Verify phase
- **High value means**: High activation magnitude in target representation
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 42.3220 | 2.5002 | 28.3750 | 48.0000 |
| DS+LlamaDraft | 7895 | 41.0801 | 2.6557 | 31.6250 | 51.7500 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.0264 | -0.0579 | -0.0751 | -0.0461 |
| DS+LlamaDraft | +0.1165 | +0.1213 | +0.1481 | +0.3230 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 42.2705 | 42.3461 | 42.6706 | 42.3739 | 41.9349 |
| DS+LlamaDraft | 40.8365 | 41.1368 | 41.2337 | 41.3506 | 42.0789 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.023 | Not useful |
| DS+LlamaDraft | +0.327 | Weak separator |

---

### 1.13 `target_hidden_cosine_sim` — Target hidden

- **Formula**: `cosine_sim between consecutive target positions`
- **Collected from**: Verify phase
- **High value means**: Target representation stable across draft positions
- **Polarity**: unclear (high = unclear confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.6050 | 0.0783 | 0.3223 | 0.8359 |
| DS+LlamaDraft | 7895 | 0.4966 | 0.0891 | 0.2031 | 0.8750 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0709 | +0.0274 | +0.0695 | +0.1543 |
| DS+LlamaDraft | +0.0897 | +0.0942 | +0.1153 | +0.2472 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6034 | 0.5923 | 0.6026 | 0.6074 | 0.6184 |
| DS+LlamaDraft | 0.4912 | 0.4974 | 0.4949 | 0.5022 | 0.5295 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.104 | Not useful |
| DS+LlamaDraft | +0.268 | Weak separator |

---

### 1.14 `target_projection_score` — Target hidden

- **Formula**: `h · W_target_lm[predicted] / ||W||`
- **Collected from**: Verify phase
- **High value means**: Target hidden commits to its prediction
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.0087 | 0.0154 | -0.0486 | 0.0752 |
| DS+LlamaDraft | 7895 | 0.0131 | 0.0175 | -0.0503 | 0.0928 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0463 | +0.0585 | +0.0522 | +0.1299 |
| DS+LlamaDraft | -0.0205 | -0.0182 | -0.0225 | +0.0675 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.0082 | 0.0078 | 0.0080 | 0.0093 | 0.0101 |
| DS+LlamaDraft | 0.0134 | 0.0129 | 0.0129 | 0.0121 | 0.0126 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.094 | Not useful |
| DS+LlamaDraft | -0.063 | Not useful |

---

### 1.15 `target_hidden_entropy` — Target hidden

- **Formula**: `entropy(h²/sum(h²)) of target hidden`
- **Collected from**: Verify phase
- **High value means**: Target activation energy spread
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 4.1192 | 0.2181 | 3.2188 | 4.8125 |
| DS+LlamaDraft | 7895 | 4.6545 | 0.1548 | 3.6719 | 4.8750 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | -0.1317 | -0.1096 | -0.1801 | -0.2784 |
| DS+LlamaDraft | -0.3451 | -0.3951 | -0.4494 | -0.4817 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 4.1471 | 4.1258 | 4.1569 | 4.1183 | 4.0382 |
| DS+LlamaDraft | 4.6870 | 4.6669 | 4.6497 | 4.5955 | 4.4551 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | -0.263 | Weak separator |
| DS+LlamaDraft | -0.961 | **Strong separator** |

---

### 1.16 `kl_approx_target_draft` — Divergence

- **Formula**: `F.cross_entropy(target_logits, draft_tokens)`
- **Collected from**: Verify phase
- **High value means**: Target assigns low probability to draft's choices
- **Polarity**: inverted (high = less confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 3.4563 | 3.8172 | 0.0000 | 25.3203 |
| DS+LlamaDraft | 7895 | 6.0878 | 4.5394 | 0.0000 | 24.5413 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.0007 | +0.0533 | +0.0514 | -0.1376 |
| DS+LlamaDraft | -0.1412 | -0.2483 | -0.2459 | -0.4560 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 3.3157 | 3.5862 | 3.3409 | 3.8104 | 3.0376 |
| DS+LlamaDraft | 6.4897 | 6.2195 | 6.0043 | 5.3023 | 3.8210 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.054 | Not useful |
| DS+LlamaDraft | -0.429 | Weak separator |

---

### 1.17 `target_draft_agree` — Agreement

- **Formula**: `fraction argmax(target)==draft_token`
- **Collected from**: Verify phase
- **High value means**: Target and draft agree on token choices
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.6380 | 0.2762 | 0.0000 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.3907 | 0.3246 | 0.0000 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.1713 | +0.4876 | +0.3998 | +0.5577 |
| DS+LlamaDraft | +0.3081 | +0.5909 | +0.5389 | +0.6712 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6068 | 0.6046 | 0.5906 | 0.6146 | 0.8015 |
| DS+LlamaDraft | 0.3319 | 0.3618 | 0.4032 | 0.4854 | 0.7799 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.287 | Weak separator |
| DS+LlamaDraft | +0.929 | **Strong separator** |

---

### 1.18 `joint_entropy_gate` — Joint

- **Formula**: `(1/(1+draft_ent)) * (1/(1+target_ent))`
- **Collected from**: Derived
- **High value means**: Both models are certain
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.3757 | 0.2420 | 0.0326 | 0.9982 |
| DS+LlamaDraft | 7895 | 0.2287 | 0.1952 | 0.0233 | 0.9996 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.3015 | +0.2918 | +0.3868 | +0.6125 |
| DS+LlamaDraft | +0.3985 | +0.4349 | +0.4786 | +0.6201 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.3162 | 0.3084 | 0.3367 | 0.3638 | 0.5840 |
| DS+LlamaDraft | 0.1869 | 0.1988 | 0.2294 | 0.2951 | 0.5530 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.554 | Moderate separator |
| DS+LlamaDraft | +0.996 | **Strong separator** |

---

### 1.19 `draft_oracle_gate` — Joint

- **Formula**: `top1_prob * rolling_accept_rate`
- **Collected from**: Derived
- **High value means**: Draft confident AND target has been accepting
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.4112 | 0.2038 | 0.0160 | 0.9001 |
| DS+LlamaDraft | 7895 | 0.1832 | 0.1930 | 0.0009 | 0.9081 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5631 | +0.6728 | +0.6253 | +0.7786 |
| DS+LlamaDraft | +0.6566 | +0.8339 | +0.7927 | +0.7560 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2347 | 0.3971 | 0.3980 | 0.4463 | 0.6241 |
| DS+LlamaDraft | 0.0962 | 0.1767 | 0.2172 | 0.3092 | 0.6295 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +1.712 | **Strong separator** |
| DS+LlamaDraft | +1.847 | **Strong separator** |

---

### 1.20 `target_oracle_gate` — Joint

- **Formula**: `target_top1_gap * rolling_accept_rate`
- **Collected from**: Derived
- **High value means**: Target confident AND has been accepting
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.4049 | 0.1818 | 0.0207 | 0.8631 |
| DS+LlamaDraft | 7895 | 0.2138 | 0.1880 | 0.0004 | 0.9062 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.3041 | +0.8516 | +0.6083 | +0.7107 |
| DS+LlamaDraft | +0.5180 | +0.9437 | +0.7741 | +0.6953 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.3624 | 0.3648 | 0.3585 | 0.3882 | 0.5782 |
| DS+LlamaDraft | 0.1609 | 0.1790 | 0.2092 | 0.3047 | 0.6158 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.556 | Moderate separator |
| DS+LlamaDraft | +1.399 | **Strong separator** |

---

### 1.21 `joint_confidence_product` — Joint

- **Formula**: `top1_prob * target_top1_gap`
- **Collected from**: Derived
- **High value means**: Both models confident in their top prediction
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.5504 | 0.2653 | 0.0078 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.3864 | 0.2721 | 0.0031 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.6055 | +0.3042 | +0.5527 | +0.8567 |
| DS+LlamaDraft | +0.6564 | +0.4775 | +0.6445 | +0.8368 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.3058 | 0.5292 | 0.5268 | 0.5996 | 0.8509 |
| DS+LlamaDraft | 0.2408 | 0.4279 | 0.4724 | 0.5985 | 0.9034 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +1.901 | **Strong separator** |
| DS+LlamaDraft | +2.295 | **Strong separator** |

---

### 1.22 `draft_oracle_gate_fixed` — Joint (fixed)

- **Formula**: `top1_prob * rolling_accept_rate`
- **Collected from**: Derived
- **High value means**: Draft confident AND historically accepted (same as original)
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.4112 | 0.2038 | 0.0160 | 0.9001 |
| DS+LlamaDraft | 7895 | 0.1832 | 0.1930 | 0.0009 | 0.9081 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.5631 | +0.6728 | +0.6253 | +0.7786 |
| DS+LlamaDraft | +0.6566 | +0.8339 | +0.7927 | +0.7560 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.2347 | 0.3971 | 0.3980 | 0.4463 | 0.6241 |
| DS+LlamaDraft | 0.0962 | 0.1767 | 0.2172 | 0.3092 | 0.6295 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +1.712 | **Strong separator** |
| DS+LlamaDraft | +1.847 | **Strong separator** |

---

### 1.23 `target_oracle_gate_fixed` — Joint (fixed)

- **Formula**: `target_top1_prob * rolling_accept_rate`
- **Collected from**: Derived
- **High value means**: Target confident (raw prob) AND historically accepted
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.4543 | 0.1708 | 0.0562 | 0.8788 |
| DS+LlamaDraft | 7895 | 0.2417 | 0.1878 | 0.0031 | 0.9074 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.2876 | +0.9268 | +0.6134 | +0.6789 |
| DS+LlamaDraft | +0.5029 | +0.9740 | +0.7679 | +0.6802 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.4183 | 0.4151 | 0.4119 | 0.4415 | 0.6071 |
| DS+LlamaDraft | 0.1903 | 0.2073 | 0.2384 | 0.3315 | 0.6293 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.521 | Moderate separator |
| DS+LlamaDraft | +1.376 | **Strong separator** |

---

### 1.24 `joint_confidence_product_fixed` — Joint (fixed)

- **Formula**: `top1_prob * target_top1_prob`
- **Collected from**: Derived
- **High value means**: Both models confident (symmetric raw probs)
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.6196 | 0.2606 | 0.0246 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.4456 | 0.2770 | 0.0039 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.6272 | +0.2564 | +0.5319 | +0.8265 |
| DS+LlamaDraft | +0.6484 | +0.4214 | +0.6051 | +0.8147 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.3547 | 0.6080 | 0.6106 | 0.6867 | 0.8938 |
| DS+LlamaDraft | 0.2904 | 0.5061 | 0.5494 | 0.6662 | 0.9267 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +2.090 | **Strong separator** |
| DS+LlamaDraft | +2.364 | **Strong separator** |

---

### 1.25 `confidence_agreement` — Joint (2x2)

- **Formula**: `1 - abs(top1_prob - target_top1_prob)`
- **Collected from**: Derived
- **High value means**: Both models at same confidence level (both certain OR both uncertain)
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.7860 | 0.1834 | 0.0959 | 1.0000 |
| DS+LlamaDraft | 7895 | 0.7042 | 0.2200 | 0.0151 | 1.0000 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.4809 | +0.0930 | +0.3222 | +0.5384 |
| DS+LlamaDraft | +0.4234 | +0.1619 | +0.3235 | +0.4573 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.6350 | 0.7888 | 0.7937 | 0.8177 | 0.9312 |
| DS+LlamaDraft | 0.6179 | 0.7514 | 0.7692 | 0.8013 | 0.9474 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +1.274 | **Strong separator** |
| DS+LlamaDraft | +1.293 | **Strong separator** |

---

### 1.26 `rolling_accept_rate` — Historical

- **Formula**: `alpha*step_rate + (1-alpha)*prev, alpha=0.3`
- **Collected from**: Verify EMA
- **High value means**: Recent acceptance has been high
- **Polarity**: direct (high = more confident)

| Dataset | N | Mean | Std | Min | Max |
|---------|---|------|-----|-----|-----|
| Llama-8B | 2027 | 0.5442 | 0.1592 | 0.0947 | 0.9265 |
| DS+LlamaDraft | 7895 | 0.2962 | 0.1919 | 0.0035 | 0.9098 |

| Dataset | r(accept_length) | r(RAR) | r(windowed_AL_3) | r(confidence) |
|---------|------------------|--------|------------------|---------------|
| Llama-8B | +0.2063 | +1.0000 | +0.5444 | +0.5070 |
| DS+LlamaDraft | +0.4537 | +1.0000 | +0.7261 | +0.6025 |

| Dataset | al=0 | al=1 | al=2 | al=3 | al=4+ |
|---------|------|------|------|------|-------|
| Llama-8B | 0.5232 | 0.5147 | 0.5137 | 0.5336 | 0.6508 |
| DS+LlamaDraft | 0.2491 | 0.2628 | 0.2951 | 0.3810 | 0.6511 |

| Dataset | Cohen's d (al=0 vs al≥3) | Verdict |
|---------|--------------------------|---------|
| Llama-8B | +0.348 | Weak separator |
| DS+LlamaDraft | +1.264 | **Strong separator** |

---

## Section 2: Correlation Rankings

### Ranked by `|r(signal, accept_length)|`

| Rank | Signal | r (Llama-8B) | r (DS+LlamaDraft) | Avg `\|r\|` | Consistent? |
|------|--------|------|------|---------|-------------|
| 1 | joint_confidence_product_fixed | +0.6272 | +0.6484 | 0.6378 | YES |
| 2 | joint_confidence_product | +0.6055 | +0.6564 | 0.6309 | YES |
| 3 | draft_oracle_gate_fixed | +0.5631 | +0.6566 | 0.6099 | YES |
| 4 | draft_oracle_gate | +0.5631 | +0.6566 | 0.6099 | YES |
| 5 | top1_prob | +0.5928 | +0.5833 | 0.5880 | YES |
| 6 | top1_minus_top2 | +0.5862 | +0.5884 | 0.5873 | YES |
| 7 | confidence_agreement | +0.4809 | +0.4234 | 0.4522 | YES |
| 8 | target_oracle_gate | +0.3041 | +0.5180 | 0.4110 | YES |
| 9 | target_oracle_gate_fixed | +0.2876 | +0.5029 | 0.3952 | YES |
| 10 | joint_entropy_gate | +0.3015 | +0.3985 | 0.3500 | YES |
| 11 | target_top1_prob | +0.3102 | +0.3617 | 0.3360 | YES |
| 12 | target_top1_gap | +0.2997 | +0.3688 | 0.3343 | YES |
| 13 | rolling_accept_rate | +0.2063 | +0.4537 | 0.3300 | YES |
| 14 | target_entropy | -0.3170 | -0.3350 | 0.3260 | YES |
| 15 | target_varentropy | -0.2885 | -0.2704 | 0.2795 | YES |
| 16 | target_draft_agree | +0.1713 | +0.3081 | 0.2397 | YES |
| 17 | target_hidden_entropy | -0.1317 | -0.3451 | 0.2384 | YES |
| 18 | draft_entropy | -0.2328 | -0.2225 | 0.2276 | YES |
| 19 | hidden_cosine_sim | -0.0699 | -0.1929 | 0.1314 | YES |
| 20 | hidden_norm | -0.1798 | -0.0467 | 0.1133 | YES |
| 21 | hidden_projection_score | +0.0629 | +0.1086 | 0.0857 | YES |
| 22 | target_hidden_cosine_sim | +0.0709 | +0.0897 | 0.0803 | YES |
| 23 | target_hidden_norm | -0.0264 | +0.1165 | 0.0714 | NO |
| 24 | kl_approx_target_draft | +0.0007 | -0.1412 | 0.0709 | NO |
| 25 | hidden_entropy | +0.0668 | -0.0026 | 0.0347 | NO |
| 26 | target_projection_score | +0.0463 | -0.0205 | 0.0334 | NO |

### Ranked by `|r(signal, rolling_accept_rate)|`

| Rank | Signal | r (Llama-8B) | r (DS+LlamaDraft) | Avg `\|r\|` | Consistent? |
|------|--------|------|------|---------|-------------|
| 1 | rolling_accept_rate | +1.0000 | +1.0000 | 1.0000 | YES |
| 2 | target_oracle_gate_fixed | +0.9268 | +0.9740 | 0.9504 | YES |
| 3 | target_oracle_gate | +0.8516 | +0.9437 | 0.8977 | YES |
| 4 | draft_oracle_gate_fixed | +0.6728 | +0.8339 | 0.7533 | YES |
| 5 | draft_oracle_gate | +0.6728 | +0.8339 | 0.7533 | YES |
| 6 | target_draft_agree | +0.4876 | +0.5909 | 0.5392 | YES |
| 7 | target_top1_gap | +0.4089 | +0.4713 | 0.4401 | YES |
| 8 | target_top1_prob | +0.4189 | +0.4596 | 0.4393 | YES |
| 9 | target_entropy | -0.4123 | -0.4155 | 0.4139 | YES |
| 10 | joint_confidence_product | +0.3042 | +0.4775 | 0.3909 | YES |
| 11 | joint_entropy_gate | +0.2918 | +0.4349 | 0.3633 | YES |
| 12 | joint_confidence_product_fixed | +0.2564 | +0.4214 | 0.3389 | YES |
| 13 | target_varentropy | -0.3453 | -0.3219 | 0.3336 | YES |
| 14 | hidden_cosine_sim | -0.2735 | -0.3602 | 0.3169 | YES |
| 15 | target_hidden_entropy | -0.1096 | -0.3951 | 0.2523 | YES |
| 16 | draft_entropy | -0.2251 | -0.2313 | 0.2282 | YES |
| 17 | top1_minus_top2 | +0.1164 | +0.2994 | 0.2079 | YES |
| 18 | top1_prob | +0.1186 | +0.2856 | 0.2021 | YES |
| 19 | kl_approx_target_draft | +0.0533 | -0.2483 | 0.1508 | NO |
| 20 | confidence_agreement | +0.0930 | +0.1619 | 0.1275 | YES |
| 21 | hidden_projection_score | +0.0905 | +0.1616 | 0.1261 | YES |
| 22 | target_hidden_norm | -0.0579 | +0.1213 | 0.0896 | NO |
| 23 | hidden_entropy | +0.0971 | +0.0287 | 0.0629 | YES |
| 24 | target_hidden_cosine_sim | +0.0274 | +0.0942 | 0.0608 | YES |
| 25 | hidden_norm | -0.0483 | -0.0517 | 0.0500 | YES |
| 26 | target_projection_score | +0.0585 | -0.0182 | 0.0383 | NO |

### Ranked by `|r(signal, windowed_AL_3)|`

| Rank | Signal | r (Llama-8B) | r (DS+LlamaDraft) | Avg `\|r\|` | Consistent? |
|------|--------|------|------|---------|-------------|
| 1 | draft_oracle_gate_fixed | +0.6253 | +0.7927 | 0.7090 | YES |
| 2 | draft_oracle_gate | +0.6253 | +0.7927 | 0.7090 | YES |
| 3 | target_oracle_gate | +0.6083 | +0.7741 | 0.6912 | YES |
| 4 | target_oracle_gate_fixed | +0.6134 | +0.7679 | 0.6907 | YES |
| 5 | rolling_accept_rate | +0.5444 | +0.7261 | 0.6352 | YES |
| 6 | joint_confidence_product | +0.5527 | +0.6445 | 0.5986 | YES |
| 7 | joint_confidence_product_fixed | +0.5319 | +0.6051 | 0.5685 | YES |
| 8 | target_top1_prob | +0.4708 | +0.4849 | 0.4779 | YES |
| 9 | target_top1_gap | +0.4566 | +0.4948 | 0.4757 | YES |
| 10 | target_draft_agree | +0.3998 | +0.5389 | 0.4694 | YES |
| 11 | target_entropy | -0.4762 | -0.4446 | 0.4604 | YES |
| 12 | top1_prob | +0.4160 | +0.4847 | 0.4503 | YES |
| 13 | top1_minus_top2 | +0.4061 | +0.4913 | 0.4487 | YES |
| 14 | joint_entropy_gate | +0.3868 | +0.4786 | 0.4327 | YES |
| 15 | target_varentropy | -0.4251 | -0.3557 | 0.3904 | YES |
| 16 | confidence_agreement | +0.3222 | +0.3235 | 0.3228 | YES |
| 17 | target_hidden_entropy | -0.1801 | -0.4494 | 0.3147 | YES |
| 18 | draft_entropy | -0.2782 | -0.2694 | 0.2738 | YES |
| 19 | hidden_cosine_sim | -0.1848 | -0.3003 | 0.2426 | YES |
| 20 | kl_approx_target_draft | +0.0514 | -0.2459 | 0.1486 | NO |
| 21 | hidden_projection_score | +0.0915 | +0.1540 | 0.1227 | YES |
| 22 | target_hidden_norm | -0.0751 | +0.1481 | 0.1116 | NO |
| 23 | hidden_norm | -0.1671 | -0.0282 | 0.0977 | YES |
| 24 | target_hidden_cosine_sim | +0.0695 | +0.1153 | 0.0924 | YES |
| 25 | hidden_entropy | +0.1000 | +0.0158 | 0.0579 | YES |
| 26 | target_projection_score | +0.0522 | -0.0225 | 0.0374 | NO |

## Section 3: Failure Pattern Analysis

### Llama-8B (2027 steps)

**Accept length distribution:**

| accept_length | Count | Pct |
|---------------|-------|-----|
| 0 | 463 | 22.8% |
| 1 | 333 | 16.4% |
| 2 | 332 | 16.4% |
| 3 | 565 | 27.9% |
| 4 | 334 | 16.5% |

**Wrong aggressive** (conf>0.55, accept=0): 90 steps (4.4%)

Mean signals in wrong-aggressive steps vs overall:

| Signal | Wrong-Agg Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.4122 | 0.4112 | +0.0009 |
| joint_confidence_product | 0.5472 | 0.5504 | -0.0032 |
| top1_prob | 0.6724 | 0.7463 | -0.0739 |
| target_draft_agree | 0.8478 | 0.6380 | +0.2097 |
| target_entropy | 0.3439 | 0.5830 | -0.2391 |
| kl_approx_target_draft | 1.9260 | 3.4563 | -1.5303 |
| rolling_accept_rate | 0.6238 | 0.5442 | +0.0797 |
| draft_entropy | 1.1587 | 1.5546 | -0.3959 |

**Wrong conservative** (conf<0.4, accept≥3): 24 steps (1.2%)

| Signal | Wrong-Con Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.3008 | 0.4112 | -0.1105 |
| joint_confidence_product | 0.3825 | 0.5504 | -0.1679 |
| top1_prob | 0.6890 | 0.7463 | -0.0573 |
| target_draft_agree | 0.3806 | 0.6380 | -0.2575 |
| target_entropy | 0.8686 | 0.5830 | +0.2856 |
| rolling_accept_rate | 0.4519 | 0.5442 | -0.0923 |

**Signature comparison: total failure (al=0) vs perfect (al≥4)**

| Signal | Fail Mean | Perfect Mean | Gap | Cohen's d |
|--------|-----------|-------------|-----|-----------|
| joint_confidence_product_fixed | 0.3547 | 0.8938 | +0.5391 | +3.410 |
| joint_confidence_product | 0.3058 | 0.8509 | +0.5451 | +3.376 |
| top1_minus_top2 | 0.2726 | 0.9380 | +0.6654 | +3.286 |
| top1_prob | 0.4467 | 0.9587 | +0.5120 | +2.974 |
| draft_oracle_gate_fixed | 0.2347 | 0.6241 | +0.3894 | +2.873 |
| draft_oracle_gate | 0.2347 | 0.6241 | +0.3894 | +2.873 |
| confidence_agreement | 0.6350 | 0.9312 | +0.2962 | +1.868 |
| target_oracle_gate | 0.3624 | 0.5782 | +0.2158 | +1.426 |
| target_top1_prob | 0.7849 | 0.9318 | +0.1469 | +1.366 |
| target_top1_gap | 0.6734 | 0.8867 | +0.2133 | +1.361 |
| target_oracle_gate_fixed | 0.4183 | 0.6071 | +0.1889 | +1.324 |
| target_entropy | 0.7018 | 0.2192 | -0.4826 | -1.302 |
| joint_entropy_gate | 0.3162 | 0.5840 | +0.2678 | +1.158 |
| target_varentropy | 1.0923 | 0.3920 | -0.7003 | -1.100 |
| rolling_accept_rate | 0.5232 | 0.6508 | +0.1276 | +0.924 |
| draft_entropy | 1.8616 | 0.7316 | -1.1300 | -0.883 |
| target_draft_agree | 0.6068 | 0.8015 | +0.1947 | +0.800 |
| hidden_norm | 66.2192 | 59.7403 | -6.4790 | -0.638 |
| target_hidden_entropy | 4.1471 | 4.0382 | -0.1089 | -0.497 |
| hidden_cosine_sim | 0.6567 | 0.6164 | -0.0404 | -0.263 |
| hidden_entropy | 4.7000 | 4.7301 | +0.0302 | +0.251 |
| hidden_projection_score | 1.1850 | 2.2688 | +1.0838 | +0.229 |
| target_hidden_cosine_sim | 0.6034 | 0.6184 | +0.0150 | +0.203 |
| target_hidden_norm | 42.2705 | 41.9349 | -0.3356 | -0.135 |
| target_projection_score | 0.0082 | 0.0101 | +0.0019 | +0.127 |
| kl_approx_target_draft | 3.3157 | 3.0376 | -0.2780 | -0.077 |

### DS+LlamaDraft (7895 steps)

**Accept length distribution:**

| accept_length | Count | Pct |
|---------------|-------|-----|
| 0 | 3951 | 50.0% |
| 1 | 1743 | 22.1% |
| 2 | 984 | 12.5% |
| 3 | 691 | 8.8% |
| 4 | 522 | 6.6% |
| 5 | 4 | 0.1% |

**Wrong aggressive** (conf>0.55, accept=0): 397 steps (5.0%)

Mean signals in wrong-aggressive steps vs overall:

| Signal | Wrong-Agg Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.2375 | 0.1832 | +0.0543 |
| joint_confidence_product | 0.5081 | 0.3864 | +0.1216 |
| top1_prob | 0.6535 | 0.5620 | +0.0915 |
| target_draft_agree | 0.6850 | 0.3907 | +0.2943 |
| target_entropy | 0.4302 | 0.7987 | -0.3685 |
| kl_approx_target_draft | 2.9127 | 6.0878 | -3.1752 |
| rolling_accept_rate | 0.3826 | 0.2962 | +0.0865 |
| draft_entropy | 1.8741 | 2.9273 | -1.0532 |

**Wrong conservative** (conf<0.4, accept≥3): 31 steps (0.4%)

| Signal | Wrong-Con Mean | Overall Mean | Delta |
|--------|---------------|-------------|-------|
| draft_oracle_gate | 0.1539 | 0.1832 | -0.0293 |
| joint_confidence_product | 0.3124 | 0.3864 | -0.0740 |
| top1_prob | 0.6063 | 0.5620 | +0.0443 |
| target_draft_agree | 0.1699 | 0.3907 | -0.2208 |
| target_entropy | 1.0518 | 0.7987 | +0.2531 |
| rolling_accept_rate | 0.2768 | 0.2962 | -0.0193 |

**Signature comparison: total failure (al=0) vs perfect (al≥4)**

| Signal | Fail Mean | Perfect Mean | Gap | Cohen's d |
|--------|-----------|-------------|-----|-----------|
| joint_confidence_product | 0.2408 | 0.9034 | +0.6627 | +4.444 |
| joint_confidence_product_fixed | 0.2904 | 0.9267 | +0.6363 | +4.210 |
| draft_oracle_gate_fixed | 0.0962 | 0.6295 | +0.5333 | +3.870 |
| draft_oracle_gate | 0.0962 | 0.6295 | +0.5333 | +3.870 |
| top1_minus_top2 | 0.2557 | 0.9495 | +0.6938 | +3.528 |
| top1_prob | 0.3900 | 0.9635 | +0.5735 | +3.229 |
| target_oracle_gate | 0.1609 | 0.6158 | +0.4549 | +2.940 |
| target_oracle_gate_fixed | 0.1903 | 0.6293 | +0.4390 | +2.872 |
| rolling_accept_rate | 0.2491 | 0.6511 | +0.4021 | +2.568 |
| target_top1_gap | 0.6118 | 0.9372 | +0.3253 | +2.128 |
| target_top1_prob | 0.7391 | 0.9615 | +0.2224 | +2.071 |
| confidence_agreement | 0.6179 | 0.9474 | +0.3295 | +2.000 |
| target_entropy | 0.9248 | 0.1313 | -0.7935 | -1.818 |
| target_draft_agree | 0.3319 | 0.7799 | +0.4480 | +1.736 |
| joint_entropy_gate | 0.1869 | 0.5530 | +0.3661 | +1.652 |
| target_hidden_entropy | 4.6870 | 4.4551 | -0.2319 | -1.535 |
| target_varentropy | 1.8724 | 0.3023 | -1.5702 | -1.371 |
| draft_entropy | 3.1537 | 1.2301 | -1.9235 | -1.128 |
| hidden_cosine_sim | 0.7980 | 0.6671 | -0.1308 | -0.874 |
| kl_approx_target_draft | 6.4897 | 3.8210 | -2.6687 | -0.657 |
| target_hidden_norm | 40.8365 | 42.0789 | +1.2424 | +0.523 |
| target_hidden_cosine_sim | 0.4912 | 0.5295 | +0.0383 | +0.469 |
| hidden_projection_score | 0.4866 | 3.0499 | +2.5633 | +0.462 |
| hidden_norm | 78.4686 | 77.5033 | -0.9653 | -0.084 |
| target_projection_score | 0.0134 | 0.0126 | -0.0008 | -0.049 |
| hidden_entropy | 4.7864 | 4.7911 | +0.0048 | +0.028 |

## Section 4: Signal Interaction Analysis

### Llama-8B: Highly correlated pairs (|r| > 0.7)

| Signal A | Signal B | r |
|----------|----------|---|
| draft_oracle_gate | draft_oracle_gate_fixed | +1.000 |
| target_oracle_gate | target_oracle_gate_fixed | +0.982 |
| target_top1_prob | target_top1_gap | +0.980 |
| joint_confidence_product | joint_confidence_product_fixed | +0.979 |
| top1_prob | top1_minus_top2 | +0.973 |
| target_entropy | target_top1_prob | -0.948 |
| target_oracle_gate_fixed | rolling_accept_rate | +0.927 |
| top1_prob | joint_confidence_product_fixed | +0.926 |
| top1_minus_top2 | joint_confidence_product_fixed | +0.902 |
| target_entropy | target_varentropy | +0.879 |
| target_entropy | target_top1_gap | -0.878 |
| target_oracle_gate | rolling_accept_rate | +0.852 |
| top1_prob | joint_confidence_product | +0.840 |
| draft_oracle_gate_fixed | joint_confidence_product_fixed | +0.824 |
| draft_oracle_gate | joint_confidence_product_fixed | +0.824 |
| top1_minus_top2 | joint_confidence_product | +0.821 |
| target_top1_gap | target_oracle_gate | +0.801 |
| joint_confidence_product | draft_oracle_gate_fixed | +0.796 |
| draft_oracle_gate | joint_confidence_product | +0.796 |
| target_top1_prob | target_oracle_gate | +0.791 |
| top1_prob | draft_oracle_gate_fixed | +0.785 |
| top1_prob | draft_oracle_gate | +0.785 |
| draft_entropy | joint_entropy_gate | -0.781 |
| top1_minus_top2 | draft_oracle_gate_fixed | +0.768 |
| top1_minus_top2 | draft_oracle_gate | +0.768 |
| joint_confidence_product_fixed | confidence_agreement | +0.736 |
| target_entropy | target_oracle_gate | -0.728 |
| top1_prob | confidence_agreement | +0.720 |
| target_top1_prob | target_varentropy | -0.720 |
| target_top1_prob | target_oracle_gate_fixed | +0.710 |

### Llama-8B: Most independent signals (max |r| with others < 0.3)

- `hidden_cosine_sim` (max |r| with others = 0.274)
- `hidden_projection_score` (max |r| with others = 0.164)
- `hidden_entropy` (max |r| with others = 0.164)
- `target_projection_score` (max |r| with others = 0.116)

### DS+LlamaDraft: Highly correlated pairs (|r| > 0.7)

| Signal A | Signal B | r |
|----------|----------|---|
| draft_oracle_gate | draft_oracle_gate_fixed | +1.000 |
| target_oracle_gate | target_oracle_gate_fixed | +0.993 |
| joint_confidence_product | joint_confidence_product_fixed | +0.982 |
| target_top1_prob | target_top1_gap | +0.978 |
| target_oracle_gate_fixed | rolling_accept_rate | +0.974 |
| top1_prob | top1_minus_top2 | +0.972 |
| target_oracle_gate | rolling_accept_rate | +0.944 |
| top1_prob | joint_confidence_product_fixed | +0.944 |
| top1_minus_top2 | joint_confidence_product_fixed | +0.923 |
| target_entropy | target_top1_prob | -0.911 |
| top1_prob | joint_confidence_product | +0.874 |
| target_entropy | target_varentropy | +0.873 |
| top1_minus_top2 | joint_confidence_product | +0.860 |
| draft_oracle_gate_fixed | target_oracle_gate_fixed | +0.852 |
| draft_oracle_gate | target_oracle_gate_fixed | +0.852 |
| target_oracle_gate | draft_oracle_gate_fixed | +0.847 |
| draft_oracle_gate | target_oracle_gate | +0.847 |
| draft_oracle_gate_fixed | rolling_accept_rate | +0.834 |
| draft_oracle_gate | rolling_accept_rate | +0.834 |
| target_entropy | target_top1_gap | -0.825 |
| joint_confidence_product | draft_oracle_gate_fixed | +0.793 |
| draft_oracle_gate | joint_confidence_product | +0.793 |
| draft_oracle_gate_fixed | joint_confidence_product_fixed | +0.774 |
| draft_oracle_gate | joint_confidence_product_fixed | +0.774 |
| top1_prob | confidence_agreement | +0.763 |
| draft_entropy | joint_entropy_gate | -0.746 |
| joint_confidence_product_fixed | confidence_agreement | +0.710 |
| top1_minus_top2 | confidence_agreement | +0.708 |

### DS+LlamaDraft: Most independent signals (max |r| with others < 0.3)

- `hidden_norm` (max |r| with others = 0.161)
- `hidden_projection_score` (max |r| with others = 0.176)
- `hidden_entropy` (max |r| with others = 0.087)
- `target_projection_score` (max |r| with others = 0.071)

## Section 5: Top-5 Signal Selection for Cross-Validation

**Selected 5 signals** (greedy: highest |r(accept_length)| with `|inter-r|` < 0.7):

1. **`joint_confidence_product_fixed`** — Joint (fixed), consensus |r|=0.6378
2. **`target_oracle_gate`** — Joint, consensus |r|=0.4110
3. **`joint_entropy_gate`** — Joint, consensus |r|=0.3500
4. **`target_varentropy`** — Target, consensus |r|=0.2795
5. **`target_draft_agree`** — Agreement, consensus |r|=0.2397

### Inter-correlation matrix of selected 5:

| Signal | joint_confid | target_oracl | joint_entrop | target_varen | target_draft |
|--------|------|------|------|------|------|
| joint_confid | +1.000 | +0.466 | +0.407 | -0.445 | +0.235 |
| target_oracl | +0.466 | +1.000 | +0.537 | -0.555 | +0.522 |
| joint_entrop | +0.407 | +0.537 | +1.000 | -0.584 | +0.418 |
| target_varen | -0.445 | -0.555 | -0.584 | +1.000 | -0.392 |
| target_draft | +0.235 | +0.522 | +0.418 | -0.392 | +1.000 |

## Section 6: Recommendations

### Final signal ranking by consensus |r(accept_length)|

| Rank | Signal | Consensus `\|r\|` | Tier |
|------|--------|----------------|------|
| 1 | joint_confidence_product_fixed | 0.6378 | **KEEP (high)** |
| 2 | joint_confidence_product | 0.6309 | **KEEP (high)** |
| 3 | draft_oracle_gate | 0.6099 | **KEEP (high)** |
| 4 | draft_oracle_gate_fixed | 0.6099 | **KEEP (high)** |
| 5 | top1_prob | 0.5880 | **KEEP (high)** |
| 6 | top1_minus_top2 | 0.5873 | **KEEP (high)** |
| 7 | confidence_agreement | 0.4522 | **KEEP (high)** |
| 8 | target_oracle_gate | 0.4110 | **KEEP (high)** |
| 9 | target_oracle_gate_fixed | 0.3952 | KEEP (medium) |
| 10 | joint_entropy_gate | 0.3500 | KEEP (medium) |
| 11 | target_top1_prob | 0.3360 | KEEP (medium) |
| 12 | target_top1_gap | 0.3343 | KEEP (medium) |
| 13 | rolling_accept_rate | 0.3300 | KEEP (medium) |
| 14 | target_entropy | 0.3260 | KEEP (medium) |
| 15 | target_varentropy | 0.2795 | KEEP (medium) |
| 16 | target_draft_agree | 0.2397 | Maybe |
| 17 | target_hidden_entropy | 0.2384 | Maybe |
| 18 | draft_entropy | 0.2276 | Maybe |
| 19 | hidden_cosine_sim | 0.1314 | Maybe |
| 20 | hidden_norm | 0.1133 | Maybe |
| 21 | hidden_projection_score | 0.0857 | DROP |
| 22 | target_hidden_cosine_sim | 0.0803 | DROP |
| 23 | target_hidden_norm | 0.0714 | DROP |
| 24 | kl_approx_target_draft | 0.0709 | DROP |
| 25 | hidden_entropy | 0.0347 | DROP |
| 26 | target_projection_score | 0.0334 | DROP |

### Summary

- **KEEP (high, |r|>0.4)**: `joint_confidence_product_fixed`, `joint_confidence_product`, `draft_oracle_gate`, `draft_oracle_gate_fixed`, `top1_prob`, `top1_minus_top2`, `confidence_agreement`, `target_oracle_gate`
- **KEEP (medium, |r|>0.25)**: `target_oracle_gate_fixed`, `joint_entropy_gate`, `target_top1_prob`, `target_top1_gap`, `rolling_accept_rate`, `target_entropy`, `target_varentropy`
- **Maybe (|r|>0.1)**: `target_draft_agree`, `target_hidden_entropy`, `draft_entropy`, `hidden_cosine_sim`, `hidden_norm`
- **DROP (|r|≤0.1)**: `hidden_projection_score`, `target_hidden_cosine_sim`, `target_hidden_norm`, `kl_approx_target_draft`, `hidden_entropy`, `target_projection_score`

## Section 7: Per-Turn Trajectory Analysis — Where Accept Length Spikes and Falls

This section examines individual turns to find moments where acceptance
dramatically changes, and what signals look like at those transitions.

### Llama-8B

**Turn 12** (272 steps, mean accept_len=2.01):

Accept length trajectory: `2 4 4 4 1 3 4 3 2 3 3 3 4 1 0 0 2 3 0 2 3 3 0 0 0 0 1 3 3 0 2 3 3 0 1 2 3 0 0 1 0 1 1 0 3 2 2 0 0 0 1 2 0 2 3 1 1 3 1 0 ... (212 more)`

**Spikes** (accept_length jumps up by ≥2): 52 transitions

Step 0→1: accept_length 2→4

| Signal | Before (al=2) | After (al=4) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4973 | 0.6475 | +0.1502 |
| joint_confidence_product | 0.3498 | 0.9961 | +0.6462 |
| joint_confidence_product_fixed | 0.5793 | 0.9961 | +0.4168 |
| confidence_agreement | 0.5878 | 0.9961 | +0.4083 |
| target_draft_agree | 0.0000 | 1.0000 | +1.0000 |
| target_top1_prob | 0.5824 | 1.0000 | +0.4176 |
| top1_prob | 0.9946 | 0.9961 | +0.0015 |
| target_entropy | 1.2947 | 0.0001 | -1.2946 |
| rolling_accept_rate | 0.5000 | 0.6500 | +0.1500 |

Step 4→5: accept_length 1→3

| Signal | Before (al=1) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.7624 | 0.4587 | -0.3037 |
| joint_confidence_product | 0.8604 | 0.4776 | -0.3828 |
| joint_confidence_product_fixed | 0.9023 | 0.5626 | -0.3396 |
| confidence_agreement | 0.9235 | 0.9923 | +0.0688 |
| target_draft_agree | 0.6000 | 0.2500 | -0.3500 |
| target_top1_prob | 0.9124 | 0.7539 | -0.1585 |
| top1_prob | 0.9889 | 0.7462 | -0.2426 |
| target_entropy | 0.2639 | 0.8014 | +0.5375 |
| rolling_accept_rate | 0.7709 | 0.6147 | -0.1563 |

Step 15→16: accept_length 0→2

| Signal | Before (al=0) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2111 | 0.1757 | -0.0354 |
| joint_confidence_product | 0.2168 | 0.1722 | -0.0446 |
| joint_confidence_product_fixed | 0.2632 | 0.2766 | +0.0134 |
| confidence_agreement | 0.7853 | 0.9375 | +0.1522 |
| target_draft_agree | 0.0000 | 0.0000 | +0.0000 |
| target_top1_prob | 0.6315 | 0.5581 | -0.0734 |
| top1_prob | 0.4168 | 0.4957 | +0.0788 |
| target_entropy | 1.1919 | 1.4308 | +0.2389 |
| rolling_accept_rate | 0.5065 | 0.3546 | -0.1520 |

**Falls** (accept_length drops by ≥2): 53 transitions

Step 3→4: accept_length 4→1

| Signal | Before (al=4) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.7577 | 0.7624 | +0.0047 |
| joint_confidence_product | 0.9989 | 0.8604 | -0.1385 |
| joint_confidence_product_fixed | 0.9989 | 0.9023 | -0.0966 |
| confidence_agreement | 0.9989 | 0.9235 | -0.0754 |
| target_draft_agree | 1.0000 | 0.6000 | -0.4000 |
| target_top1_prob | 1.0000 | 0.9124 | -0.0876 |
| top1_prob | 0.9989 | 0.9889 | -0.0100 |
| target_entropy | 0.0001 | 0.2639 | +0.2638 |
| rolling_accept_rate | 0.7585 | 0.7709 | +0.0124 |

Step 12→13: accept_length 4→1

| Signal | Before (al=4) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.6704 | 0.8438 | +0.1734 |
| joint_confidence_product | 0.5018 | 0.6223 | +0.1204 |
| joint_confidence_product_fixed | 0.6154 | 0.7180 | +0.1026 |
| confidence_agreement | 0.9274 | 0.8778 | -0.0496 |
| target_draft_agree | 1.0000 | 1.0000 | +0.0000 |
| target_top1_prob | 0.8216 | 0.7885 | -0.0331 |
| top1_prob | 0.7490 | 0.9106 | +0.1616 |
| target_entropy | 0.4808 | 0.6019 | +0.1211 |
| rolling_accept_rate | 0.8951 | 0.9265 | +0.0315 |

Step 17→18: accept_length 3→0

| Signal | Before (al=3) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4791 | 0.2415 | -0.2376 |
| joint_confidence_product | 0.4888 | 0.3232 | -0.1656 |
| joint_confidence_product_fixed | 0.6504 | 0.3328 | -0.3176 |
| confidence_agreement | 0.8703 | 0.4108 | -0.4595 |
| target_draft_agree | 1.0000 | 1.0000 | +0.0000 |
| target_top1_prob | 0.7442 | 0.9424 | +0.1981 |
| top1_prob | 0.8739 | 0.3532 | -0.5208 |
| target_entropy | 0.7771 | 0.2552 | -0.5219 |
| rolling_accept_rate | 0.5482 | 0.6837 | +0.1355 |

**Sustained success streaks** (≥3 consecutive steps with accept_length ≥ 3): 16

Streak 1: steps 1-3 (3 steps), mean accept_len=4.00

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.6867 | 0.4310 | +0.2557 |
| joint_confidence_product | 0.9538 | 0.5280 | +0.4257 |
| joint_confidence_product_fixed | 0.9699 | 0.6087 | +0.3612 |
| confidence_agreement | 0.9700 | 0.7934 | +0.1767 |
| target_draft_agree | 0.8333 | 0.6670 | +0.1664 |
| target_top1_prob | 0.9716 | 0.8122 | +0.1594 |
| top1_prob | 0.9983 | 0.7429 | +0.2554 |
| target_entropy | 0.0953 | 0.5569 | -0.4616 |
| rolling_accept_rate | 0.6878 | 0.5728 | +0.1151 |

Streak 2: steps 5-7 (3 steps), mean accept_len=3.33

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.6165 | 0.4310 | +0.1855 |
| joint_confidence_product | 0.7248 | 0.5280 | +0.1968 |
| joint_confidence_product_fixed | 0.7784 | 0.6087 | +0.1697 |
| confidence_agreement | 0.9833 | 0.7934 | +0.1899 |
| target_draft_agree | 0.6833 | 0.6670 | +0.0164 |
| target_top1_prob | 0.8806 | 0.8122 | +0.0684 |
| top1_prob | 0.8748 | 0.7429 | +0.1319 |
| target_entropy | 0.3963 | 0.5569 | -0.1607 |
| rolling_accept_rate | 0.6987 | 0.5728 | +0.1259 |

**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): 2

Failure streak 1: steps 22-25 (4 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.1621 | 0.4310 | -0.2688 |
| joint_confidence_product | 0.1744 | 0.5280 | -0.3537 |
| joint_confidence_product_fixed | 0.2250 | 0.6087 | -0.3837 |
| confidence_agreement | 0.7031 | 0.7934 | -0.0903 |
| target_draft_agree | 0.2708 | 0.6670 | -0.3961 |
| target_top1_prob | 0.6443 | 0.8122 | -0.1679 |
| top1_prob | 0.3474 | 0.7429 | -0.3955 |
| target_entropy | 1.2359 | 0.5569 | +0.6789 |
| rolling_accept_rate | 0.4725 | 0.5728 | -0.1003 |

Failure streak 2: steps 47-49 (3 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.1245 | 0.4310 | -0.3064 |
| joint_confidence_product | 0.1583 | 0.5280 | -0.3698 |
| joint_confidence_product_fixed | 0.2311 | 0.6087 | -0.3776 |
| confidence_agreement | 0.7116 | 0.7934 | -0.0818 |
| target_draft_agree | 0.2778 | 0.6670 | -0.3892 |
| target_top1_prob | 0.6464 | 0.8122 | -0.1658 |
| top1_prob | 0.3580 | 0.7429 | -0.3848 |
| target_entropy | 0.8910 | 0.5569 | +0.3340 |
| rolling_accept_rate | 0.3535 | 0.5728 | -0.2193 |

---

**Turn 0** (239 steps, mean accept_len=2.05):

Accept length trajectory: `3 4 1 4 4 1 0 4 4 2 1 3 4 4 3 2 4 3 0 3 4 2 4 3 2 2 2 4 1 1 0 3 3 1 0 2 2 4 0 0 1 1 0 2 3 4 0 2 3 4 4 3 2 4 3 0 3 1 3 3 ... (179 more)`

**Spikes** (accept_length jumps up by ≥2): 55 transitions

Step 2→3: accept_length 1→4

| Signal | Before (al=1) | After (al=4) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4629 | 0.4576 | -0.0052 |
| joint_confidence_product | 0.7936 | 0.7421 | -0.0514 |
| joint_confidence_product_fixed | 0.7987 | 0.8106 | +0.0120 |
| confidence_agreement | 0.8162 | 0.8269 | +0.0107 |
| target_draft_agree | 0.8000 | 0.4000 | -0.4000 |
| target_top1_prob | 0.9903 | 0.8180 | -0.1724 |
| top1_prob | 0.8065 | 0.9911 | +0.1846 |
| target_entropy | 0.0587 | 0.6363 | +0.5777 |
| rolling_accept_rate | 0.5739 | 0.4618 | -0.1122 |

Step 6→7: accept_length 0→4

| Signal | Before (al=0) | After (al=4) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2224 | 0.3633 | +0.1409 |
| joint_confidence_product | 0.2780 | 0.8683 | +0.5902 |
| joint_confidence_product_fixed | 0.3273 | 0.9026 | +0.5752 |
| confidence_agreement | 0.5983 | 0.9927 | +0.3944 |
| target_draft_agree | 0.2500 | 0.6667 | +0.4167 |
| target_top1_prob | 0.8072 | 0.9537 | +0.1465 |
| top1_prob | 0.4055 | 0.9464 | +0.5409 |
| target_entropy | 0.5888 | 0.1863 | -0.4026 |
| rolling_accept_rate | 0.5484 | 0.3839 | -0.1645 |

Step 10→11: accept_length 1→3

| Signal | Before (al=1) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5462 | 0.4692 | -0.0770 |
| joint_confidence_product | 0.4491 | 0.3371 | -0.1120 |
| joint_confidence_product_fixed | 0.6843 | 0.5317 | -0.1526 |
| confidence_agreement | 0.7519 | 0.6222 | -0.1297 |
| target_draft_agree | 0.2500 | 0.6667 | +0.4167 |
| target_top1_prob | 0.7124 | 0.5643 | -0.1481 |
| top1_prob | 0.9605 | 0.9421 | -0.0184 |
| target_entropy | 0.5767 | 1.1266 | +0.5500 |
| rolling_accept_rate | 0.5687 | 0.4981 | -0.0706 |

**Falls** (accept_length drops by ≥2): 51 transitions

Step 1→2: accept_length 4→1

| Signal | Before (al=4) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4257 | 0.4629 | +0.0372 |
| joint_confidence_product | 0.6668 | 0.7936 | +0.1267 |
| joint_confidence_product_fixed | 0.7287 | 0.7987 | +0.0700 |
| confidence_agreement | 0.9241 | 0.8162 | -0.1080 |
| target_draft_agree | 0.6000 | 0.8000 | +0.2000 |
| target_top1_prob | 0.8165 | 0.9903 | +0.1738 |
| top1_prob | 0.8924 | 0.8065 | -0.0859 |
| target_entropy | 0.6339 | 0.0587 | -0.5752 |
| rolling_accept_rate | 0.4770 | 0.5739 | +0.0969 |

Step 4→5: accept_length 4→1

| Signal | Before (al=4) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5696 | 0.3693 | -0.2003 |
| joint_confidence_product | 0.6491 | 0.5114 | -0.1377 |
| joint_confidence_product_fixed | 0.7552 | 0.5237 | -0.2314 |
| confidence_agreement | 0.9124 | 0.5870 | -0.3254 |
| target_draft_agree | 1.0000 | 0.8000 | -0.2000 |
| target_top1_prob | 0.8263 | 0.9591 | +0.1328 |
| top1_prob | 0.9139 | 0.5461 | -0.3678 |
| target_entropy | 0.5234 | 0.1829 | -0.3405 |
| rolling_accept_rate | 0.6232 | 0.6763 | +0.0530 |

Step 8→9: accept_length 4→2

| Signal | Before (al=4) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5509 | 0.5980 | +0.0471 |
| joint_confidence_product | 0.8710 | 0.6091 | -0.2618 |
| joint_confidence_product_fixed | 0.9178 | 0.7408 | -0.1769 |
| confidence_agreement | 0.9786 | 0.7410 | -0.2376 |
| target_draft_agree | 1.0000 | 0.5000 | -0.5000 |
| target_top1_prob | 0.9474 | 0.7409 | -0.2064 |
| top1_prob | 0.9688 | 0.9999 | +0.0312 |
| target_entropy | 0.1725 | 0.8554 | +0.6829 |
| rolling_accept_rate | 0.5687 | 0.5981 | +0.0294 |

**Sustained success streaks** (≥3 consecutive steps with accept_length ≥ 3): 13

Streak 1: steps 11-14 (4 steps), mean accept_len=3.50

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.5757 | 0.3946 | +0.1811 |
| joint_confidence_product | 0.5169 | 0.5334 | -0.0165 |
| joint_confidence_product_fixed | 0.6567 | 0.6038 | +0.0529 |
| confidence_agreement | 0.7035 | 0.7637 | -0.0602 |
| target_draft_agree | 0.8167 | 0.6238 | +0.1929 |
| target_top1_prob | 0.7471 | 0.8303 | -0.0832 |
| top1_prob | 0.9002 | 0.7221 | +0.1781 |
| target_entropy | 0.6663 | 0.5216 | +0.1447 |
| rolling_accept_rate | 0.6417 | 0.5408 | +0.1008 |

Streak 2: steps 48-51 (4 steps), mean accept_len=3.50

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.6052 | 0.3946 | +0.2106 |
| joint_confidence_product | 0.7776 | 0.5334 | +0.2442 |
| joint_confidence_product_fixed | 0.8226 | 0.6038 | +0.2188 |
| confidence_agreement | 0.9112 | 0.7637 | +0.1475 |
| target_draft_agree | 0.8167 | 0.6238 | +0.1929 |
| target_top1_prob | 0.9242 | 0.8303 | +0.0939 |
| top1_prob | 0.8898 | 0.7221 | +0.1678 |
| target_entropy | 0.2301 | 0.5216 | -0.2915 |
| rolling_accept_rate | 0.6791 | 0.5408 | +0.1383 |

**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): 2

Failure streak 1: steps 65-68 (4 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.0952 | 0.3946 | -0.2994 |
| joint_confidence_product | 0.2172 | 0.5334 | -0.3162 |
| joint_confidence_product_fixed | 0.2447 | 0.6038 | -0.3591 |
| confidence_agreement | 0.5384 | 0.7637 | -0.2253 |
| target_draft_agree | 0.7083 | 0.6238 | +0.0846 |
| target_top1_prob | 0.7751 | 0.8303 | -0.0552 |
| top1_prob | 0.3134 | 0.7221 | -0.4086 |
| target_entropy | 0.8956 | 0.5216 | +0.3740 |
| rolling_accept_rate | 0.2937 | 0.5408 | -0.2471 |

Failure streak 2: steps 128-132 (5 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.2066 | 0.3946 | -0.1880 |
| joint_confidence_product | 0.2510 | 0.5334 | -0.2824 |
| joint_confidence_product_fixed | 0.3153 | 0.6038 | -0.2885 |
| confidence_agreement | 0.7955 | 0.7637 | +0.0319 |
| target_draft_agree | 0.2333 | 0.6238 | -0.3904 |
| target_top1_prob | 0.6609 | 0.8303 | -0.1694 |
| top1_prob | 0.4564 | 0.7221 | -0.2656 |
| target_entropy | 1.1713 | 0.5216 | +0.6497 |
| rolling_accept_rate | 0.3942 | 0.5408 | -0.1467 |

---

**Turn 1** (174 steps, mean accept_len=2.20):

Accept length trajectory: `1 3 4 4 2 4 4 4 4 0 0 0 0 0 0 3 3 0 2 3 3 2 2 2 3 3 3 1 1 2 3 0 3 4 4 1 3 0 3 2 3 4 4 4 4 0 2 3 3 3 4 3 1 4 4 4 0 1 0 1 ... (114 more)`

**Spikes** (accept_length jumps up by ≥2): 37 transitions

Step 0→1: accept_length 1→3

| Signal | Before (al=1) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2512 | 0.4486 | +0.1973 |
| joint_confidence_product | 0.4383 | 0.7010 | +0.2627 |
| joint_confidence_product_fixed | 0.4703 | 0.7746 | +0.3043 |
| confidence_agreement | 0.5665 | 0.7803 | +0.2138 |
| target_draft_agree | 1.0000 | 0.3333 | -0.6667 |
| target_top1_prob | 0.9360 | 0.7771 | -0.1589 |
| top1_prob | 0.5025 | 0.9968 | +0.4943 |
| target_entropy | 0.1934 | 1.0623 | +0.8689 |
| rolling_accept_rate | 0.5000 | 0.4500 | -0.0500 |

Step 4→5: accept_length 2→4

| Signal | Before (al=2) | After (al=4) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.7093 | 0.6962 | -0.0131 |
| joint_confidence_product | 0.7022 | 0.9879 | +0.2856 |
| joint_confidence_product_fixed | 0.7467 | 0.9937 | +0.2470 |
| confidence_agreement | 0.7468 | 0.9948 | +0.2480 |
| target_draft_agree | 0.6000 | 0.6667 | +0.0667 |
| target_top1_prob | 0.7467 | 0.9942 | +0.2475 |
| top1_prob | 0.9999 | 0.9994 | -0.0005 |
| target_entropy | 1.0124 | 0.0280 | -0.9844 |
| rolling_accept_rate | 0.7093 | 0.6965 | -0.0128 |

Step 14→15: accept_length 0→3

| Signal | Before (al=0) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.0587 | 0.0939 | +0.0352 |
| joint_confidence_product | 0.2158 | 0.3768 | +0.1610 |
| joint_confidence_product_fixed | 0.2740 | 0.5892 | +0.3152 |
| confidence_agreement | 0.8021 | 0.6025 | -0.1997 |
| target_draft_agree | 0.5000 | 0.0000 | -0.5000 |
| target_top1_prob | 0.6317 | 0.5941 | -0.0375 |
| top1_prob | 0.4338 | 0.9917 | +0.5579 |
| target_entropy | 1.2464 | 1.1990 | -0.0474 |
| rolling_accept_rate | 0.1352 | 0.0947 | -0.0406 |

**Falls** (accept_length drops by ≥2): 38 transitions

Step 3→4: accept_length 4→2

| Signal | Before (al=4) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5499 | 0.7093 | +0.1594 |
| joint_confidence_product | 0.8185 | 0.7022 | -0.1162 |
| joint_confidence_product_fixed | 0.8193 | 0.7467 | -0.0726 |
| confidence_agreement | 0.8211 | 0.7468 | -0.0743 |
| target_draft_agree | 0.8000 | 0.6000 | -0.2000 |
| target_top1_prob | 0.9990 | 0.7467 | -0.2522 |
| top1_prob | 0.8201 | 0.9999 | +0.1798 |
| target_entropy | 0.0065 | 1.0124 | +1.0059 |
| rolling_accept_rate | 0.6705 | 0.7093 | +0.0388 |

Step 8→9: accept_length 4→0

| Signal | Before (al=4) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.8065 | 0.4733 | -0.3332 |
| joint_confidence_product | 0.9273 | 0.5752 | -0.3522 |
| joint_confidence_product_fixed | 0.9553 | 0.5799 | -0.3754 |
| confidence_agreement | 0.9553 | 0.6024 | -0.3529 |
| target_draft_agree | 0.6000 | 0.8000 | +0.2000 |
| target_top1_prob | 0.9553 | 0.9858 | +0.0305 |
| top1_prob | 1.0000 | 0.5883 | -0.4117 |
| target_entropy | 0.1939 | 0.0765 | -0.1174 |
| rolling_accept_rate | 0.8065 | 0.8046 | -0.0020 |

Step 16→17: accept_length 3→0

| Signal | Before (al=3) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2738 | 0.1086 | -0.1652 |
| joint_confidence_product | 0.6166 | 0.1800 | -0.4367 |
| joint_confidence_product_fixed | 0.6563 | 0.1864 | -0.4699 |
| confidence_agreement | 0.8695 | 0.2398 | -0.6297 |
| target_draft_agree | 1.0000 | 1.0000 | +0.0000 |
| target_top1_prob | 0.8780 | 0.9553 | +0.0773 |
| top1_prob | 0.7475 | 0.1951 | -0.5523 |
| target_entropy | 0.4916 | 0.1790 | -0.3126 |
| rolling_accept_rate | 0.3663 | 0.5564 | +0.1901 |

**Sustained success streaks** (≥3 consecutive steps with accept_length ≥ 3): 14

Streak 1: steps 1-3 (3 steps), mean accept_len=3.67

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.5378 | 0.4848 | +0.0530 |
| joint_confidence_product | 0.8350 | 0.6476 | +0.1873 |
| joint_confidence_product_fixed | 0.8617 | 0.7103 | +0.1514 |
| confidence_agreement | 0.8642 | 0.8144 | +0.0499 |
| target_draft_agree | 0.7111 | 0.6513 | +0.0598 |
| target_top1_prob | 0.9224 | 0.8527 | +0.0697 |
| top1_prob | 0.9389 | 0.8253 | +0.1136 |
| target_entropy | 0.3750 | 0.4859 | -0.1109 |
| rolling_accept_rate | 0.5785 | 0.5816 | -0.0031 |

Streak 2: steps 5-8 (4 steps), mean accept_len=4.00

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.7351 | 0.4848 | +0.2503 |
| joint_confidence_product | 0.8989 | 0.6476 | +0.2512 |
| joint_confidence_product_fixed | 0.9275 | 0.7103 | +0.2173 |
| confidence_agreement | 0.9329 | 0.8144 | +0.1186 |
| target_draft_agree | 0.7167 | 0.6513 | +0.0653 |
| target_top1_prob | 0.9581 | 0.8527 | +0.1054 |
| top1_prob | 0.9692 | 0.8253 | +0.1439 |
| target_entropy | 0.1528 | 0.4859 | -0.3331 |
| rolling_accept_rate | 0.7600 | 0.5816 | +0.1784 |

**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): 1

Failure streak 1: steps 9-14 (6 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.1669 | 0.4848 | -0.3178 |
| joint_confidence_product | 0.3086 | 0.6476 | -0.3390 |
| joint_confidence_product_fixed | 0.3496 | 0.7103 | -0.3607 |
| confidence_agreement | 0.6901 | 0.8144 | -0.1243 |
| target_draft_agree | 0.4250 | 0.6513 | -0.2263 |
| target_top1_prob | 0.7542 | 0.8527 | -0.0985 |
| top1_prob | 0.4443 | 0.8253 | -0.3810 |
| target_entropy | 0.8366 | 0.4859 | +0.3506 |
| rolling_accept_rate | 0.3944 | 0.5816 | -0.1872 |

---

### DS+LlamaDraft

**Turn 12** (904 steps, mean accept_len=0.74):

Accept length trajectory: `0 0 0 0 1 1 1 2 0 1 2 4 4 4 2 0 0 0 0 0 1 2 0 2 0 2 3 0 0 1 2 0 1 1 0 0 0 0 1 1 0 0 0 0 0 1 0 1 0 0 1 3 1 0 1 0 0 2 1 0 ... (844 more)`

**Spikes** (accept_length jumps up by ≥2): 117 transitions

Step 10→11: accept_length 2→4

| Signal | Before (al=2) | After (al=4) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.1000 | 0.3752 | +0.2752 |
| joint_confidence_product | 0.3205 | 0.9774 | +0.6570 |
| joint_confidence_product_fixed | 0.3496 | 0.9775 | +0.6278 |
| confidence_agreement | 0.4623 | 0.9775 | +0.5153 |
| target_draft_agree | 0.6667 | 0.6667 | +0.0000 |
| target_top1_prob | 0.9184 | 1.0000 | +0.0815 |
| top1_prob | 0.3807 | 0.9775 | +0.5968 |
| target_entropy | 0.2049 | 0.0005 | -0.2044 |
| rolling_accept_rate | 0.2626 | 0.3838 | +0.1212 |

Step 22→23: accept_length 0→2

| Signal | Before (al=0) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.0135 | 0.1935 | +0.1801 |
| joint_confidence_product | 0.0255 | 0.6493 | +0.6239 |
| joint_confidence_product_fixed | 0.0318 | 0.7050 | +0.6733 |
| confidence_agreement | 0.2073 | 0.8842 | +0.6769 |
| target_draft_agree | 0.6667 | 0.5000 | -0.1667 |
| target_top1_prob | 0.8310 | 0.8996 | +0.0686 |
| top1_prob | 0.0382 | 0.7837 | +0.7455 |
| target_entropy | 0.3517 | 0.3228 | -0.0290 |
| rolling_accept_rate | 0.3528 | 0.2469 | -0.1058 |

Step 24→25: accept_length 0→2

| Signal | Before (al=0) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.1334 | 0.0295 | -0.1039 |
| joint_confidence_product | 0.2249 | 0.0446 | -0.1802 |
| joint_confidence_product_fixed | 0.2894 | 0.0744 | -0.2150 |
| confidence_agreement | 0.5492 | 0.4563 | -0.0930 |
| target_draft_agree | 0.6667 | 0.6667 | +0.0000 |
| target_top1_prob | 0.8086 | 0.6570 | -0.1517 |
| top1_prob | 0.3579 | 0.1132 | -0.2447 |
| target_entropy | 0.4544 | 0.9283 | +0.4740 |
| rolling_accept_rate | 0.3729 | 0.2610 | -0.1119 |

**Falls** (accept_length drops by ≥2): 126 transitions

Step 7→8: accept_length 2→0

| Signal | Before (al=2) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2597 | 0.1148 | -0.1448 |
| joint_confidence_product | 0.9201 | 0.3171 | -0.6029 |
| joint_confidence_product_fixed | 0.9520 | 0.3299 | -0.6221 |
| confidence_agreement | 0.9522 | 0.3931 | -0.5590 |
| target_draft_agree | 0.5000 | 0.5000 | +0.0000 |
| target_top1_prob | 0.9521 | 0.9530 | +0.0009 |
| top1_prob | 0.9999 | 0.3461 | -0.6538 |
| target_entropy | 0.2101 | 0.1670 | -0.0431 |
| rolling_accept_rate | 0.2597 | 0.3318 | +0.0721 |

Step 13→14: accept_length 4→2

| Signal | Before (al=4) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5955 | 0.5246 | -0.0709 |
| joint_confidence_product | 0.9692 | 0.7963 | -0.1729 |
| joint_confidence_product_fixed | 0.9834 | 0.7968 | -0.1865 |
| confidence_agreement | 0.9854 | 0.7997 | -0.1857 |
| target_draft_agree | 0.8000 | 0.8000 | +0.0000 |
| target_top1_prob | 0.9844 | 0.9984 | +0.0140 |
| top1_prob | 0.9990 | 0.7981 | -0.2009 |
| target_entropy | 0.0622 | 0.0116 | -0.0506 |
| rolling_accept_rate | 0.5961 | 0.6572 | +0.0612 |

Step 14→15: accept_length 2→0

| Signal | Before (al=2) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5246 | 0.1284 | -0.3962 |
| joint_confidence_product | 0.7963 | 0.1528 | -0.6435 |
| joint_confidence_product_fixed | 0.7968 | 0.1781 | -0.6188 |
| confidence_agreement | 0.7997 | 0.4168 | -0.3829 |
| target_draft_agree | 0.8000 | 0.4000 | -0.4000 |
| target_top1_prob | 0.9984 | 0.8045 | -0.1939 |
| top1_prob | 0.7981 | 0.2213 | -0.5768 |
| target_entropy | 0.0116 | 0.5616 | +0.5500 |
| rolling_accept_rate | 0.6572 | 0.5801 | -0.0772 |

**Sustained success streaks** (≥3 consecutive steps with accept_length ≥ 3): 3

Streak 1: steps 11-13 (3 steps), mean accept_len=4.00

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.4930 | 0.1279 | +0.3651 |
| joint_confidence_product | 0.9615 | 0.3341 | +0.6274 |
| joint_confidence_product_fixed | 0.9749 | 0.3995 | +0.5754 |
| confidence_agreement | 0.9760 | 0.6916 | +0.2844 |
| target_draft_agree | 0.6889 | 0.3574 | +0.3314 |
| target_top1_prob | 0.9829 | 0.7494 | +0.2335 |
| top1_prob | 0.9920 | 0.5276 | +0.4644 |
| target_entropy | 0.0741 | 0.8764 | -0.8023 |
| rolling_accept_rate | 0.4962 | 0.2453 | +0.2509 |

Streak 2: steps 250-252 (3 steps), mean accept_len=3.00

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.3243 | 0.1279 | +0.1964 |
| joint_confidence_product | 0.5673 | 0.3341 | +0.2332 |
| joint_confidence_product_fixed | 0.6169 | 0.3995 | +0.2173 |
| confidence_agreement | 0.8185 | 0.6916 | +0.1269 |
| target_draft_agree | 0.5500 | 0.3574 | +0.1926 |
| target_top1_prob | 0.8736 | 0.7494 | +0.1242 |
| top1_prob | 0.6921 | 0.5276 | +0.1645 |
| target_entropy | 0.3570 | 0.8764 | -0.5194 |
| rolling_accept_rate | 0.4728 | 0.2453 | +0.2276 |

**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): 75

Failure streak 1: steps 0-3 (4 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.0524 | 0.1279 | -0.0755 |
| joint_confidence_product | 0.0995 | 0.3341 | -0.2346 |
| joint_confidence_product_fixed | 0.1336 | 0.3995 | -0.2659 |
| confidence_agreement | 0.5330 | 0.6916 | -0.1585 |
| target_draft_agree | 0.1875 | 0.3574 | -0.1699 |
| target_top1_prob | 0.6864 | 0.7494 | -0.0630 |
| top1_prob | 0.2194 | 0.5276 | -0.3081 |
| target_entropy | 1.3585 | 0.8764 | +0.4821 |
| rolling_accept_rate | 0.3166 | 0.2453 | +0.0713 |

Failure streak 2: steps 15-19 (5 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.0945 | 0.1279 | -0.0334 |
| joint_confidence_product | 0.1325 | 0.3341 | -0.2017 |
| joint_confidence_product_fixed | 0.1910 | 0.3995 | -0.2085 |
| confidence_agreement | 0.6386 | 0.6916 | -0.0529 |
| target_draft_agree | 0.1800 | 0.3574 | -0.1774 |
| target_top1_prob | 0.6608 | 0.7494 | -0.0886 |
| top1_prob | 0.2994 | 0.5276 | -0.2281 |
| target_entropy | 0.9453 | 0.8764 | +0.0689 |
| rolling_accept_rate | 0.3217 | 0.2453 | +0.0764 |

---

**Turn 0** (710 steps, mean accept_len=0.66):

Accept length trajectory: `1 4 4 0 0 3 4 1 1 0 2 3 1 0 1 0 0 3 1 0 1 1 0 0 0 0 0 0 0 0 2 0 2 0 0 0 0 0 2 1 0 2 0 1 2 0 0 0 0 0 0 0 0 0 1 1 0 2 0 1 ... (650 more)`

**Spikes** (accept_length jumps up by ≥2): 78 transitions

Step 0→1: accept_length 1→4

| Signal | Before (al=1) | After (al=4) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2510 | 0.3348 | +0.0838 |
| joint_confidence_product | 0.3210 | 0.9469 | +0.6259 |
| joint_confidence_product_fixed | 0.4398 | 0.9710 | +0.5311 |
| confidence_agreement | 0.9821 | 0.9800 | -0.0021 |
| target_draft_agree | 0.3333 | 0.2500 | -0.0833 |
| target_top1_prob | 0.6543 | 0.9755 | +0.3211 |
| top1_prob | 0.6722 | 0.9954 | +0.3232 |
| target_entropy | 1.0159 | 0.0793 | -0.9366 |
| rolling_accept_rate | 0.3733 | 0.3363 | -0.0370 |

Step 4→5: accept_length 0→3

| Signal | Before (al=0) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.1191 | 0.2588 | +0.1397 |
| joint_confidence_product | 0.3138 | 0.8858 | +0.5720 |
| joint_confidence_product_fixed | 0.3165 | 0.9168 | +0.6003 |
| confidence_agreement | 0.3281 | 0.9338 | +0.6057 |
| target_draft_agree | 0.6667 | 0.3333 | -0.3333 |
| target_top1_prob | 0.9912 | 0.9250 | -0.0663 |
| top1_prob | 0.3193 | 0.9911 | +0.6718 |
| target_entropy | 0.0412 | 0.4020 | +0.3609 |
| rolling_accept_rate | 0.3730 | 0.2611 | -0.1119 |

Step 9→10: accept_length 0→2

| Signal | Before (al=0) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.1476 | 0.2547 | +0.1071 |
| joint_confidence_product | 0.2715 | 0.8320 | +0.5606 |
| joint_confidence_product_fixed | 0.3101 | 0.8573 | +0.5471 |
| confidence_agreement | 0.5300 | 0.9694 | +0.4394 |
| target_draft_agree | 0.6667 | 0.3333 | -0.3333 |
| target_top1_prob | 0.8395 | 0.9413 | +0.1019 |
| top1_prob | 0.3694 | 0.9107 | +0.5413 |
| target_entropy | 0.5547 | 0.2681 | -0.2866 |
| rolling_accept_rate | 0.3995 | 0.2796 | -0.1198 |

**Falls** (accept_length drops by ≥2): 79 transitions

Step 2→3: accept_length 4→0

| Signal | Before (al=4) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4728 | 0.2544 | -0.2184 |
| joint_confidence_product | 0.9848 | 0.3282 | -0.6566 |
| joint_confidence_product_fixed | 0.9895 | 0.3831 | -0.6064 |
| confidence_agreement | 0.9992 | 0.6750 | -0.3242 |
| target_draft_agree | 0.8000 | 0.5000 | -0.3000 |
| target_top1_prob | 0.9951 | 0.8025 | -0.1927 |
| top1_prob | 0.9944 | 0.4775 | -0.5169 |
| target_entropy | 0.0245 | 0.6979 | +0.6734 |
| rolling_accept_rate | 0.4754 | 0.5328 | +0.0574 |

Step 6→7: accept_length 4→1

| Signal | Before (al=4) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.3976 | 0.4893 | +0.0917 |
| joint_confidence_product | 0.9746 | 0.7383 | -0.2363 |
| joint_confidence_product_fixed | 0.9749 | 0.7858 | -0.1891 |
| confidence_agreement | 0.9754 | 0.9125 | -0.0629 |
| target_draft_agree | 0.5000 | 0.8000 | +0.3000 |
| target_top1_prob | 0.9997 | 0.8438 | -0.1559 |
| top1_prob | 0.9751 | 0.9312 | -0.0439 |
| target_entropy | 0.0025 | 0.5844 | +0.5819 |
| rolling_accept_rate | 0.4078 | 0.5254 | +0.1177 |

Step 11→12: accept_length 3→1

| Signal | Before (al=3) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2293 | 0.2807 | +0.0514 |
| joint_confidence_product | 0.5132 | 0.4851 | -0.0281 |
| joint_confidence_product_fixed | 0.5730 | 0.4984 | -0.0745 |
| confidence_agreement | 0.7993 | 0.5556 | -0.2437 |
| target_draft_agree | 0.5000 | 1.0000 | +0.5000 |
| target_top1_prob | 0.8639 | 0.9624 | +0.0984 |
| top1_prob | 0.6632 | 0.5179 | -0.1453 |
| target_entropy | 0.4863 | 0.1652 | -0.3211 |
| rolling_accept_rate | 0.3457 | 0.5420 | +0.1963 |

**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): 54

Failure streak 1: steps 22-29 (8 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.0664 | 0.1114 | -0.0450 |
| joint_confidence_product | 0.2676 | 0.3332 | -0.0656 |
| joint_confidence_product_fixed | 0.3232 | 0.3895 | -0.0662 |
| confidence_agreement | 0.7464 | 0.6641 | +0.0823 |
| target_draft_agree | 0.1875 | 0.3281 | -0.1406 |
| target_top1_prob | 0.6886 | 0.7595 | -0.0709 |
| top1_prob | 0.4460 | 0.5049 | -0.0590 |
| target_entropy | 1.0692 | 0.9140 | +0.1552 |
| rolling_accept_rate | 0.1189 | 0.2207 | -0.1017 |

Failure streak 2: steps 33-37 (5 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.1172 | 0.1114 | +0.0059 |
| joint_confidence_product | 0.2902 | 0.3332 | -0.0430 |
| joint_confidence_product_fixed | 0.3682 | 0.3895 | -0.0213 |
| confidence_agreement | 0.7617 | 0.6641 | +0.0975 |
| target_draft_agree | 0.2500 | 0.3281 | -0.0781 |
| target_top1_prob | 0.7302 | 0.7595 | -0.0293 |
| top1_prob | 0.5004 | 0.5049 | -0.0045 |
| target_entropy | 0.7735 | 0.9140 | -0.1405 |
| rolling_accept_rate | 0.2512 | 0.2207 | +0.0306 |

---

**Turn 4** (587 steps, mean accept_len=0.75):

Accept length trajectory: `1 0 0 3 3 4 1 0 2 1 0 3 3 0 1 0 0 1 0 2 2 0 2 1 1 1 0 1 0 0 0 1 1 1 1 1 0 0 0 0 1 0 1 0 3 0 0 1 1 2 0 2 2 0 0 0 1 0 0 0 ... (527 more)`

**Spikes** (accept_length jumps up by ≥2): 71 transitions

Step 2→3: accept_length 0→3

| Signal | Before (al=0) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.2169 | 0.2000 | -0.0169 |
| joint_confidence_product | 0.5284 | 0.5520 | +0.0236 |
| joint_confidence_product_fixed | 0.5538 | 0.6745 | +0.1207 |
| confidence_agreement | 0.8840 | 0.8369 | -0.0471 |
| target_draft_agree | 0.0000 | 0.0000 | +0.0000 |
| target_top1_prob | 0.8045 | 0.7438 | -0.0607 |
| top1_prob | 0.6884 | 0.9069 | +0.2185 |
| target_entropy | 1.1029 | 0.9286 | -0.1743 |
| rolling_accept_rate | 0.3150 | 0.2205 | -0.0945 |

Step 7→8: accept_length 0→2

| Signal | Before (al=0) | After (al=2) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.0643 | 0.1142 | +0.0499 |
| joint_confidence_product | 0.0795 | 0.1504 | +0.0709 |
| joint_confidence_product_fixed | 0.0959 | 0.1988 | +0.1029 |
| confidence_agreement | 0.4425 | 0.7820 | +0.3395 |
| target_draft_agree | 0.2000 | 0.0000 | -0.2000 |
| target_top1_prob | 0.6954 | 0.5680 | -0.1274 |
| top1_prob | 0.1379 | 0.3500 | +0.2121 |
| target_entropy | 0.9959 | 1.4086 | +0.4127 |
| rolling_accept_rate | 0.4661 | 0.3263 | -0.1398 |

Step 10→11: accept_length 0→3

| Signal | Before (al=0) | After (al=3) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.0865 | 0.2542 | +0.1677 |
| joint_confidence_product | 0.0892 | 0.3542 | +0.2650 |
| joint_confidence_product_fixed | 0.1322 | 0.4923 | +0.3602 |
| confidence_agreement | 0.5445 | 0.7379 | +0.1934 |
| target_draft_agree | 0.4000 | 0.0000 | -0.4000 |
| target_top1_prob | 0.6567 | 0.5827 | -0.0740 |
| top1_prob | 0.2012 | 0.8449 | +0.6436 |
| target_entropy | 1.0105 | 1.3749 | +0.3644 |
| rolling_accept_rate | 0.4299 | 0.3009 | -0.1290 |

**Falls** (accept_length drops by ≥2): 65 transitions

Step 5→6: accept_length 4→1

| Signal | Before (al=4) | After (al=1) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.5309 | 0.4265 | -0.1044 |
| joint_confidence_product | 0.9619 | 0.6896 | -0.2723 |
| joint_confidence_product_fixed | 0.9687 | 0.7119 | -0.2568 |
| confidence_agreement | 0.9867 | 0.7666 | -0.2201 |
| target_draft_agree | 0.7500 | 0.5000 | -0.2500 |
| target_top1_prob | 0.9909 | 0.9685 | -0.0224 |
| top1_prob | 0.9776 | 0.7351 | -0.2425 |
| target_entropy | 0.0541 | 0.1260 | +0.0719 |
| rolling_accept_rate | 0.5430 | 0.5801 | +0.0371 |

Step 12→13: accept_length 3→0

| Signal | Before (al=3) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4101 | 0.2667 | -0.1434 |
| joint_confidence_product | 0.7316 | 0.2577 | -0.4739 |
| joint_confidence_product_fixed | 0.7530 | 0.3461 | -0.4069 |
| confidence_agreement | 0.8653 | 0.7986 | -0.0667 |
| target_draft_agree | 1.0000 | 0.4000 | -0.6000 |
| target_top1_prob | 0.9377 | 0.6976 | -0.2401 |
| top1_prob | 0.8030 | 0.4962 | -0.3069 |
| target_entropy | 0.2660 | 0.8668 | +0.6008 |
| rolling_accept_rate | 0.5106 | 0.5374 | +0.0268 |

Step 20→21: accept_length 2→0

| Signal | Before (al=2) | After (al=0) | Delta |
|--------|--------|-------|-------|
| draft_oracle_gate | 0.4096 | 0.0783 | -0.3313 |
| joint_confidence_product | 0.3251 | 0.0892 | -0.2359 |
| joint_confidence_product_fixed | 0.5096 | 0.1117 | -0.3979 |
| confidence_agreement | 0.5835 | 0.4416 | -0.1419 |
| target_draft_agree | 1.0000 | 0.6667 | -0.3333 |
| target_top1_prob | 0.5353 | 0.7146 | +0.1793 |
| top1_prob | 0.9518 | 0.1563 | -0.7956 |
| target_entropy | 1.4934 | 0.8939 | -0.5996 |
| rolling_accept_rate | 0.4304 | 0.5013 | +0.0709 |

**Sustained success streaks** (≥3 consecutive steps with accept_length ≥ 3): 2

Streak 1: steps 3-5 (3 steps), mean accept_len=3.33

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.3441 | 0.1256 | +0.2186 |
| joint_confidence_product | 0.6259 | 0.3240 | +0.3019 |
| joint_confidence_product_fixed | 0.7037 | 0.3868 | +0.3169 |
| confidence_agreement | 0.9274 | 0.6720 | +0.2554 |
| target_draft_agree | 0.5833 | 0.3549 | +0.2284 |
| target_top1_prob | 0.8132 | 0.7595 | +0.0538 |
| top1_prob | 0.8494 | 0.5029 | +0.3465 |
| target_entropy | 0.5964 | 0.7729 | -0.1765 |
| rolling_accept_rate | 0.4060 | 0.2404 | +0.1656 |

Streak 2: steps 513-515 (3 steps), mean accept_len=3.33

| Signal | Streak Mean | Turn Mean | Delta |
|--------|------------|-----------|-------|
| draft_oracle_gate | 0.4924 | 0.1256 | +0.3668 |
| joint_confidence_product | 0.6607 | 0.3240 | +0.3368 |
| joint_confidence_product_fixed | 0.7167 | 0.3868 | +0.3299 |
| confidence_agreement | 0.8044 | 0.6720 | +0.1324 |
| target_draft_agree | 0.8222 | 0.3549 | +0.4673 |
| target_top1_prob | 0.9110 | 0.7595 | +0.1516 |
| top1_prob | 0.7943 | 0.5029 | +0.2915 |
| target_entropy | 0.2460 | 0.7729 | -0.5269 |
| rolling_accept_rate | 0.5962 | 0.2404 | +0.3558 |

**Sustained failure streaks** (≥3 consecutive steps with accept_length = 0): 51

Failure streak 1: steps 28-30 (3 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.1015 | 0.1256 | -0.0240 |
| joint_confidence_product | 0.2694 | 0.3240 | -0.0546 |
| joint_confidence_product_fixed | 0.3144 | 0.3868 | -0.0724 |
| confidence_agreement | 0.6451 | 0.6720 | -0.0269 |
| target_draft_agree | 0.1667 | 0.3549 | -0.1883 |
| target_top1_prob | 0.7630 | 0.7595 | +0.0035 |
| top1_prob | 0.4080 | 0.5029 | -0.0948 |
| target_entropy | 0.9619 | 0.7729 | +0.1890 |
| rolling_accept_rate | 0.2415 | 0.2404 | +0.0011 |

Failure streak 2: steps 36-39 (4 steps)

| Signal | Failure Mean | Turn Mean | Delta |
|--------|-------------|-----------|-------|
| draft_oracle_gate | 0.0814 | 0.1256 | -0.0442 |
| joint_confidence_product | 0.2295 | 0.3240 | -0.0945 |
| joint_confidence_product_fixed | 0.2801 | 0.3868 | -0.1067 |
| confidence_agreement | 0.6540 | 0.6720 | -0.0180 |
| target_draft_agree | 0.4583 | 0.3549 | +0.1034 |
| target_top1_prob | 0.7382 | 0.7595 | -0.0213 |
| top1_prob | 0.3922 | 0.5029 | -0.1107 |
| target_entropy | 0.8986 | 0.7729 | +0.1257 |
| rolling_accept_rate | 0.2061 | 0.2404 | -0.0342 |

---
