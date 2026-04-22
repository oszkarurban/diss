# Phase B — Analysis summary

Source: 33 Phase A trace cells (Llama full 18/18, Qwen 15/18 — v3_dynamic missing for Qwen).

## 1. Cross-cell signal stationarity (KS divergence)

| signal              |   max_ks |   mean_ks |   max_ks_within_model |   max_ks_across_model | decision_gate        |
|:--------------------|---------:|----------:|----------------------:|----------------------:|:---------------------|
| top1_prob           |    0.317 |     0.168 |                 0.317 |                 0.284 | per-model thresholds |
| target_top1_prob    |    0.321 |     0.167 |                 0.197 |                 0.321 | per-model thresholds |
| rolling_accept_rate |    0.441 |     0.228 |                 0.334 |                 0.441 | per-model thresholds |
| DOG                 |    0.384 |     0.215 |                 0.343 |                 0.384 | per-model thresholds |
| t                   |    0.324 |     0.167 |                 0.324 |                 0.265 | per-model thresholds |

## 2. Signal vs AL discriminability (Spearman rank correlation)

- Median Spearman r across cells × configs:
  - **DOG** = 0.457
  - **t**   = 0.524
- **Winning signal: `t`**

### Per-cell Spearman r (DOG)

| signal   | model   | dataset       | config         |     n |   spearman_r |   p_value |
|:---------|:--------|:--------------|:---------------|------:|-------------:|----------:|
| DOG      | llama   | livecodebench | static_3_1_4   |  3821 |        0.677 |     0.000 |
| DOG      | llama   | livecodebench | static_6_10_60 |  3205 |        0.680 |     0.000 |
| DOG      | llama   | livecodebench | static_7_1_8   |  3118 |        0.629 |     0.000 |
| DOG      | llama   | livecodebench | static_7_4_8   |  2882 |        0.660 |     0.000 |
| DOG      | llama   | math500       | static_3_1_4   |  3834 |        0.529 |     0.000 |
| DOG      | llama   | math500       | static_6_10_60 |  2802 |        0.443 |     0.000 |
| DOG      | llama   | math500       | static_7_1_8   |  3499 |        0.543 |     0.000 |
| DOG      | llama   | math500       | static_7_4_8   |  2878 |        0.556 |     0.000 |
| DOG      | llama   | mtbench       | static_3_1_4   |  4580 |        0.463 |     0.000 |
| DOG      | llama   | mtbench       | static_6_10_60 |  2472 |        0.379 |     0.000 |
| DOG      | llama   | mtbench       | static_7_1_8   |  3696 |        0.451 |     0.000 |
| DOG      | llama   | mtbench       | static_7_4_8   |  3539 |        0.446 |     0.000 |
| DOG      | qwen    | livecodebench | static_3_1_4   | 21188 |        0.375 |     0.000 |
| DOG      | qwen    | livecodebench | static_6_10_60 | 12823 |        0.332 |     0.000 |
| DOG      | qwen    | livecodebench | static_7_1_8   | 20303 |        0.361 |     0.000 |
| DOG      | qwen    | livecodebench | static_7_4_8   | 17016 |        0.367 |     0.000 |
| DOG      | qwen    | math500       | static_3_1_4   | 20620 |        0.410 |     0.000 |
| DOG      | qwen    | math500       | static_6_10_60 | 12158 |        0.385 |     0.000 |
| DOG      | qwen    | math500       | static_7_1_8   | 19714 |        0.385 |     0.000 |
| DOG      | qwen    | math500       | static_7_4_8   | 16586 |        0.389 |     0.000 |
| DOG      | qwen    | mtbench       | static_3_1_4   | 26058 |        0.538 |     0.000 |
| DOG      | qwen    | mtbench       | static_6_10_60 | 14229 |        0.545 |     0.000 |
| DOG      | qwen    | mtbench       | static_7_1_8   | 18465 |        0.611 |     0.000 |
| DOG      | qwen    | mtbench       | static_7_4_8   | 18114 |        0.616 |     0.000 |

### Per-cell Spearman r (t)

| signal   | model   | dataset       | config         |     n |   spearman_r |   p_value |
|:---------|:--------|:--------------|:---------------|------:|-------------:|----------:|
| t        | llama   | livecodebench | static_3_1_4   |  3821 |        0.723 |     0.000 |
| t        | llama   | livecodebench | static_6_10_60 |  3205 |        0.724 |     0.000 |
| t        | llama   | livecodebench | static_7_1_8   |  3118 |        0.719 |     0.000 |
| t        | llama   | livecodebench | static_7_4_8   |  2882 |        0.701 |     0.000 |
| t        | llama   | math500       | static_3_1_4   |  3834 |        0.550 |     0.000 |
| t        | llama   | math500       | static_6_10_60 |  2802 |        0.422 |     0.000 |
| t        | llama   | math500       | static_7_1_8   |  3499 |        0.593 |     0.000 |
| t        | llama   | math500       | static_7_4_8   |  2878 |        0.564 |     0.000 |
| t        | llama   | mtbench       | static_3_1_4   |  4580 |        0.530 |     0.000 |
| t        | llama   | mtbench       | static_6_10_60 |  2472 |        0.403 |     0.000 |
| t        | llama   | mtbench       | static_7_1_8   |  3696 |        0.541 |     0.000 |
| t        | llama   | mtbench       | static_7_4_8   |  3539 |        0.483 |     0.000 |
| t        | qwen    | livecodebench | static_3_1_4   | 21188 |        0.485 |     0.000 |
| t        | qwen    | livecodebench | static_6_10_60 | 12823 |        0.343 |     0.000 |
| t        | qwen    | livecodebench | static_7_1_8   | 20303 |        0.483 |     0.000 |
| t        | qwen    | livecodebench | static_7_4_8   | 17016 |        0.397 |     0.000 |
| t        | qwen    | math500       | static_3_1_4   | 20620 |        0.519 |     0.000 |
| t        | qwen    | math500       | static_6_10_60 | 12158 |        0.396 |     0.000 |
| t        | qwen    | math500       | static_7_1_8   | 19714 |        0.515 |     0.000 |
| t        | qwen    | math500       | static_7_4_8   | 16586 |        0.438 |     0.000 |
| t        | qwen    | mtbench       | static_3_1_4   | 26058 |        0.544 |     0.000 |
| t        | qwen    | mtbench       | static_6_10_60 | 14229 |        0.484 |     0.000 |
| t        | qwen    | mtbench       | static_7_1_8   | 18465 |        0.636 |     0.000 |
| t        | qwen    | mtbench       | static_7_4_8   | 18114 |        0.600 |     0.000 |

## 3. Threshold recommendations per cell

### Using DOG (V3 signal)

| model   | dataset       |   cheap_threshold |   premium_threshold | distinct_best_configs   |
|:--------|:--------------|------------------:|--------------------:|:------------------------|
| llama   | livecodebench |             0.828 |                 nan | ['static_3_1_4']        |
| llama   | math500       |             0.850 |                 nan | ['static_3_1_4']        |
| llama   | mtbench       |             0.754 |                 nan | ['static_3_1_4']        |
| qwen    | livecodebench |             0.392 |                 nan | ['static_3_1_4']        |
| qwen    | math500       |             0.446 |                 nan | ['static_3_1_4']        |
| qwen    | mtbench       |             0.877 |                 nan | ['static_3_1_4']        |

### Using t (V6 signal)

| model   | dataset       |   cheap_threshold |   premium_threshold | distinct_best_configs   |
|:--------|:--------------|------------------:|--------------------:|:------------------------|
| llama   | livecodebench |             0.996 |                 nan | ['static_3_1_4']        |
| llama   | math500       |             0.996 |                 nan | ['static_3_1_4']        |
| llama   | mtbench       |             0.986 |                 nan | ['static_3_1_4']        |
| qwen    | livecodebench |             0.966 |                 nan | ['static_3_1_4']        |
| qwen    | math500       |             0.981 |                 nan | ['static_3_1_4']        |
| qwen    | mtbench       |             1.000 |                 nan | ['static_3_1_4']        |

## 4. Where static makes mistakes (wasted-tail fractions)

| config         | model   | dataset       |   num_steps |   wasted_tail_fraction |   mean_al |   n_steps |
|:---------------|:--------|:--------------|------------:|-----------------------:|----------:|----------:|
| static_3_1_4   | llama   | livecodebench |           3 |                  0.332 |     2.538 |      3821 |
| static_6_10_60 | llama   | livecodebench |           6 |                  0.430 |     3.335 |      3205 |
| static_7_1_8   | llama   | livecodebench |           7 |                  0.699 |     3.022 |      3118 |
| static_7_4_8   | llama   | livecodebench |           7 |                  0.502 |     3.305 |      2882 |
| static_3_1_4   | llama   | math500       |           3 |                  0.236 |     2.750 |      3834 |
| static_6_10_60 | llama   | math500       |           6 |                  0.190 |     4.414 |      2802 |
| static_7_1_8   | llama   | math500       |           7 |                  0.584 |     3.460 |      3499 |
| static_7_4_8   | llama   | math500       |           7 |                  0.372 |     3.830 |      2878 |
| static_3_1_4   | llama   | mtbench       |           3 |                  0.251 |     2.726 |      4580 |
| static_6_10_60 | llama   | mtbench       |           6 |                  0.119 |     4.996 |      2472 |
| static_7_1_8   | llama   | mtbench       |           7 |                  0.603 |     3.494 |      3696 |
| static_7_4_8   | llama   | mtbench       |           7 |                  0.378 |     3.729 |      3539 |
| static_3_1_4   | qwen    | livecodebench |           3 |                  0.460 |     1.902 |     21188 |
| static_6_10_60 | qwen    | livecodebench |           6 |                  0.272 |     3.144 |     12823 |
| static_7_1_8   | qwen    | livecodebench |           7 |                  0.888 |     1.985 |     20303 |
| static_7_4_8   | qwen    | livecodebench |           7 |                  0.585 |     2.389 |     17016 |
| static_3_1_4   | qwen    | math500       |           3 |                  0.452 |     1.965 |     20620 |
| static_6_10_60 | qwen    | math500       |           6 |                  0.290 |     3.185 |     12158 |
| static_7_1_8   | qwen    | math500       |           7 |                  0.871 |     2.044 |     19714 |
| static_7_4_8   | qwen    | math500       |           7 |                  0.600 |     2.439 |     16586 |
| static_3_1_4   | qwen    | mtbench       |           3 |                  0.233 |     2.640 |     26058 |
| static_6_10_60 | qwen    | mtbench       |           6 |                  0.195 |     4.609 |     14229 |
| static_7_1_8   | qwen    | mtbench       |           7 |                  0.628 |     3.343 |     18465 |
| static_7_4_8   | qwen    | mtbench       |           7 |                  0.439 |     3.603 |     18114 |

## 5. Phase C scope

- KS = 0.441 → **per-model thresholds required**. Phase C should split the V6 constants by model.
