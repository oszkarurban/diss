# Phase B.2 — Follow-up analysis summary

## Per-cell Spearman r (signal = t)

| model   | dataset       | config         |   spearman_r |     n |
|:--------|:--------------|:---------------|-------------:|------:|
| llama   | livecodebench | static_3_1_4   |        0.723 |  3821 |
| llama   | livecodebench | static_6_10_60 |        0.724 |  3205 |
| llama   | livecodebench | static_7_1_8   |        0.719 |  3118 |
| llama   | livecodebench | static_7_4_8   |        0.701 |  2882 |
| llama   | math500       | static_3_1_4   |        0.550 |  3834 |
| llama   | math500       | static_6_10_60 |        0.422 |  2802 |
| llama   | math500       | static_7_1_8   |        0.593 |  3499 |
| llama   | math500       | static_7_4_8   |        0.564 |  2878 |
| llama   | mtbench       | static_3_1_4   |        0.530 |  4580 |
| llama   | mtbench       | static_6_10_60 |        0.403 |  2472 |
| llama   | mtbench       | static_7_1_8   |        0.541 |  3696 |
| llama   | mtbench       | static_7_4_8   |        0.483 |  3539 |
| qwen    | livecodebench | static_3_1_4   |        0.485 | 21188 |
| qwen    | livecodebench | static_6_10_60 |        0.343 | 12823 |
| qwen    | livecodebench | static_7_1_8   |        0.483 | 20303 |
| qwen    | livecodebench | static_7_4_8   |        0.397 | 17016 |
| qwen    | math500       | static_3_1_4   |        0.519 | 20620 |
| qwen    | math500       | static_6_10_60 |        0.396 | 12158 |
| qwen    | math500       | static_7_1_8   |        0.515 | 19714 |
| qwen    | math500       | static_7_4_8   |        0.438 | 16586 |
| qwen    | mtbench       | static_3_1_4   |        0.544 | 26058 |
| qwen    | mtbench       | static_6_10_60 |        0.484 | 14229 |
| qwen    | mtbench       | static_7_1_8   |        0.636 | 18465 |
| qwen    | mtbench       | static_7_4_8   |        0.600 | 18114 |

## Per-cell Spearman r (signal = DOG)

| model   | dataset       | config         |   spearman_r |     n |
|:--------|:--------------|:---------------|-------------:|------:|
| llama   | livecodebench | static_3_1_4   |        0.677 |  3821 |
| llama   | livecodebench | static_6_10_60 |        0.680 |  3205 |
| llama   | livecodebench | static_7_1_8   |        0.629 |  3118 |
| llama   | livecodebench | static_7_4_8   |        0.660 |  2882 |
| llama   | math500       | static_3_1_4   |        0.529 |  3834 |
| llama   | math500       | static_6_10_60 |        0.443 |  2802 |
| llama   | math500       | static_7_1_8   |        0.543 |  3499 |
| llama   | math500       | static_7_4_8   |        0.556 |  2878 |
| llama   | mtbench       | static_3_1_4   |        0.463 |  4580 |
| llama   | mtbench       | static_6_10_60 |        0.379 |  2472 |
| llama   | mtbench       | static_7_1_8   |        0.451 |  3696 |
| llama   | mtbench       | static_7_4_8   |        0.446 |  3539 |
| qwen    | livecodebench | static_3_1_4   |        0.375 | 21188 |
| qwen    | livecodebench | static_6_10_60 |        0.332 | 12823 |
| qwen    | livecodebench | static_7_1_8   |        0.361 | 20303 |
| qwen    | livecodebench | static_7_4_8   |        0.367 | 17016 |
| qwen    | math500       | static_3_1_4   |        0.410 | 20620 |
| qwen    | math500       | static_6_10_60 |        0.385 | 12158 |
| qwen    | math500       | static_7_1_8   |        0.385 | 19714 |
| qwen    | math500       | static_7_4_8   |        0.389 | 16586 |
| qwen    | mtbench       | static_3_1_4   |        0.538 | 26058 |
| qwen    | mtbench       | static_6_10_60 |        0.545 | 14229 |
| qwen    | mtbench       | static_7_1_8   |        0.611 | 18465 |
| qwen    | mtbench       | static_7_4_8   |        0.616 | 18114 |

## Per-model medians (supplementary, signal = t)

| model   |   median |   min |   max |   n_cells |
|:--------|---------:|------:|------:|----------:|
| llama   |    0.557 | 0.403 | 0.724 |        12 |
| qwen    |    0.485 | 0.343 | 0.636 |        12 |

## Per-dataset medians (supplementary, signal = t)

| dataset       |   median |   min |   max |   n_cells |
|:--------------|---------:|------:|------:|----------:|
| livecodebench |    0.593 | 0.343 | 0.724 |         8 |
| math500       |    0.517 | 0.396 | 0.593 |         8 |
| mtbench       |    0.535 | 0.403 | 0.636 |         8 |

## Per-cell thresholds derived from Plot-5 v2 (signal = t)

| model   | dataset       |   cheap_threshold |   premium_threshold | distinct_best_configs                                              |   n_cheap_deciles |   n_premium_deciles |
|:--------|:--------------|------------------:|--------------------:|:-------------------------------------------------------------------|------------------:|--------------------:|
| llama   | livecodebench |             0.816 |               0.334 | ['static_3_1_4', 'static_6_10_60', 'static_7_1_8', 'static_7_4_8'] |                 3 |                   5 |
| llama   | math500       |           nan     |               0.339 | ['static_6_10_60', 'static_7_1_8', 'static_7_4_8']                 |                 0 |                   9 |
| llama   | mtbench       |           nan     |               0.350 | ['static_6_10_60', 'static_7_1_8']                                 |                 0 |                   8 |
| qwen    | livecodebench |           nan     |               0.349 | ['static_6_10_60']                                                 |                 0 |                  10 |
| qwen    | math500       |           nan     |               0.354 | ['static_6_10_60']                                                 |                 0 |                  10 |
| qwen    | mtbench       |           nan     |               0.378 | ['static_6_10_60', 'static_7_1_8']                                 |                 0 |                   9 |

## Policy simulation — expected throughput per cell

| model   | dataset       |   policy_cheap_threshold |   policy_premium_threshold | use_ideal_switching   |   expected_tput_tok_s | variant           |   gain_vs_global_pct |
|:--------|:--------------|-------------------------:|---------------------------:|:----------------------|----------------------:|:------------------|---------------------:|
| llama   | livecodebench |                     0.30 |                       0.95 | False                 |                158.93 | v6_current_global |                 0.00 |
| llama   | math500       |                     0.30 |                       0.95 | False                 |                177.42 | v6_current_global |                 0.00 |
| llama   | mtbench       |                     0.30 |                       0.95 | False                 |                184.40 | v6_current_global |                 0.00 |
| qwen    | livecodebench |                     0.30 |                       0.95 | False                 |                122.13 | v6_current_global |                 0.00 |
| qwen    | math500       |                     0.30 |                       0.95 | False                 |                126.58 | v6_current_global |                 0.00 |
| qwen    | mtbench       |                     0.30 |                       0.95 | False                 |                177.12 | v6_current_global |                 0.00 |
| llama   | livecodebench |                     0.82 |                       0.34 | False                 |                162.01 | per_model_llama   |                 1.94 |
| llama   | math500       |                     0.82 |                       0.34 | False                 |                170.82 | per_model_llama   |                -3.72 |
| llama   | mtbench       |                     0.82 |                       0.34 | False                 |                187.18 | per_model_llama   |                 1.51 |
| qwen    | livecodebench |                   nan    |                       0.35 | False                 |                142.33 | per_model_qwen    |                16.55 |
| qwen    | math500       |                   nan    |                       0.35 | False                 |                149.84 | per_model_qwen    |                18.37 |
| qwen    | mtbench       |                   nan    |                       0.35 | False                 |                204.03 | per_model_qwen    |                15.19 |
| llama   | livecodebench |                   nan    |                     nan    | True                  |                163.14 | ideal_per_decile  |                 2.65 |
| llama   | math500       |                   nan    |                     nan    | True                  |                192.88 | ideal_per_decile  |                 8.72 |
| llama   | mtbench       |                   nan    |                     nan    | True                  |                203.36 | ideal_per_decile  |                10.28 |
| qwen    | livecodebench |                   nan    |                     nan    | True                  |                143.98 | ideal_per_decile  |                17.89 |
| qwen    | math500       |                   nan    |                     nan    | True                  |                149.84 | ideal_per_decile  |                18.37 |
| qwen    | mtbench       |                   nan    |                     nan    | True                  |                214.14 | ideal_per_decile  |                20.90 |

Interpretation: compare `per_model_*` rows vs `v6_current_global`. 
Gains > 3% on a cell justify per-model tuning for Phase C. The 
`ideal_per_decile` row is an UPPER BOUND (requires oracle) — the 
gap between per_model and ideal is the remaining unexplained gain.

## Adaptive-quantile feasibility

| model   | dataset       |   cheap_threshold |   cheap_percentile |   premium_threshold |   premium_percentile |
|:--------|:--------------|------------------:|-------------------:|--------------------:|---------------------:|
| llama   | livecodebench |             0.816 |              0.548 |               0.334 |                0.150 |
| llama   | math500       |           nan     |            nan     |               0.339 |                0.047 |
| llama   | mtbench       |           nan     |            nan     |               0.350 |                0.044 |
| qwen    | livecodebench |           nan     |            nan     |               0.349 |                0.042 |
| qwen    | math500       |           nan     |            nan     |               0.354 |                0.044 |
| qwen    | mtbench       |           nan     |            nan     |               0.378 |                0.042 |

Interpretation: if `cheap_percentile` is tightly clustered across the 6 cells (e.g. all in 0.25–0.40), a fixed rolling-quantile threshold can replace per-model constants. If values are spread > 0.2 apart, adaptive quantiles won't generalise.

## V3 capability

| model   | dataset       |   n_steps |   frac_at_max | ideal_configs_observed                                             | all_ideal_within_v3_range   |
|:--------|:--------------|----------:|--------------:|:-------------------------------------------------------------------|:----------------------------|
| llama   | livecodebench |      2833 |         0.000 | ['static_3_1_4', 'static_6_10_60', 'static_7_1_8', 'static_7_4_8'] | False                       |
| llama   | math500       |      3102 |         0.000 | ['static_6_10_60', 'static_7_1_8', 'static_7_4_8']                 | False                       |
| llama   | mtbench       |      3397 |         0.000 | ['static_6_10_60', 'static_7_1_8']                                 | False                       |

Interpretation: `all_ideal_within_v3_range=False` means V3's ceiling of (7,4,8) is too low to reach that cell's optimal config. Phase C should widen V3's max for those cells.

## V3 / V6 actual pick vs ideal (match rate)

| policy     | model   | dataset       |   match_rate |   n_deciles |
|:-----------|:--------|:--------------|-------------:|------------:|
| v3_dynamic | llama   | livecodebench |        0.200 |          10 |
| v3_dynamic | llama   | math500       |        0.000 |          10 |
| v3_dynamic | llama   | mtbench       |        0.200 |          10 |
| v6_dynamic | llama   | livecodebench |        0.100 |          10 |
| v6_dynamic | llama   | math500       |        0.000 |          10 |
| v6_dynamic | llama   | mtbench       |        0.000 |          10 |
| v6_dynamic | qwen    | livecodebench |        0.000 |          10 |
| v6_dynamic | qwen    | math500       |        0.000 |          10 |
| v6_dynamic | qwen    | mtbench       |        0.000 |          10 |

## Phase C recommendation (prioritised)

| Action | Effort | Gain signal |
|---|---|---|
| V6 per-model thresholds | ~20 LOC | See §Policy simulation table |
| V6 adaptive quantile thresholds | ~30 LOC | Depends on §Adaptive feasibility |
| V6 higher max bound (→ 10,10,60 on Llama) | 0 LOC | Lift ceiling above best static |
| V3 wider max bound (→ 6,10,60) | 0 LOC | V3 cannot reach best static otherwise |

Ranking depends on the numerical tables above. Pick the top 2–3 for Phase C runs.
