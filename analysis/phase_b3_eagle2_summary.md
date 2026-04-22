# Phase B.3 — EAGLE-2-style analysis summary

Central question (from EAGLE-2 paper Figure 6): does signal X linearly predict acceptance, and at what confidence intervals does each tree shape pay off?

## 1. Signal linearity ranking (median R² across 24 cell × config)

| signal           |   median_r2 |   median_slope |
|:-----------------|------------:|---------------:|
| t                |       0.936 |          0.629 |
| DOG              |       0.932 |          0.653 |
| top1_prob        |       0.929 |          0.462 |
| target_top1_prob |       0.894 |          0.476 |

**Winning signal: `t`** (highest median R²).

Per-cell linearity scores:

| model   | dataset       | config         |   slope_a |   intercept_b |   r_squared |   pearson_r |   n_bins_valid | signal           |
|:--------|:--------------|:---------------|----------:|--------------:|------------:|------------:|---------------:|:-----------------|
| llama   | livecodebench | static_3_1_4   |     0.727 |         0.273 |       0.890 |       0.943 |             10 | top1_prob        |
| llama   | livecodebench | static_6_10_60 |     0.527 |         0.195 |       0.855 |       0.925 |             10 | top1_prob        |
| llama   | livecodebench | static_7_1_8   |     0.447 |         0.085 |       0.756 |       0.870 |             10 | top1_prob        |
| llama   | livecodebench | static_7_4_8   |     0.439 |         0.125 |       0.778 |       0.882 |             10 | top1_prob        |
| llama   | math500       | static_3_1_4   |     0.687 |         0.370 |       0.958 |       0.979 |             10 | top1_prob        |
| llama   | math500       | static_6_10_60 |     0.555 |         0.324 |       0.892 |       0.945 |             10 | top1_prob        |
| llama   | math500       | static_7_1_8   |     0.454 |         0.126 |       0.925 |       0.962 |             10 | top1_prob        |
| llama   | math500       | static_7_4_8   |     0.465 |         0.178 |       0.943 |       0.971 |             10 | top1_prob        |
| llama   | mtbench       | static_3_1_4   |     0.728 |         0.374 |       0.974 |       0.987 |             10 | top1_prob        |
| llama   | mtbench       | static_6_10_60 |     0.585 |         0.428 |       0.908 |       0.953 |             10 | top1_prob        |
| llama   | mtbench       | static_7_1_8   |     0.493 |         0.144 |       0.970 |       0.985 |             10 | top1_prob        |
| llama   | mtbench       | static_7_4_8   |     0.459 |         0.205 |       0.986 |       0.993 |             10 | top1_prob        |
| qwen    | livecodebench | static_3_1_4   |     0.479 |         0.338 |       0.943 |       0.971 |             10 | top1_prob        |
| qwen    | livecodebench | static_6_10_60 |     0.339 |         0.310 |       0.912 |       0.955 |             10 | top1_prob        |
| qwen    | livecodebench | static_7_1_8   |     0.232 |         0.141 |       0.937 |       0.968 |             10 | top1_prob        |
| qwen    | livecodebench | static_7_4_8   |     0.238 |         0.191 |       0.955 |       0.977 |             10 | top1_prob        |
| qwen    | math500       | static_3_1_4   |     0.491 |         0.326 |       0.921 |       0.960 |             10 | top1_prob        |
| qwen    | math500       | static_6_10_60 |     0.340 |         0.309 |       0.978 |       0.989 |             10 | top1_prob        |
| qwen    | math500       | static_7_1_8   |     0.234 |         0.134 |       0.897 |       0.947 |             10 | top1_prob        |
| qwen    | math500       | static_7_4_8   |     0.247 |         0.183 |       0.945 |       0.972 |             10 | top1_prob        |
| qwen    | mtbench       | static_3_1_4   |     0.665 |         0.320 |       0.965 |       0.982 |             10 | top1_prob        |
| qwen    | mtbench       | static_6_10_60 |     0.576 |         0.299 |       0.930 |       0.964 |             10 | top1_prob        |
| qwen    | mtbench       | static_7_1_8   |     0.416 |         0.111 |       0.828 |       0.910 |             10 | top1_prob        |
| qwen    | mtbench       | static_7_4_8   |     0.454 |         0.141 |       0.928 |       0.963 |             10 | top1_prob        |
| llama   | livecodebench | static_3_1_4   |     0.482 |         0.422 |       0.805 |       0.897 |              8 | target_top1_prob |
| llama   | livecodebench | static_6_10_60 |     1.017 |        -0.051 |       0.923 |       0.961 |              8 | target_top1_prob |
| llama   | livecodebench | static_7_1_8   |     0.370 |         0.164 |       0.638 |       0.799 |              8 | target_top1_prob |
| llama   | livecodebench | static_7_4_8   |     0.503 |         0.078 |       0.941 |       0.970 |              8 | target_top1_prob |
| llama   | math500       | static_3_1_4   |     0.413 |         0.536 |       0.742 |       0.862 |              7 | target_top1_prob |
| llama   | math500       | static_6_10_60 |     0.453 |         0.448 |       0.442 |       0.665 |              7 | target_top1_prob |
| llama   | math500       | static_7_1_8   |     0.373 |         0.188 |       0.673 |       0.820 |              8 | target_top1_prob |
| llama   | math500       | static_7_4_8   |     0.582 |         0.057 |       0.902 |       0.950 |              6 | target_top1_prob |
| llama   | mtbench       | static_3_1_4   |     0.734 |         0.315 |       0.977 |       0.988 |              7 | target_top1_prob |
| llama   | mtbench       | static_6_10_60 |     0.860 |         0.168 |       0.975 |       0.987 |              6 | target_top1_prob |
| llama   | mtbench       | static_7_1_8   |     0.664 |        -0.020 |       0.944 |       0.972 |              6 | target_top1_prob |
| llama   | mtbench       | static_7_4_8   |     0.471 |         0.163 |       0.898 |       0.948 |              7 | target_top1_prob |
| qwen    | livecodebench | static_3_1_4   |     0.270 |         0.412 |       0.833 |       0.913 |              7 | target_top1_prob |
| qwen    | livecodebench | static_6_10_60 |     0.291 |         0.299 |       0.973 |       0.986 |              5 | target_top1_prob |
| qwen    | livecodebench | static_7_1_8   |     0.228 |         0.109 |       0.889 |       0.943 |              7 | target_top1_prob |
| qwen    | livecodebench | static_7_4_8   |     0.157 |         0.213 |       0.873 |       0.934 |              7 | target_top1_prob |
| qwen    | math500       | static_3_1_4   |     0.360 |         0.335 |       0.838 |       0.915 |              7 | target_top1_prob |
| qwen    | math500       | static_6_10_60 |     0.561 |         0.096 |       0.932 |       0.965 |              5 | target_top1_prob |
| qwen    | math500       | static_7_1_8   |     0.218 |         0.117 |       0.879 |       0.938 |              7 | target_top1_prob |
| qwen    | math500       | static_7_4_8   |     0.262 |         0.126 |       0.906 |       0.952 |              6 | target_top1_prob |
| qwen    | mtbench       | static_3_1_4   |     0.708 |         0.231 |       0.954 |       0.977 |              7 | target_top1_prob |
| qwen    | mtbench       | static_6_10_60 |     0.899 |         0.049 |       0.876 |       0.936 |              6 | target_top1_prob |
| qwen    | mtbench       | static_7_1_8   |     0.631 |        -0.055 |       0.769 |       0.877 |              6 | target_top1_prob |
| qwen    | mtbench       | static_7_4_8   |     0.749 |        -0.131 |       0.926 |       0.963 |              7 | target_top1_prob |
| llama   | livecodebench | static_3_1_4   |     0.931 |         0.114 |       0.888 |       0.942 |              9 | t                |
| llama   | livecodebench | static_6_10_60 |     0.831 |         0.028 |       0.896 |       0.947 |              9 | t                |
| llama   | livecodebench | static_7_1_8   |     0.642 |        -0.028 |       0.806 |       0.898 |              9 | t                |
| llama   | livecodebench | static_7_4_8   |     0.617 |         0.007 |       0.832 |       0.912 |              9 | t                |
| llama   | math500       | static_3_1_4   |     0.854 |         0.217 |       0.957 |       0.978 |              9 | t                |
| llama   | math500       | static_6_10_60 |     0.742 |         0.179 |       0.937 |       0.968 |              9 | t                |
| llama   | math500       | static_7_1_8   |     0.577 |         0.032 |       0.918 |       0.958 |              9 | t                |
| llama   | math500       | static_7_4_8   |     0.633 |         0.040 |       0.935 |       0.967 |              8 | t                |
| llama   | mtbench       | static_3_1_4   |     0.767 |         0.322 |       0.797 |       0.893 |              9 | t                |
| llama   | mtbench       | static_6_10_60 |     0.752 |         0.283 |       0.926 |       0.962 |              8 | t                |
| llama   | mtbench       | static_7_1_8   |     0.664 |         0.007 |       0.952 |       0.976 |              8 | t                |
| llama   | mtbench       | static_7_4_8   |     0.624 |         0.069 |       0.985 |       0.992 |              8 | t                |
| qwen    | livecodebench | static_3_1_4   |     0.626 |         0.215 |       0.952 |       0.976 |              9 | t                |
| qwen    | livecodebench | static_6_10_60 |     0.458 |         0.216 |       0.967 |       0.983 |              8 | t                |
| qwen    | livecodebench | static_7_1_8   |     0.254 |         0.122 |       0.782 |       0.884 |              9 | t                |
| qwen    | livecodebench | static_7_4_8   |     0.322 |         0.122 |       0.957 |       0.978 |              8 | t                |
| qwen    | math500       | static_3_1_4   |     0.632 |         0.201 |       0.933 |       0.966 |              9 | t                |
| qwen    | math500       | static_6_10_60 |     0.464 |         0.210 |       0.974 |       0.987 |              8 | t                |
| qwen    | math500       | static_7_1_8   |     0.310 |         0.075 |       0.912 |       0.955 |              9 | t                |
| qwen    | math500       | static_7_4_8   |     0.340 |         0.106 |       0.940 |       0.970 |              8 | t                |
| qwen    | mtbench       | static_3_1_4   |     0.839 |         0.157 |       0.959 |       0.979 |              9 | t                |
| qwen    | mtbench       | static_6_10_60 |     0.845 |         0.084 |       0.973 |       0.986 |              9 | t                |
| qwen    | mtbench       | static_7_1_8   |     0.560 |         0.002 |       0.858 |       0.926 |              9 | t                |
| qwen    | mtbench       | static_7_4_8   |     0.596 |         0.018 |       0.936 |       0.968 |              9 | t                |
| llama   | livecodebench | static_3_1_4   |     0.917 |         0.495 |       0.933 |       0.966 |             10 | DOG              |
| llama   | livecodebench | static_6_10_60 |    -0.989 |         0.812 |       0.266 |      -0.516 |              4 | DOG              |
| llama   | livecodebench | static_7_1_8   |     0.889 |         0.258 |       0.917 |       0.958 |             10 | DOG              |
| llama   | livecodebench | static_7_4_8   |     0.879 |         0.260 |       0.957 |       0.978 |             10 | DOG              |
| llama   | math500       | static_3_1_4   |     0.773 |         0.554 |       0.966 |       0.983 |             10 | DOG              |
| llama   | math500       | static_6_10_60 |    -0.442 |         0.839 |       0.075 |      -0.273 |              3 | DOG              |
| llama   | math500       | static_7_1_8   |     0.802 |         0.269 |       0.983 |       0.991 |             10 | DOG              |
| llama   | math500       | static_7_4_8   |     0.763 |         0.300 |       0.977 |       0.989 |             10 | DOG              |
| llama   | mtbench       | static_3_1_4   |     0.733 |         0.586 |       0.946 |       0.973 |             10 | DOG              |
| llama   | mtbench       | static_6_10_60 |     0.132 |         0.878 |       0.107 |       0.328 |              5 | DOG              |
| llama   | mtbench       | static_7_1_8   |     0.559 |         0.355 |       0.839 |       0.916 |              9 | DOG              |
| llama   | mtbench       | static_7_4_8   |     0.610 |         0.353 |       0.943 |       0.971 |             10 | DOG              |
| qwen    | livecodebench | static_3_1_4   |     0.587 |         0.547 |       0.870 |       0.933 |             10 | DOG              |
| qwen    | livecodebench | static_6_10_60 |    -0.700 |         0.522 |       0.918 |      -0.958 |              5 | DOG              |
| qwen    | livecodebench | static_7_1_8   |     0.626 |         0.235 |       0.983 |       0.992 |              7 | DOG              |
| qwen    | livecodebench | static_7_4_8   |     0.680 |         0.257 |       0.991 |       0.996 |              8 | DOG              |
| qwen    | math500       | static_3_1_4   |     0.714 |         0.502 |       0.957 |       0.978 |             10 | DOG              |
| qwen    | math500       | static_6_10_60 |    -1.055 |         0.635 |       0.719 |      -0.848 |              4 | DOG              |
| qwen    | math500       | static_7_1_8   |     0.422 |         0.274 |       0.773 |       0.879 |              7 | DOG              |
| qwen    | math500       | static_7_4_8   |     0.501 |         0.284 |       0.909 |       0.953 |              7 | DOG              |
| qwen    | mtbench       | static_3_1_4   |     0.761 |         0.530 |       0.969 |       0.985 |             10 | DOG              |
| qwen    | mtbench       | static_6_10_60 |    -0.761 |         0.978 |       0.356 |      -0.596 |              5 | DOG              |
| qwen    | mtbench       | static_7_1_8   |     0.826 |         0.252 |       0.931 |       0.965 |             10 | DOG              |
| qwen    | mtbench       | static_7_4_8   |     0.828 |         0.261 |       0.995 |       0.998 |             10 | DOG              |

## 2. Interpretation

- **R² close to 1.0 with slope ≈ 1**: signal is a near-perfect linear predictor of acceptance rate — matches EAGLE-2 Figure 6 for Vicuna-7B. This justifies using the signal to gate tree depth.
- **R² close to 1.0 but slope < 1**: signal is MONOTONIC but acceptance grows slower than confidence. Means tree-shape decisions should weight signal less aggressively at the top.
- **R² < 0.5**: signal poorly predicts acceptance on that cell. Policy based on this signal will be noisy — may need hysteresis or a different signal.

## 3. Per-depth cumulative acceptance (chain_7_1_8)

See `plot_e2b_depth_cumulative_top1_prob.png`. For each signal bin, curves show P(depth k accepted) vs k=0..6. Steep drop at low k = chain breaks early (depth wasted). Flat near 1.0 up to high k = chain is paying off.

**Policy reading**:
- Bins where depth-6 accept prob > 0.5 → deep chain justified
- Bins where depth-3 accept prob < 0.5 → switch to short chain
- Bins where even depth-1 drops below 0.3 → skip speculation / use minimal config (the ORIGINAL V6 CHEAP zone rationale)

## 4. Wide vs deep (chain 7,1,8 vs tree 7,4,8 at same depth)

| model   | dataset       |   max_delta_tree_minus_chain |   bin_of_max |   mean_delta |
|:--------|:--------------|-----------------------------:|-------------:|-------------:|
| llama   | livecodebench |                        0.312 |        0.150 |        0.152 |
| llama   | math500       |                        0.548 |        0.250 |        0.344 |
| llama   | mtbench       |                        0.517 |        0.250 |        0.263 |
| qwen    | livecodebench |                        0.467 |        0.250 |        0.347 |
| qwen    | math500       |                        0.464 |        0.250 |        0.359 |
| qwen    | mtbench       |                        0.487 |        0.150 |        0.253 |

**Interpretation**: positive delta (green shading in plot) = tree branching at each depth adds AL beyond what a single chain delivers. Typically helps at LOW confidence (redundant candidates protect against single-chain failures) and stops helping at HIGH confidence (single chain already accepts every depth).

## 5. Transition points (where optimal config changes)

| model   | dataset       | signal   | bin_best_config_by_bin                                                                                                                                                 | transitions                                                                                                                                                                                   |
|:--------|:--------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| llama   | livecodebench | t        | [None, 'static_3_1_4', 'static_3_1_4', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_3_1_4', 'static_7_1_8', 'static_7_1_8']         | [(np.float64(0.35000000000000003), 'static_3_1_4', 'static_6_10_60'), (np.float64(0.75), 'static_6_10_60', 'static_3_1_4'), (np.float64(0.8500000000000001), 'static_3_1_4', 'static_7_1_8')] |
| llama   | math500       | t        | [None, 'static_3_1_4', 'static_7_4_8', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_7_4_8', 'static_7_4_8']       | [(np.float64(0.25), 'static_3_1_4', 'static_7_4_8'), (np.float64(0.35000000000000003), 'static_7_4_8', 'static_6_10_60'), (np.float64(0.8500000000000001), 'static_6_10_60', 'static_7_4_8')] |
| llama   | mtbench       | t        | [None, None, 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_7_1_8']             | [(np.float64(0.95), 'static_6_10_60', 'static_7_1_8')]                                                                                                                                        |
| qwen    | livecodebench | t        | [None, None, 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60']           | []                                                                                                                                                                                            |
| qwen    | math500       | t        | [None, None, 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60']           | []                                                                                                                                                                                            |
| qwen    | mtbench       | t        | [None, 'static_3_1_4', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60', 'static_6_10_60'] | [(np.float64(0.25), 'static_3_1_4', 'static_6_10_60')]                                                                                                                                        |

These are the concrete "switch here" signal values per cell. V6 / V3 threshold decisions should coincide with these.

## 6. Synthesis — data-driven policy recommendations

Based on sections 1-5 above, recommended policy structure:

| Signal zone | Config | Rationale |
|---|---|---|
| Very low signal (top1 < P1) | (1,1,2) minimal or skip | Chain breaks at depth 0-1 |
| Low signal (P1 < top1 < P2) | (3,1,4) short chain | Chain OK to ~3 depth; branching doesn't help |
| Mid signal (P2 < top1 < P3) | (7,4,8) tree | Chain starts breaking beyond depth 3; branching recovers |
| High signal (top1 > P3) | (7,1,8) or (10,6,60) deep | Chain reliable; depth pays off directly |

Actual values of P1, P2, P3 read from plot_e2b transition points.

## 7. Files produced

- `plot_e2a_fig6_{top1_prob,target_top1_prob,t,DOG}.{png,pdf}` — Fig 6 replications
- `plot_e2b_depth_cumulative_top1_prob.{png,pdf}` — per-depth curves
- `plot_e2c_wide_vs_deep_top1_prob.{png,pdf}` — tree-vs-chain crossover
- `tables/e2_linearity_scores.csv` — R² per cell × config × signal
- `tables/e2_binned_accept_{signal}.csv` — raw binned data
- `tables/e2_depth_cumulative_*.csv` — per-depth data
- `tables/e2_wide_vs_deep_*.csv` — tree vs chain data
- `tables/e2_transition_points.csv` — where best config changes