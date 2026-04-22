# Static EAGLE3 SGLang Baselines

Reference table of all static (vanilla) EAGLE3 speculative-decoding baselines collected from `hpc/bench_{llama,deepseek,qwen}{,_chains,_trees}.sh`.

- **Hardware**: Cambridge Wilkes3 A100, `ampere` partition, `batch_size=1`, `temperature=0`, `--dtype bfloat16`, `--max-running-requests 1`
- **Config notation**: `(topk, num_steps, num_draft_tokens)`
- **Metric format**: `throughput_tok/s / accept_length`
- **Sweep window**: April 13–16 2026 (SGLang HEAD `8dc32bbb8`, vanilla EAGLE3 server — no dynamic policy)
- **Selection rule**: best (max) throughput across available runs per cell

## Llama-3.1-8B-Instruct (matched draft: `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`)

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| nospec | 88.48 / 1.00 | 87.11 / 1.00 | 88.76 / 1.00 | 88.60 / 1.00 | 87.33 / 1.00 |
| (3,1,4) | 176.97 / 2.93 | 101.59 / 1.68 | 174.27 / 2.79 | 189.75 / 3.05 | 148.18 / 2.43 |
| (7,1,8) | 204.02 / 3.94 | 88.78 / 1.74 | 176.39 / 3.37 | 212.80 / 4.10 | 143.01 / 2.80 |
| (7,4,8) | 216.12 / 4.28 | 100.68 / 2.02 | 195.15 / 3.81 | **224.95 / 4.43** | 156.17 / 3.13 |
| (10,6,60) | **219.44 / 6.07** | 86.64 / 2.42 | 175.94 / 4.80 | 220.83 / 6.09 | 138.81 / 3.89 |

**Llama SOTA targets:** mtbench → `(10,6,60)` @ 219.44, humaneval → `(7,4,8)` @ 224.95, livecodebench → `(7,4,8)` @ 156.17.
GSM8k pattern is unusual — throughput barely scales with tree size because the draft model generates long chains-of-thought that the target rejects at low rates; smaller trees win.

### Llama source files

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| nospec | [141732](results/llama_vanilla_nospec_20260415_135244_results_20260415_141732.jsonl) | [142145](results/llama_vanilla_nospec_20260415_135244_results_20260415_142145.jsonl) | [144318](results/llama_vanilla_nospec_20260415_135244_results_20260415_144318.jsonl) | [145510](results/llama_vanilla_nospec_20260415_135244_results_20260415_145510.jsonl) | [150712](results/llama_vanilla_nospec_20260415_135244_results_20260415_150712.jsonl) |
| (3,1,4) | [235304](results/llama_vanilla_314_20260413_233232_results_20260413_235304.jsonl) | [235634](results/llama_vanilla_314_20260413_233232_results_20260413_235634.jsonl) | [000838](results/llama_vanilla_314_20260413_233232_results_20260414_000838.jsonl) | [001431](results/llama_vanilla_314_20260413_233232_results_20260414_001431.jsonl) | [002150](results/llama_vanilla_314_20260413_233232_results_20260414_002150.jsonl) |
| (7,1,8) | [002829](results/llama_vanilla_718_20260413_233232_results_20260414_002829.jsonl) | [003156](results/llama_vanilla_718_20260413_233232_results_20260414_003156.jsonl) | [004300](results/llama_vanilla_718_20260413_233232_results_20260414_004300.jsonl) | [004832](results/llama_vanilla_718_20260413_233232_results_20260414_004832.jsonl) | [005549](results/llama_vanilla_718_20260413_233232_results_20260414_005549.jsonl) |
| (7,4,8) | [145212](results/llama_vanilla_748_20260415_143440_results_20260415_145212.jsonl) | [145609](results/llama_vanilla_748_20260415_143440_results_20260415_145609.jsonl) | [150637](results/llama_vanilla_748_20260415_143440_results_20260415_150637.jsonl) | [151144](results/llama_vanilla_748_20260415_143440_results_20260415_151144.jsonl) | [151843](results/llama_vanilla_748_20260415_143440_results_20260415_151843.jsonl) |
| (10,6,60) | [004723](results/llama_vanilla_10_6_60_20260413_235434_results_20260414_004723.jsonl) | [153026](results/llama_vanilla_10_6_60_20260415_143440_results_20260415_153026.jsonl) | [154108](results/llama_vanilla_10_6_60_20260415_143440_results_20260415_154108.jsonl) | [154712](results/llama_vanilla_10_6_60_20260415_143440_results_20260415_154712.jsonl) | [011432](results/llama_vanilla_10_6_60_20260413_235434_results_20260414_011432.jsonl) |

## Qwen3-8B (matched draft: `AngelSlim/Qwen3-8B_eagle3`)

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| (3,1,4) | 160.53 / 2.59 | 122.17 / 2.00 | 123.18 / 1.95 | 122.82 / 1.94 | 119.12 / 1.90 |
| (7,1,8) | 158.17 / 3.03 | 105.56 / 2.06 | 107.45 / 2.03 | 106.98 / 2.02 | 103.36 / 1.96 |
| (7,4,8) | 169.39 / 3.35 | 117.59 / 2.35 | 124.59 / 2.42 | 125.21 / 2.43 | 121.69 / 2.37 |
| (10,6,60) | **178.46 / 4.40** | 119.02 / 2.94 | 128.05 / 3.07 | **131.54 / 3.12** | 125.15 / 3.03 |

**Qwen SOTA targets:** mtbench → `(10,6,60)` @ 178.46, reasoning/code benchmarks → `(10,6,60)` or `(3,1,4)` (tight — within 5 tok/s across all configs).
Qwen has no nospec baseline collected.

### Qwen source files

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| (3,1,4) | [003700](results/qwen_vanilla_314_20260413_235920_results_20260414_003700.jsonl) | [004053](results/qwen_vanilla_314_20260413_235920_results_20260414_004053.jsonl) | [013421](results/qwen_vanilla_314_20260413_235920_results_20260414_013421.jsonl) | [015745](results/qwen_vanilla_314_20260413_235920_results_20260414_015745.jsonl) | [025333](results/qwen_vanilla_314_20260413_235920_results_20260414_025333.jsonl) |
| (7,1,8) | [032318](results/qwen_vanilla_718_20260413_235920_results_20260414_032318.jsonl) | [032719](results/qwen_vanilla_718_20260413_235920_results_20260414_032719.jsonl) | [042754](results/qwen_vanilla_718_20260413_235920_results_20260414_042754.jsonl) | [045429](results/qwen_vanilla_718_20260413_235920_results_20260414_045429.jsonl) | [055919](results/qwen_vanilla_718_20260413_235920_results_20260414_055919.jsonl) |
| (7,4,8) | [003516](results/qwen_vanilla_748_20260413_235921_results_20260414_003516.jsonl) | [003922](results/qwen_vanilla_748_20260413_235921_results_20260414_003922.jsonl) | [013218](results/qwen_vanilla_748_20260413_235921_results_20260414_013218.jsonl) | [015521](results/qwen_vanilla_748_20260413_235921_results_20260414_015521.jsonl) | [025025](results/qwen_vanilla_748_20260413_235921_results_20260414_025025.jsonl) |
| (10,6,60) | [031731](results/qwen_vanilla_10_6_60_20260413_235921_results_20260414_031731.jsonl) | [032117](results/qwen_vanilla_10_6_60_20260413_235921_results_20260414_032117.jsonl) | [041252](results/qwen_vanilla_10_6_60_20260413_235921_results_20260414_041252.jsonl) | [043508](results/qwen_vanilla_10_6_60_20260413_235921_results_20260414_043508.jsonl) | [052821](results/qwen_vanilla_10_6_60_20260413_235921_results_20260414_052821.jsonl) |

## DeepSeek-R1-Distill-Llama-8B (matched draft: `yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B`)

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| nospec | 87.92 / 1.00 | 87.26 / 1.00 | 87.50 / 1.00 | 88.26 / 1.00 | 87.40 / 1.00 |
| (3,1,4) | 69.71 / 1.55 † | 72.76 / 1.63 | 147.61 / 3.31 | 135.70 / 3.00 | 131.39 / 2.84 |
| (7,1,8) | 58.64 / 1.59 † | 63.55 / 1.70 | 187.42 / 5.00 | 155.55 / 4.10 | 140.14 / 3.69 |
| (7,4,8) | 69.36 / 1.55 † | 73.50 / 1.64 | 146.95 / 3.31 | 135.45 / 3.00 | 132.04 / 2.85 |
| (10,6,60) | 59.78 / 1.61 † | 63.57 / 1.69 | **188.77 / 5.01** | **155.33 / 4.09** | **141.17 / 3.71** |

**DeepSeek SOTA targets:** mtbench/gsm8k → `(3,1,4)` ≈ `(7,4,8)` (short chats don't benefit from trees), reasoning/code → `(10,6,60)` or `(7,1,8)` (long CoT responses benefit from deep trees).

**† Environmental regression caveat**: historical April-9 DeepSeek mtbench numbers under the same static configs were ~118 tok/s. The April-13/14 reruns on HEAD `8dc32bbb8` dropped to 58–70 tok/s for mtbench while reasoning benchmarks (math500/humaneval/livecodebench) match or exceed historical levels. Cause unknown — same sglang commit, same deps, same model cache. **Treat mtbench DeepSeek static as suspect; trust the reasoning-benchmark numbers.**

### DeepSeek (matched) source files

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| nospec | [002457](results/deepseek_vanilla_nospec_20260414_234119_results_20260415_002457.jsonl) | [003022](results/deepseek_vanilla_nospec_20260414_234119_results_20260415_003022.jsonl) | [013550](results/deepseek_vanilla_nospec_20260414_234119_results_20260415_013550.jsonl) | [023027](results/deepseek_vanilla_nospec_20260414_234119_results_20260415_023027.jsonl) | [151854](results/deepseek_vanilla_nospec_20260415_135244_results_20260415_151854.jsonl) |
| (3,1,4) | [000151](results/deepseek_vanilla_314_20260413_230844_results_20260414_000151.jsonl) | [000807](results/deepseek_vanilla_314_20260413_230844_results_20260414_000807.jsonl) | [004646](results/deepseek_vanilla_314_20260413_230844_results_20260414_004646.jsonl) | [010717](results/deepseek_vanilla_314_20260413_230844_results_20260414_010717.jsonl) | [015626](results/deepseek_vanilla_314_20260413_230844_results_20260414_015626.jsonl) |
| (7,1,8) | [024200](results/deepseek_vanilla_718_20260413_230844_results_20260414_024200.jsonl) | [024920](results/deepseek_vanilla_718_20260413_230844_results_20260414_024920.jsonl) | [032006](results/deepseek_vanilla_718_20260413_230844_results_20260414_032006.jsonl) | [033902](results/deepseek_vanilla_718_20260413_230844_results_20260414_033902.jsonl) | [042513](results/deepseek_vanilla_718_20260413_230844_results_20260414_042513.jsonl) |
| (7,4,8) | [000100](results/deepseek_vanilla_748_20260413_231351_results_20260414_000100.jsonl) | [000715](results/deepseek_vanilla_748_20260413_231351_results_20260414_000715.jsonl) | [004619](results/deepseek_vanilla_748_20260413_231351_results_20260414_004619.jsonl) | [010717](results/deepseek_vanilla_748_20260413_231351_results_20260414_010717.jsonl) | [015643](results/deepseek_vanilla_748_20260413_231351_results_20260414_015643.jsonl) |
| (10,6,60) | [024250](results/deepseek_vanilla_10_6_60_20260413_231351_results_20260414_024250.jsonl) | [025003](results/deepseek_vanilla_10_6_60_20260413_231351_results_20260414_025003.jsonl) | [032053](results/deepseek_vanilla_10_6_60_20260413_231351_results_20260414_032053.jsonl) | [033902](results/deepseek_vanilla_10_6_60_20260413_231351_results_20260414_033902.jsonl) | [042517](results/deepseek_vanilla_10_6_60_20260413_231351_results_20260414_042517.jsonl) |

## DeepSeek-R1-Distill-Llama-8B (mismatched Llama draft: `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B`)

Mismatched-draft variant — weak draft expected. Uses Llama-3.1 EAGLE3 head on DeepSeek target (matches the past `signal_data_*_deepseek8b_llamadraft.json` corpus).

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| (3,1,4) | 59.74 / 1.87 | 57.26 / 1.82 | 62.82 / 1.98 | 60.98 / 1.92 | 58.56 / 1.86 |
| (7,1,8) | 52.98 / 2.02 | 51.03 / 1.91 | 56.31 / 2.13 | — | — |
| (7,4,8) | 58.22 / 1.88 | 55.25 / 1.82 | 62.48 / 1.99 | 60.50 / 1.92 | 58.10 / 1.85 |
| (10,6,60) | 52.92 / 2.02 | 50.40 / 1.93 | 55.86 / 2.12 | — | — |

**DeepSeek-LD SOTA target:** `(3,1,4)` across the board — deep trees always lose because the Llama draft's token distribution diverges from the DeepSeek target's, so extra draft depth wastes compute without raising acceptance. This is the target regime dynamic spec should dominate (V6.5 dynamic doubles throughput to ~116 tok/s on gsm8k, see CLAUDE.md).

### DeepSeek-LD source files

| Config | mtbench | gsm8k | math500 | humaneval | livecodebench |
|--------|---------|-------|---------|-----------|---------------|
| (3,1,4) | [070900](results/deepseek_llamadraft_vanilla_314_20260416_062026_results_20260416_070900.jsonl) | [071656](results/deepseek_llamadraft_vanilla_314_20260416_062026_results_20260416_071656.jsonl) | [084746](results/deepseek_llamadraft_vanilla_314_20260416_062026_results_20260416_084746.jsonl) | [093813](results/deepseek_llamadraft_vanilla_314_20260416_062026_results_20260416_093813.jsonl) | [112838](results/deepseek_llamadraft_vanilla_314_20260416_062026_results_20260416_112838.jsonl) |
| (7,1,8) | [122431](results/deepseek_llamadraft_vanilla_718_20260416_062026_results_20260416_122431.jsonl) | [123326](results/deepseek_llamadraft_vanilla_718_20260416_062026_results_20260416_123326.jsonl) | [141547](results/deepseek_llamadraft_vanilla_718_20260416_062026_results_20260416_141547.jsonl) | — | — |
| (7,4,8) | [070851](results/deepseek_llamadraft_vanilla_748_20260416_062026_results_20260416_070851.jsonl) | [071658](results/deepseek_llamadraft_vanilla_748_20260416_062026_results_20260416_071658.jsonl) | [084753](results/deepseek_llamadraft_vanilla_748_20260416_062026_results_20260416_084753.jsonl) | [093822](results/deepseek_llamadraft_vanilla_748_20260416_062026_results_20260416_093822.jsonl) | [112824](results/deepseek_llamadraft_vanilla_748_20260416_062026_results_20260416_112824.jsonl) |
| (10,6,60) | [122422](results/deepseek_llamadraft_vanilla_10_6_60_20260416_062026_results_20260416_122422.jsonl) | [123316](results/deepseek_llamadraft_vanilla_10_6_60_20260416_062026_results_20260416_123316.jsonl) | [141540](results/deepseek_llamadraft_vanilla_10_6_60_20260416_062026_results_20260416_141540.jsonl) | — | — |

## Summary — throughput SOTA per model × benchmark

| Benchmark | Llama | Qwen | DeepSeek matched | DeepSeek LD |
|-----------|-------|------|------------------|-------------|
| mtbench | **219.44** `(10,6,60)` | **178.46** `(10,6,60)` | 69.71 `(3,1,4)` † | 59.74 `(3,1,4)` |
| gsm8k | 101.59 `(3,1,4)` | 122.17 `(3,1,4)` | 73.50 `(7,4,8)` | 57.26 `(3,1,4)` |
| math500 | 195.15 `(7,4,8)` | 128.05 `(10,6,60)` | **188.77** `(10,6,60)` | 62.82 `(3,1,4)` |
| humaneval | **224.95** `(7,4,8)` | **131.54** `(10,6,60)` | **155.55** `(7,1,8)` | 60.98 `(3,1,4)` |
| livecodebench | 156.17 `(7,4,8)` | 125.15 `(10,6,60)` | 141.17 `(10,6,60)` | 58.56 `(3,1,4)` |

The Llama/Qwen SOTA is `(10,6,60)` on mtbench and `(7,4,8)` or `(10,6,60)` on code/reasoning benchmarks. DeepSeek-matched splits cleanly: small trees win on short-response benchmarks (mtbench, gsm8k), deep trees win on long-reasoning benchmarks (math500, humaneval, livecodebench). DeepSeek-LD is always fastest at `(3,1,4)` — tree depth costs more than it saves when the draft diverges from the target.
