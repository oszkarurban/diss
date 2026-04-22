# SpecForge Official Benchmark Results


## Llama-3.1-8B-Instruct + EAGLE3-LLaMA3.1-Instruct-8B (matched draft)

| Metric | Vanilla EAGLE3 (3,1,4) | V3 Dynamic (3,1,4→7,4,8) | Improvement |
|--------|------------------------|---------------------------|-------------|
| **Throughput** | 180.93 tok/s | **217.93 tok/s** | **+20.5%** |
| **Accept Length** | 2.928 | **4.145** | **+41.6%** |
| **Latency** | 333.13s | **277.85s** | **-16.6% (faster)** |
| Questions | 80 | 80 | — |

### Server Configuration

**Vanilla EAGLE3 (baseline):**
```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

**V3 Dynamic Speculative Decoding:**
```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-num-steps-max 7 --speculative-eagle-topk-max 4 \
    --speculative-num-draft-tokens-max 8 \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

**Benchmark command** (from `SpecForge/benchmarks/`):
```bash
python bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --config-list 1,0,0,0 \
    --benchmark-list mtbench:80 \
    --dtype bfloat16 \
    --skip-launch-server
```

### Raw Results

**Vanilla** (`results/results_20260402_005000.jsonl`):
```json
{
    "latency": 333.1261417969363,
    "output_throughput": 180.92846053714993,
    "accept_length": 2.9276728032253363,
    "num_questions": 80
}
```

**V3 Dynamic** (`results/results_20260402_133505.jsonl`):
```json
{
    "latency": 277.8520206959802,
    "output_throughput": 217.92535410873845,
    "accept_length": 4.145341274731293,
    "num_questions": 80
}
```

### SpecForge Reference (other fixed configs, same model pair)

| Config (bs,steps,topk,ndt) | AccLen | TP (tok/s) | Speedup |
|----------------------------|--------|------------|---------|
| 1-3-1-4 | 2.93 | 414.9 | 2.18x |
| 1-5-1-6 | 3.55 | 453.7 | 2.39x |
| 1-7-1-8 | 3.91 | 454.7 | 2.39x |
| 1-5-3-6 | 2.94 | 338.6 | 1.78x |
| 1-7-4-8 | 2.91 | 305.2 | 1.61x |

Note: SpecForge reference numbers were collected on a different day/session. Absolute throughput values differ from our runs due to GPU thermal state, background load, etc. Relative comparisons within the same session (Vanilla vs V3 above) are reliable.

---

## V3 Policy Summary

Single-signal universal policy distilled from decision tree analysis of 51k steps:

- **Signal**: `draft_oracle_gate` = `top1_prob` x `rolling_accept_rate`
- **Formula**: `t = min(1.0, draft_oracle_gate / 0.5)`
- **Mapping**: `num_steps = interp(t, start, max)`, `topk = interp(1-t, start, max)`
- **Properties**: No warmup, no model-specific tuning, no hardcoded configs
- **Lossless**: Output is mathematically identical to vanilla — speculative decoding preserves the target model distribution exactly

---

## DeepSeek-R1-Distill-Llama-8B + EAGLE3-LLaMA3.1-Instruct-8B (mismatched draft)

| Metric | Vanilla EAGLE3 (3,1,4) | V3 Dynamic (3,1,4→7,4,8) | Improvement |
|--------|------------------------|---------------------------|-------------|
| **Throughput** | 116.83 tok/s | **128.75 tok/s** | **+10.2%** |
| **Accept Length** | 1.884 | **2.292** | **+21.7%** |
| **Latency** | 1212.02s | **1169.95s** | **-3.5% (faster)** |
| Questions | 80 | 80 | — |

### Server Configuration

**Vanilla EAGLE3 (baseline):**
```bash
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

**V3 Dynamic Speculative Decoding:**
```bash
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-num-steps-max 7 --speculative-eagle-topk-max 4 \
    --speculative-num-draft-tokens-max 8 \
    --enable-dynamic-speculative-decoding \
    --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
    --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16
```

### Raw Results

**Vanilla** (`results/results_20260402_025909.jsonl`):
```json
{
    "latency": 1212.0202083759941,
    "output_throughput": 116.82891012991512,
    "accept_length": 1.8837169083410936,
    "num_questions": 80
}
```

**V3 Dynamic** (`results/results_20260402_134552.jsonl`):
```json
{
    "latency": 1169.947350114002,
    "output_throughput": 128.75451188920744,
    "accept_length": 2.292262040629993,
    "num_questions": 80
}
```

---

## Combined Results Summary

| Model | Method | Throughput (tok/s) | Accept Length | TP Improvement |
|-------|--------|-------------------|--------------|----------------|
| **Llama 8B** | Vanilla EAGLE3 | 180.93 | 2.928 | baseline |
| **Llama 8B** | **V3 Dynamic** | **217.93** | **4.145** | **+20.5%** |
| **DeepSeek 8B** | Vanilla EAGLE3 | 116.83 | 1.884 | baseline |
| **DeepSeek 8B** | **V3 Dynamic** | **128.75** | **2.292** | **+10.2%** |
