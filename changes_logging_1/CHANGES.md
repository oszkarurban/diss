# Changes since initial SGLang clone

Base commit: `d35fea1b2` (upstream SGLang HEAD at time of fork)

## Summary

All changes are **purely additive instrumentation** on top of vanilla EAGLE3 speculative decoding. They log data that the existing code path already produces internally but does not expose to the client. No decision logic or behaviour is modified.

## What is logged

After each `verify()` call in the EAGLE3 decode loop, the following is recorded per verify step:

| Field | Type (per step) | Source |
|-------|-----------------|--------|
| `spec_draft_tokens` | `List[int]` | `candidates` tensor — all draft token IDs proposed |
| `spec_accepted_tokens_log` | `List[int]` | subset of draft tokens accepted by rejection sampling |
| `spec_rejected_tokens_log` | `List[int]` | subset of draft tokens rejected |
| `spec_accept_index_log` | `List[int]` | tree node indices of accepted tokens |
| `spec_retrive_next_token` | `List[int]` | tree child pointers (first child of each node, -1 = leaf) |
| `spec_retrive_next_sibling` | `List[int]` | tree sibling pointers (next sibling, -1 = none) |
| `spec_logged_topk` | `int` | topk branching factor used this step |
| `spec_logged_num_steps` | `int` | number of draft steps used this step |
| `spec_logged_draft_token_num` | `int` | draft token budget used this step |
| `spec_logged_threshold_single` | `float` | rejection sampling threshold used this step |

All fields are lists indexed by verify step number. The tree pointers (`retrive_next_token`, `retrive_next_sibling`) together with the draft tokens and accept indices fully define the tree topology and which paths were accepted — enough to reconstruct and visualise the complete speculative decode tree at each step.

## Data flow

```
eagle_info.py:verify()  — populates fields on Req object
    |
scheduler_output_processor_mixin.py  — collects from Req into batch lists
    |
detokenizer_manager.py  — passes through IPC
    |
multi_tokenizer_mixin.py  — handles multi-tokenizer batches
    |
tokenizer_manager.py  — writes to meta_info dict
    |
HTTP response  — accessible via response.json()["meta_info"]
```

## Files modified (relative to `sglang/python/sglang/srt/`)

| File | Change |
|------|--------|
| `speculative/eagle_info.py` | `candidates.tolist()` + per-step logging of all fields after rejection sampling |
| `managers/schedule_batch.py` | New `List` fields on `Req` for all logged data |
| `managers/io_struct.py` | Type annotations in `SpeculativeDecodingMetricsMixin` dataclass |
| `managers/scheduler_output_processor_mixin.py` | Collect from `Req` objects into batch output |
| `managers/detokenizer_manager.py` | Pass-through in IPC between scheduler and detokenizer |
| `managers/multi_tokenizer_mixin.py` | Pass-through for both `BatchTokenIDOutput` and `BatchStrOutput` |
| `managers/tokenizer_manager.py` | Write to `meta_info` dict in HTTP response |

## Accessing the data

```python
import requests
response = requests.post(f"http://localhost:30000/generate", json={
    "input_ids": [...],
    "sampling_params": {"temperature": 0, "max_new_tokens": 200},
})
meta = response.json()["meta_info"]

# Per-step data (all lists of length == number of verify steps)
meta["spec_draft_tokens"]           # [[tok, tok, ...], [tok, ...], ...]
meta["spec_accepted_tokens_log"]    # [[tok, ...], ...]
meta["spec_rejected_tokens_log"]    # [[tok, ...], ...]
meta["spec_accept_index_log"]       # [[idx, ...], ...]
meta["spec_topk"]                   # [topk_step0, topk_step1, ...]
meta["spec_num_steps"]              # [steps_step0, ...]
meta["spec_draft_token_num"]        # [dtn_step0, ...]
meta["spec_threshold_single"]       # [thresh_step0, ...]
meta["spec_retrive_next_token"]     # [[child_ptr, ...], ...]
meta["spec_retrive_next_sibling"]   # [[sibling_ptr, ...], ...]

# Aggregate metrics (from vanilla SGLang, unchanged)
meta["spec_verify_ct"]              # total verify() calls
meta["spec_accept_rate"]            # accepted / total draft tokens
meta["spec_accept_length"]          # tokens per verify step
meta["spec_accept_histogram"]       # histogram[k] = steps with k accepted
```

## Verification

Tested with two server configurations producing 200 tokens from the first MT-Bench question:

| Config | topk | steps | dtn | verify_steps | accept_length | accept_rate |
|--------|------|-------|-----|--------------|---------------|-------------|
| narrow | 1 | 3 | 4 | 73 | 2.74 | 58.5% |
| wide | 3 | 5 | 6 | 60 | 3.33 | 46.7% |

- Narrow produces linear chain trees (no branching). Wide produces branching trees with up to 3 children per node.
- All consistency checks pass: `len(acc)+len(rej)==len(draft)` per step, array lengths align, tree pointer sizes match dtn.
- Per-step tree pointers correctly vary between steps in the wide config (different tree topologies depending on which candidates scored highest).
- Accepted token paths reconstruct into coherent text matching the generated output.

Verification script: `verify_logging.py`

## Commits

1. `01916e3e7` — "log proposed, accepted, rejected tokens per step" (initial instrumentation)
2. `0c2dc1110` — "fix variable overlap" (rename `spec_topk` -> `spec_logged_topk` etc. to avoid field collision)
3. Per-step hyperparameter logging + bug fixes:
   - Fixed `retrive_next_token.tolist()` storing entire batch instead of per-req row (`[i]` indexing)
   - Changed tree structure and hyperparams from "log once" scalars to per-step lists
   - Added missing fields in `BatchTokenIDOutput` path in `multi_tokenizer_mixin.py`
   - Fixed incomplete rename in `BatchStrOutput` path (`spec_topk` -> `spec_logged_topk`)
   - Added `spec_logged_threshold_single` throughout the pipeline
   - Added config mismatch check in `verify_logging.py`


(sglang-dev) [ou222@gpu-q-31 sglang]$ git push
Enumerating objects: 27, done.
Counting objects: 100% (27/27), done.
Delta compression using up to 128 threads
Compressing objects: 100% (15/15), done.
Writing objects: 100% (15/15), 27.49 KiB | 5.50 MiB/s, done.
Total 15 (delta 12), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (12/12), completed with 12 local objects.
To github.com:oszkarurban/sglang-clean.git
   0c2dc1110..c8ac92d4b  main -> main
(sglang-dev) [ou222@gpu-q-31 sglang]$ 