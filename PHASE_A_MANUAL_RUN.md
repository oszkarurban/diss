# Phase A manual run guide — 1h INTR, one model at a time

This is the fallback workflow when you prefer interactive execution over
`sbatch hpc/run_trace_collection.sh` (which needs a 4h allocation to run
both models). Each session runs **one model's 18 benchmarks** (6 configs
× 3 datasets × 20 q) in ~45 min — fits a 1h INTR with margin.

## Prereq once per login

Nothing to install — the patches are already in the `sglang` submodule and
signal collection is via `test_signal_collection.py`.

Recommended before each session:

```bash
# Prune down / up nodes that have burned you today
sinfo -p ampere -h -o "%T" | sort | uniq -c
sinfo -p ampere -R -o "%20N %30E" 2>/dev/null | head -20  # which are down
```

## Session 1 — Llama (in a 1h INTR)

```bash
# 1. Grab an INTR node (1h, GPU)
sintr -A MASCOLO-SL2-GPU -p ampere -t 1:0:0 -N 1 --gres=gpu:1 --qos=INTR --exclude=gpu-q-1,gpu-q-3

# 2. Inside the sintr shell — env setup
cd /rds/user/ou222/hpc-work/diss
conda activate sglang-dev
source hpc/unload_prepare.sh

# 3. Smoke test (ONE TIME, skip on reruns) — confirms code patches work.
#    ~5-8 min. Expect "=== Smoke test PASSED ===".
bash hpc/smoke_test_signal_logging.sh

# 4. Phase A for Llama only (~45 min). Output goes to results/traces/.
ONLY_MODEL=llama bash hpc/run_trace_collection.sh --no-sbatch
```

Expect 18 JSON files named
`<config>_llama_<dataset>_<timestamp>.json` under `results/traces/`.

## Session 2 — Qwen (a later 1h INTR)

Same as session 1 but `ONLY_MODEL=qwen`:

```bash
sintr -A MASCOLO-SL2-GPU -p ampere -t 1:0:0 -N 1 --gres=gpu:1 --qos=INTR --exclude=gpu-q-1,gpu-q-3
cd /rds/user/ou222/hpc-work/diss
conda activate sglang-dev
source hpc/unload_prepare.sh

# Skip the smoke test — already verified in session 1.
ONLY_MODEL=qwen bash hpc/run_trace_collection.sh --no-sbatch
```

Another 18 JSON files, this time `*_qwen_*`.

## After both sessions — flatten into analysis-friendly JSONL.gz

On the login node (no GPU needed):

```bash
cd /rds/user/ou222/hpc-work/diss
python3 analysis/extract_signal_traces.py results/traces/
```

This creates one gzipped JSONL per (config, model, dataset) group, e.g.
`results/traces/static_3_1_4_llama_mtbench_signals.jsonl.gz`. Each row is
one decode step with columns: `config, model, dataset, turn, step,
top1_prob, target_top1_prob, rolling_accept_rate, top1_threshold,
target_threshold, chosen_topk, chosen_num_steps, chosen_num_draft_tokens,
accept_length`.

## Sanity checks per session

Inside the live sintr (takes ~5 sec):

```bash
# How many outputs landed for this model?
ls results/traces/*_llama_*.json | wc -l    # expected: 18

# Peek at one static run to verify chosen_* was kept at static values
python3 -c "
import json
import glob
path = sorted(glob.glob('results/traces/static_3_1_4_llama_mtbench_*.json'))[-1]
d = json.load(open(path))
turns = d['per_turn_logs']
for turn in turns:
    if turn.get('signals'):
        s = turn['signals'][0]
        print(f\"{path}\")
        print(f\"  first step: topk={s['chosen_topk']} ns={s['chosen_num_steps']} ndt={s['chosen_num_draft_tokens']}\")
        print(f\"  top1={s['top1_prob']:.3f} target={s['target_top1_prob']:.3f} rar={s['rolling_accept_rate']:.3f}\")
        break
"
```

Expected for static (3,1,4): `topk=1 ns=3 ndt=4`. If chosen_* varies, the
static signal-logging branch in eagle_worker.py got bypassed — report back.

## Per-config time budget (what to kill if running out)

If the 1h wall clock is getting tight, the order in
`run_trace_collection.sh` is static-first, dynamic-last:

1. static_3_1_4
2. static_7_1_8
3. static_7_4_8
4. static_6_10_60
5. v3_dynamic
6. v6_dynamic

Each config takes ~7 min (60s cold import + 3×~2min bench + kill). You
can Ctrl+C the bench loop between configs to bail out early; partial
output files are still usable.

## Recovering a half-finished session

`extract_signal_traces.py` handles duplicates by picking the newest
timestamp per (config, model, dataset). If a config ran in session 1 but
you want to redo it cleanly in session 2, just rerun the script — the
older file stays on disk but the newer one is used.

To force merging timestamps (e.g. you split livecodebench across two
sessions):

```bash
python3 analysis/extract_signal_traces.py results/traces/ --append --overwrite
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sglang'` | Still in `base` conda env | `conda activate sglang-dev` |
| `sglang.__file__ is None` | CWD shadowing by `diss/sglang/` dir | Script already cds to `/tmp` — but if launching manually, `cd /tmp` first |
| Server never prints for > 3 min | Cold Lustre on new node | Wait — first import is 3-5 min. Not broken. |
| `/home/ou222/miniforge3/bin/gcc: No such file or directory` in log | `unload_prepare.sh` wasn't sourced after `conda activate` | Re-source: `conda activate sglang-dev && source hpc/unload_prepare.sh` |
| Process stuck at STAT=I with 0 CPU | Lustre I/O stall on node | `scancel <jobid>`, resintr with different `--exclude` |

## Notes

- Outputs share the same directory as existing `results/` benchmark files.
  That's intentional — the extractor filters by the
  `<config>_<model>_<dataset>_<timestamp>.json` pattern and ignores
  anything else.
- Each `test_signal_collection.py` run sends requests serially (bs=1)
  matching the CLAUDE.md benchmark convention. Do not increase
  `--batch-size` without separately re-measuring static SOTA baselines.
- `test_signal_collection.py` uses `max_new_tokens=2048`. livecodebench
  prompts in particular may be clipped — note this in Phase B analysis.
