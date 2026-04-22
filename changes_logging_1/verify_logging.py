"""
Verify per-step speculative decoding logging and render draft trees.

Start the server first, then run this script:

    # Config A (narrow: steps=3, topk=1, dtn=4)
    python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
        --speculative-num-steps 3 --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16

    python verify_logging.py --port 30000 --config narrow

    # Config B (wide: steps=5, topk=3, dtn=6)
    python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
        --speculative-num-steps 5 --speculative-eagle-topk 3 \
        --speculative-num-draft-tokens 6 \
        --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \
        --trust-remote-code --host 0.0.0.0 --port 30000 --dtype bfloat16

    python verify_logging.py --port 30000 --config wide
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

# ── Colours ──────────────────────────────────────────────────────────────────
GRN  = "\033[32m"
RED  = "\033[31m"
DIM  = "\033[90m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RST  = "\033[0m"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CONFIGS = {
    "narrow": {"steps": 3, "topk": 1, "dtn": 4},
    "wide":   {"steps": 5, "topk": 3, "dtn": 6},
}

MTBENCH_PATH = Path(__file__).parent / "SpecForge" / "benchmarks" / "mtbench.jsonl"


def load_prompt():
    """Load the first MT-Bench question."""
    with open(MTBENCH_PATH) as f:
        entry = json.loads(f.readline())
    return entry["turns"][0]


def send_request(port: int, prompt: str, tokenizer, max_tokens: int = 200):
    """Send a generate request and return (output_text, meta_info)."""
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    resp = requests.post(
        f"http://localhost:{port}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    return result.get("text", ""), result["meta_info"]


def print_meta_info(meta: dict):
    """Pretty-print all meta_info keys."""
    print(f"\n{BOLD}-- Raw meta_info --{RST}")

    # Aggregate metrics first
    agg_keys = [
        "spec_verify_ct", "spec_accept_rate", "spec_accept_length",
        "spec_accept_token_num", "spec_draft_token_num",
    ]
    for k in agg_keys:
        if k in meta:
            v = meta[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if "spec_accept_histogram" in meta:
        print(f"  spec_accept_histogram: {meta['spec_accept_histogram']}")

    # Per-step fields
    per_step_keys = [
        "spec_topk", "spec_num_steps", "spec_draft_token_num",
        "spec_threshold_single",
        "spec_draft_tokens", "spec_accepted_tokens_log",
        "spec_rejected_tokens_log", "spec_accept_index_log",
        "spec_retrive_next_token", "spec_retrive_next_sibling",
    ]
    print()
    for k in per_step_keys:
        if k not in meta:
            print(f"  {RED}MISSING: {k}{RST}")
            continue
        v = meta[k]
        if isinstance(v, list) and len(v) > 3:
            preview = repr(v[:3])[:-1] + ", ...]"
            print(f"  {k}: ({len(v)} steps) {preview}")
        else:
            print(f"  {k}: {v}")
    print()


def run_checks(meta: dict, expected_cfg: dict = None):
    """Run consistency checks on per-step logging, return True if all pass."""
    ok = True
    draft = meta.get("spec_draft_tokens", [])
    acc = meta.get("spec_accepted_tokens_log", [])
    rej = meta.get("spec_rejected_tokens_log", [])
    topk = meta.get("spec_topk", [])
    num_steps = meta.get("spec_num_steps", [])
    dtn = meta.get("spec_draft_token_num", [])
    threshold = meta.get("spec_threshold_single", [])
    next_tok = meta.get("spec_retrive_next_token", [])
    next_sib = meta.get("spec_retrive_next_sibling", [])
    acc_idx = meta.get("spec_accept_index_log", [])

    n = len(draft)
    checks = [
        ("per-step fields present", all(k in meta for k in [
            "spec_draft_tokens", "spec_accepted_tokens_log",
            "spec_rejected_tokens_log", "spec_accept_index_log",
            "spec_topk", "spec_num_steps", "spec_draft_token_num",
            "spec_threshold_single",
            "spec_retrive_next_token", "spec_retrive_next_sibling",
        ])),
        ("acc+rej==draft for all steps", all(
            len(a) + len(r) == len(d) for d, a, r in zip(draft, acc, rej)
        )),
        ("len(spec_topk)==num_verify_steps", len(topk) == n),
        ("len(spec_num_steps)==num_verify_steps", len(num_steps) == n),
        ("len(spec_draft_token_num)==num_verify_steps", len(dtn) == n),
        ("len(spec_threshold_single)==num_verify_steps", len(threshold) == n),
        ("len(retrive_next_token)==num_verify_steps", len(next_tok) == n),
        ("len(retrive_next_sibling)==num_verify_steps", len(next_sib) == n),
    ]

    # Tree pointer length matches dtn per step
    if dtn and next_tok and len(dtn) == len(next_tok):
        checks.append((
            "tree pointer length == dtn per step",
            all(len(nt) == d for nt, d in zip(next_tok, dtn))
        ))

    # Config mismatch check — warn if server params don't match --config
    if expected_cfg and topk:
        actual_topk = topk[0] if topk else None
        actual_steps = num_steps[0] if num_steps else None
        actual_dtn = dtn[0] if dtn else None
        mismatches = []
        if actual_topk != expected_cfg["topk"]:
            mismatches.append(f"topk: expected {expected_cfg['topk']}, got {actual_topk}")
        if actual_steps != expected_cfg["steps"]:
            mismatches.append(f"steps: expected {expected_cfg['steps']}, got {actual_steps}")
        if actual_dtn != expected_cfg["dtn"]:
            mismatches.append(f"dtn: expected {expected_cfg['dtn']}, got {actual_dtn}")
        checks.append((
            f"server config matches --config ({', '.join(mismatches) if mismatches else 'ok'})",
            len(mismatches) == 0,
        ))

    print(f"{BOLD}-- Checks --{RST}")
    for name, passed in checks:
        mark = f"{GRN}PASS{RST}" if passed else f"{RED}FAIL{RST}"
        print(f"  [{mark}] {name}")
        if not passed:
            ok = False
    print()
    return ok


def render_tree_step(meta: dict, step_i: int, tokenizer):
    """Render the draft tree for a single verify step."""
    draft = meta["spec_draft_tokens"][step_i]
    acc_idx = meta["spec_accept_index_log"][step_i]
    acc_toks = meta["spec_accepted_tokens_log"][step_i]
    rej_toks = meta["spec_rejected_tokens_log"][step_i]
    next_token = meta["spec_retrive_next_token"][step_i]
    next_sibling = meta["spec_retrive_next_sibling"][step_i]

    # Per-step hyperparams
    topk = meta["spec_topk"][step_i]
    num_steps = meta["spec_num_steps"][step_i]
    dtn = meta["spec_draft_token_num"][step_i]
    threshold = meta["spec_threshold_single"][step_i]

    acc_set = set(acc_idx)
    n_acc = len(acc_toks)
    n_total = len(draft)

    # Progress bar
    bar_width = 16
    filled = round(bar_width * n_acc / n_total) if n_total else 0
    bar = "[" + "#" * filled + "." * (bar_width - filled) + "]"

    print(f"  {BOLD}Step {step_i:>2}{RST}  topk={topk}  steps={num_steps}  "
          f"dtn={dtn}  threshold={threshold}")
    print(f"  {CYAN}{bar}{RST}  {n_acc}/{n_total} accepted")

    def get_children(node):
        children = []
        child = next_token[node]
        while child != -1 and child < len(next_sibling):
            children.append(child)
            child = next_sibling[child]
        return children

    def render_node(node, prefix, is_last):
        connector = "  +-- " if is_last else "  |-- "
        tid = draft[node]
        text = repr(tokenizer.decode([tid]))
        if node in acc_set:
            col, mark = GRN, "ACC"
        else:
            col, mark = DIM, "rej"
        print(f"{prefix}{connector}{col}[{mark}] [{node:2d}] {tid:>7}  {text}{RST}")
        children = get_children(node)
        child_prefix = prefix + ("      " if is_last else "  |   ")
        for k, child in enumerate(children):
            render_node(child, child_prefix, k == len(children) - 1)

    # Root node
    tid = draft[0]
    text = repr(tokenizer.decode([tid]))
    if 0 in acc_set:
        col, mark = GRN, "ACC"
    else:
        col, mark = DIM, "rej"
    print(f"      {col}[{mark}] [ 0] {tid:>7}  {text}{RST}")

    root_children = get_children(0)
    for k, child in enumerate(root_children):
        render_node(child, "      ", k == len(root_children) - 1)

    # Decoded accepted/rejected text
    if acc_toks:
        print(f"\n      {GRN}accepted >{RST}  {repr(tokenizer.decode(acc_toks))}")
        print(f"      {GRN}indices  >{RST}  {acc_idx}")
    if rej_toks:
        print(f"      {DIM}rejected >  {repr(tokenizer.decode(rej_toks))}{RST}")
    print()


def run_config(config_name: str, port: int, tokenizer):
    """Send a request and display full results."""
    cfg = CONFIGS[config_name]
    prompt = load_prompt()

    W = 64
    print(f"\n{'=' * W}")
    print(f"  Config: {config_name}  "
          f"(topk={cfg['topk']}, steps={cfg['steps']}, dtn={cfg['dtn']})")
    print(f"  Prompt: {prompt[:60]}...")
    print(f"{'=' * W}")

    print(f"\n  Sending request to localhost:{port}...")
    try:
        output_text, meta = send_request(port, prompt, tokenizer)
    except requests.exceptions.ConnectionError:
        print(f"\n  {RED}ERROR: Cannot connect to localhost:{port}. "
              f"Is the server running?{RST}")
        print(f"  Start it with:")
        print(f"    python3 -m sglang.launch_server \\")
        print(f"      --model {MODEL} \\")
        print(f"      --speculative-algorithm EAGLE3 \\")
        print(f"      --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \\")
        print(f"      --speculative-num-steps {cfg['steps']} "
              f"--speculative-eagle-topk {cfg['topk']} \\")
        print(f"      --speculative-num-draft-tokens {cfg['dtn']} \\")
        print(f"      --mem-fraction-static 0.75 --cuda-graph-max-bs 1 --tp 1 \\")
        print(f"      --trust-remote-code --host 0.0.0.0 --port {port} --dtype bfloat16")
        return False

    print(f"\n{BOLD}-- Generated text (first 200 chars) --{RST}")
    print(f"  {output_text[:200]}")

    print_meta_info(meta)
    all_ok = run_checks(meta, expected_cfg=cfg)

    # Render per-step trees
    num_steps = len(meta.get("spec_draft_tokens", []))
    if num_steps == 0:
        print(f"  {RED}No per-step data found in meta_info.{RST}")
        return False

    print(f"{BOLD}-- Tree visualisation ({num_steps} verify steps) --{RST}\n")
    for step_i in range(num_steps):
        render_tree_step(meta, step_i, tokenizer)

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify speculative decoding logging")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="narrow",
                        help="Server config to validate against (narrow or wide)")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading tokenizer: {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    ok = run_config(args.config, args.port, tokenizer)
    if ok:
        print(f"\n{GRN}{BOLD}All checks passed.{RST}\n")
    else:
        print(f"\n{RED}{BOLD}Some checks failed.{RST}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
