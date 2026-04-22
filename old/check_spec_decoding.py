# import requests
# import json
# from transformers import AutoTokenizer

# PORT = 30000
# MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# SHOW_ACTUAL_TOKENS = True  # set to False to hide decoded text

# tok = AutoTokenizer.from_pretrained(MODEL)

# input_ids = tok.apply_chat_template(
#     [{"role": "user", "content": "Think step by step. What is 2+2?"}],
#     tokenize=True,
#     add_generation_prompt=True,
# )

# data = {
#     "input_ids": input_ids,
#     "sampling_params": {
#         "temperature": 0,
#         "max_new_tokens": 200,
#     },
# }

# response = requests.post(f"http://localhost:{PORT}/generate", json=data)
# result = response.json()
# meta = result["meta_info"]

# print("spec_verify_ct:", meta["spec_verify_ct"])
# print("spec_accept_rate:", meta["spec_accept_rate"])
# print("spec_accept_length:", meta["spec_accept_length"])
# print()
# print("num verify steps:", len(meta["spec_draft_tokens"]))
# print("draft tokens per step:", [len(s) for s in meta["spec_draft_tokens"]])
# print()

# all_ok = all(
#     len(acc) + len(rej) == len(draft)
#     for draft, acc, rej in zip(
#         meta["spec_draft_tokens"],
#         meta["spec_accepted_tokens_log"],
#         meta["spec_rejected_tokens_log"],
#     )
# )
# print("sum_check all steps:", all_ok)
# print()

# for i, (draft, acc, rej) in enumerate(zip(
#     meta["spec_draft_tokens"],
#     meta["spec_accepted_tokens_log"],
#     meta["spec_rejected_tokens_log"],
# )):
#     sum_ok = len(acc) + len(rej) == len(draft)
#     if SHOW_ACTUAL_TOKENS:
#         draft_str = repr(tok.decode(draft))
#         acc_str   = repr(tok.decode(acc)) if acc else "''"
#         rej_str   = repr(tok.decode(rej)) if rej else "''"
#         print(f"step {i:3d}: proposed={len(draft)} accepted={len(acc)} rejected={len(rej)} sum_check={sum_ok}")
#         print(f"         proposed : {draft_str}")
#         print(f"         accepted : {acc_str}")
#         print(f"         rejected : {rej_str}")
#     else:
#         print(f"step {i:3d}: proposed={len(draft)} accepted={len(acc)} rejected={len(rej)} sum_check={sum_ok}")

"""
EAGLE3 speculative decoding tree visualizer.
"""

import requests
from transformers import AutoTokenizer

PORT  = 30000
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

GRN  = "\033[32m"
DIM  = "\033[90m"
CYAN = "\033[36m"
RST  = "\033[0m"
BOLD = "\033[1m"

tok = AutoTokenizer.from_pretrained(MODEL)

# ── Request ────────────────────────────────────────────────────────────────────
input_ids = tok.apply_chat_template(
    [{"role": "user", "content": "Think step by step. What is 2+2?"}],
    tokenize=True,
    add_generation_prompt=True,
)
response = requests.post(f"http://localhost:{PORT}/generate", json={
    "input_ids": input_ids,
    "sampling_params": {"temperature": 0, "max_new_tokens": 50},
})
meta = response.json()["meta_info"]

# ── Fields ─────────────────────────────────────────────────────────────────────
draft_tokens_per_step    = meta["spec_draft_tokens"]         # List[List[int]]  len=num_verify_steps
accepted_tokens_per_step = meta["spec_accepted_tokens_log"]  # List[List[int]]
rejected_tokens_per_step = meta["spec_rejected_tokens_log"]  # List[List[int]]
accept_index_per_step    = meta["spec_accept_index_log"]     # List[List[int]]  each entry = accepted node indices

topk            = meta["spec_topk"]
num_steps       = meta["spec_num_steps"]
draft_token_num = meta["spec_draft_token_num"]

# Shape: [1][draft_token_num]  (logged once per request, not per step)
# next_token[0][n]   = first child of node n  (-1 = leaf)
# next_sibling[0][n] = next sibling of node n (-1 = none)
next_token   = meta["spec_retrive_next_token"][0]    # List[int]  len=draft_token_num
next_sibling = meta["spec_retrive_next_sibling"][0]  # List[int]  len=draft_token_num

num_verify_steps = len(draft_tokens_per_step)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_children(node: int) -> list:
    """Return all children of node using next_token + next_sibling chains."""
    children = []
    child = next_token[node]
    while child != -1:
        children.append(child)
        child = next_sibling[child]
    return children


def format_bar(n_acc: int, n_total: int, width: int = 16) -> str:
    filled = round(width * n_acc / n_total) if n_total else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render_tree(draft: list, acc_set: set, prefix: str = "      "):
    """Recursively render the tree rooted at node 0."""
    def _render(node: int, pre: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        tid   = draft[node]
        text  = repr(tok.decode([tid]))
        col   = GRN if node in acc_set else DIM
        mark  = "✓" if node in acc_set else "✗"
        print(f"{pre}{connector}{col}{mark} [{node:2d}] {tid:>7}  {text}{RST}")
        children  = get_children(node)
        child_pre = pre + ("    " if is_last else "│   ")
        for k, child in enumerate(children):
            _render(child, child_pre, k == len(children) - 1)

    # Root
    tid  = draft[0]
    text = repr(tok.decode([tid]))
    col  = GRN if 0 in acc_set else DIM
    mark = "✓" if 0 in acc_set else "✗"
    print(f"{prefix}{col}{mark} [ 0] {tid:>7}  {text}{RST}")

    root_children = get_children(0)
    for k, child in enumerate(root_children):
        _render(child, prefix, k == len(root_children) - 1)


# ── Header ─────────────────────────────────────────────────────────────────────
W = 64
print(f"\n{BOLD}{'─' * W}{RST}")
print(f"{BOLD}  EAGLE3 speculative decode tree{RST}")
print(f"{'─' * W}")
print(f"  topk={topk}  num_steps={num_steps}  draft_token_num={draft_token_num}")
print(f"  verify_steps   {num_verify_steps}")
print(f"  accept_rate    {meta['spec_accept_rate']:.3f}")
print(f"  accept_length  {meta['spec_accept_length']:.3f}")
print(f"  verify_ct      {meta['spec_verify_ct']}")
sum_ok = all(
    len(a) + len(r) == len(d)
    for d, a, r in zip(draft_tokens_per_step, accepted_tokens_per_step, rejected_tokens_per_step)
)
print(f"  len_check      {'✓ |acc|+|rej|==|draft| for all steps' if sum_ok else '✗ MISMATCH'}")
print(f"  tree topology  next_token={next_token}")
print(f"                 next_sibling={next_sibling}")
print(f"{'─' * W}\n")

# ── Per-step ───────────────────────────────────────────────────────────────────
for step_i, (draft, acc, rej, acc_idx) in enumerate(zip(
    draft_tokens_per_step,
    accepted_tokens_per_step,
    rejected_tokens_per_step,
    accept_index_per_step,
)):
    n_acc   = len(acc)
    n_total = len(draft)
    bar     = format_bar(n_acc, n_total)

    print(f"  {BOLD}step {step_i:>2}{RST}  {CYAN}{bar}{RST}  {n_acc}/{n_total} accepted")

    acc_set = set(acc_idx)
    render_tree(draft, acc_set)

    if acc:
        print(f"\n        {GRN}accepted ▶{RST}  {repr(tok.decode(acc))}")
        print(f"        {GRN}indices  ▶{RST}  {acc_idx}")
    if rej:
        print(f"        {DIM}rejected ▶  {repr(tok.decode(rej))}{RST}")
    print()