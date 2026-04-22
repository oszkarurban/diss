"""
Per-step accept_length distributions from signal_data_*.json to characterise
regime mix under static EAGLE3. Answers: "is there headroom for a smart
dynamic policy?" — if the accept_length histogram is unimodal near num_steps
the biggest static config wins; if bimodal (many 0s and many saturated)
there is oracle-upper-bound gain proportional to regime B + regime A share.
"""
import json
import os
from collections import Counter


def load_steps(path):
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict) or "per_turn_logs" not in d:
        return None, None
    pt = d["per_turn_logs"]
    steps = []
    configs = Counter()
    for turn in pt:
        if isinstance(turn, list):
            for sig in turn:
                steps.append(sig)
                configs[(sig.get("chosen_num_steps"),
                         sig.get("chosen_topk"),
                         sig.get("chosen_num_draft_tokens"))] += 1
        elif isinstance(turn, dict) and "signals" in turn:
            signals = turn["signals"]
            topk = turn.get("spec_topk") or []
            ns = turn.get("spec_num_steps") or []
            ndt = turn.get("spec_draft_token_num") or []
            for i, sig in enumerate(signals):
                steps.append(sig)
                tk = topk[i] if i < len(topk) else sig.get("chosen_topk")
                n = ns[i] if i < len(ns) else sig.get("chosen_num_steps")
                nd = ndt[i] if i < len(ndt) else sig.get("chosen_num_draft_tokens")
                configs[(n, tk, nd)] += 1
    return steps, configs


def summarize(path, label=None):
    steps, configs = load_steps(path)
    if steps is None:
        return None
    label = label or os.path.basename(path)
    acc = [s.get("accept_length", 0) for s in steps]
    total = len(acc)
    top_config = configs.most_common(1)[0][0]
    num_steps_cfg = top_config[0]
    hist = Counter(acc)
    reg_B = sum(c for k, c in hist.items() if k == 0)
    reg_A = sum(c for k, c in hist.items() if k == num_steps_cfg)
    reg_C = total - reg_B - reg_A
    mean = sum(acc) / total if total else 0
    var = sum((x - mean) ** 2 for x in acc) / total if total else 0
    return {
        "label": label,
        "path": path,
        "steps": total,
        "config_mix": dict(configs.most_common(5)),
        "top_config_(ns,tk,ndt)": top_config,
        "accept_length_mean": round(mean, 3),
        "accept_length_var": round(var, 3),
        "hist": dict(sorted(hist.items())),
        "regime_B_reject_pct": round(100 * reg_B / total, 1),
        "regime_C_partial_pct": round(100 * reg_C / total, 1),
        "regime_A_saturated_pct": round(100 * reg_A / total, 1),
    }


def pretty(s):
    if s is None:
        return "<no per_turn_logs>"
    cfg = s["top_config_(ns,tk,ndt)"]
    return "\n".join([
        f"=== {s['label']} ===",
        f"  file      : {s['path']}",
        f"  top config: (num_steps,topk,ndt) = {cfg}   steps logged = {s['steps']}",
        f"  config mix: {s['config_mix']}",
        f"  accept_len mean/var: {s['accept_length_mean']}  /  {s['accept_length_var']}",
        f"  regime B (reject, accept_len=0)  : {s['regime_B_reject_pct']}%",
        f"  regime C (partial)               : {s['regime_C_partial_pct']}%",
        f"  regime A (saturated, =num_steps) : {s['regime_A_saturated_pct']}%",
        f"  histogram: {s['hist']}",
    ])


if __name__ == "__main__":
    DIR = "/rds/user/ou222/hpc-work/diss"
    targets = [
        ("Llama matched, static (7,4,8)",
         f"{DIR}/signal_data_static_748_llama.json"),
        ("Llama matched, static (7,4,8) v61",
         f"{DIR}/signal_data_v61_llama.json"),
        ("DeepSeek matched, static (7,4,8)",
         f"{DIR}/signal_data_analysis_static_deepseek.json"),
        ("DeepSeek-LD (mismatched), (7,4,8)",
         f"{DIR}/signal_data_tree_deepseek_516_748.json"),
        ("DeepSeek-LD (mismatched), tree file 314_748",
         f"{DIR}/signal_data_tree_deepseek_314_748.json"),
        ("Llama-matched vanilla 6active",
         f"{DIR}/signal_data_6active_llama8b.json"),
        ("DeepSeek-LD 26signals vanilla",
         f"{DIR}/signal_data_26signals_deepseek8b_llamadraft.json"),
        ("Llama-matched vanilla_llama8b",
         f"{DIR}/signal_data_vanilla_llama8b.json"),
    ]
    for label, path in targets:
        try:
            s = summarize(path, label)
            print(pretty(s))
            print()
        except FileNotFoundError:
            print(f"=== {label} === FILE NOT FOUND: {path}\n")
        except Exception as e:
            print(f"=== {label} === ERROR: {e}\n")
