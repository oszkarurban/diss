"""
Per-step accept_length regime analysis using results/traces/master.parquet.

Regimes (per step). The master.parquet `accept_length` *includes* the bonus
token, so the minimum possible value is 1 (bonus only; no draft token was
accepted) and the maximum is num_steps+1 (all draft tokens + bonus accepted).
  B (reject)    — accept_length == 1            (zero draft tokens accepted)
  C (partial)   — 1 < accept_length <= num_steps
  A (saturated) — accept_length == num_steps+1  (every draft token accepted)

A static config is optimal if regime C dominates AND regime A is rare (would
otherwise motivate a bigger tree). Dynamic has headroom when regime B+A are
material — B because a smaller tree would produce the same bonus token at
fraction of the cost, A because a bigger tree could accept more.

Also reports yield_per_compute statistics — the trace column that already
measures (accept_length+1) / tree_size, a direct per-step efficiency proxy.
"""
import pandas as pd


PARQUET = "/rds/user/ou222/hpc-work/diss/results/traces/master.parquet"


def regime_table(df):
    rows = []
    grouped = df.groupby(["model", "dataset", "config"], observed=True)
    for (model, dataset, cfg), g in grouped:
        ns = g["chosen_num_steps"]
        acc = g["accept_length"]
        # accept_length includes bonus → min=1, max=ns+1
        is_B = (acc == 1)
        is_A = (acc == ns + 1)
        is_C = ~is_B & ~is_A
        n = len(g)
        rows.append({
            "model": model,
            "dataset": dataset,
            "config": cfg,
            "steps": n,
            "mean_acc_len": round(acc.mean(), 2),
            "pct_B_reject": round(100 * is_B.sum() / n, 1),
            "pct_C_partial": round(100 * is_C.sum() / n, 1),
            "pct_A_saturated": round(100 * is_A.sum() / n, 1),
            "mean_ypc": round(g["yield_per_compute"].mean(), 3),
        })
    out = pd.DataFrame(rows).sort_values(["model", "dataset", "config"]).reset_index(drop=True)
    return out


def hist_per_cfg(df, model, dataset, cfg, max_al=11):
    sub = df[(df.model == model) & (df.dataset == dataset) & (df.config == cfg)]
    if sub.empty:
        return None
    hist = sub["accept_length"].value_counts().sort_index()
    return hist


def oracle_shrink_gain(df, large_cfg, small_cfg):
    """
    For a step that's regime-B under `large_cfg`, the draft+verify compute is
    wasted — only the bonus token counts. If `small_cfg` were run instead, the
    same bonus token is produced at lower cost. Approximate the throughput
    gain assuming step_time scales linearly with tree_size.

    This is an *upper bound* because it assumes perfect regime-B oracle and
    ignores that switching small might change accept on non-B steps.
    """
    results = []
    for (model, dataset), g_large in df[df.config == large_cfg].groupby(["model", "dataset"], observed=True):
        g_small = df[(df.config == small_cfg) & (df.model == model) & (df.dataset == dataset)]
        if g_small.empty:
            continue
        is_B = (g_large["accept_length"] == 1)
        pct_B = is_B.sum() / len(g_large)
        # naive cost ratio from tree size
        large_tree = g_large["draft_tree_size"].mean()
        small_tree = g_small["draft_tree_size"].mean()
        # on regime-B steps, large_cfg produces 1 token, small_cfg also produces 1 token,
        # but at cost small_tree / large_tree fraction of the compute.
        # throughput gain ≈ pct_B * (1 - small_tree/large_tree)
        saved_per_step = pct_B * (1 - small_tree / large_tree) if large_tree > 0 else 0
        results.append({
            "model": model,
            "dataset": dataset,
            "large_cfg": large_cfg,
            "small_cfg": small_cfg,
            "pct_B": round(100 * pct_B, 1),
            "large_tree": round(large_tree, 1),
            "small_tree": round(small_tree, 1),
            "oracle_shrink_gain_pct": round(100 * saved_per_step, 1),
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = pd.read_parquet(PARQUET)
    print(f"loaded {len(df):,} per-step rows")
    print(f"configs: {sorted(df['config'].unique())}")
    print(f"models : {sorted(df['model'].unique())}")
    print(f"datasets: {sorted(df['dataset'].unique())}\n")

    tbl = regime_table(df)
    print("==== REGIME MIX per (model, dataset, config) ====")
    print(tbl.to_string(index=False))

    print("\n==== HISTOGRAMS of accept_length on MTBench for Llama ====")
    for cfg in ["static_3_1_4", "static_7_1_8", "static_7_4_8", "static_6_10_60", "v3_dynamic", "v6_dynamic"]:
        h = hist_per_cfg(df, "llama", "mtbench", cfg)
        if h is not None:
            print(f"\n  {cfg:18s} (n={h.sum()})")
            for k, v in h.items():
                bar = "#" * int(40 * v / h.max())
                print(f"    {int(k):2d}: {int(v):6d}  {bar}")

    print("\n==== ORACLE-SHRINK-GAIN: static_6_10_60 -> static_3_1_4 on regime-B steps ====")
    og = oracle_shrink_gain(df, "static_6_10_60", "static_3_1_4")
    print(og.to_string(index=False))

    print("\n==== ORACLE-SHRINK-GAIN: static_7_4_8 -> static_3_1_4 on regime-B steps ====")
    og2 = oracle_shrink_gain(df, "static_7_4_8", "static_3_1_4")
    print(og2.to_string(index=False))

    # Compare dynamic's config distribution: did v6 actually use many different configs?
    print("\n==== v6_dynamic config distribution on Llama MTBench ====")
    sub = df[(df.config == "v6_dynamic") & (df.model == "llama") & (df.dataset == "mtbench")]
    if not sub.empty:
        tup = list(zip(sub["chosen_num_steps"], sub["chosen_topk"], sub["chosen_num_draft_tokens"]))
        from collections import Counter
        c = Counter(tup)
        total = len(tup)
        for cfg, n in c.most_common():
            print(f"  (ns={cfg[0]}, tk={cfg[1]}, ndt={cfg[2]}) : {n:6d}  ({100*n/total:.1f}%)")
