#!/usr/bin/env python3
"""
train_decision_tree.py — Train a decision tree to predict the optimal
speculative decoding config (topk, num_steps) from runtime signals.

Uses all collected signal data from V1/V2/V1+CB runs to learn which
(topk, num_steps) config maximizes throughput efficiency for a given
signal state. The trained tree can replace the hand-tuned mapping in
DynamicSpecPolicy.get_config().

Usage:
    python train_decision_tree.py [--max-depth 5] [--output tree_policy.json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ── Data loading ──────────────────────────────────────────────────────────────

# Signal features used as input to the decision tree.
# These are the non-zero signals available at decision time.
FEATURE_KEYS = [
    "top1_prob",
    "target_top1_prob",
    "draft_oracle_gate",
    "target_oracle_gate_fixed",
    "joint_confidence_product_fixed",
    "confidence_agreement",
    "rolling_accept_rate",
    "rolling_accept_length",
]

# All signal data files to use for training
DATA_FILES = [
    # V1 runs (inverted topk, best hand-tuned policy)
    "signal_data_dynamic_talon_llama_314_548.json",
    "signal_data_dynamic_talon_llama_516_748.json",
    "signal_data_dynamic_talon_deepseek_314_548.json",
    "signal_data_dynamic_talon_deepseek_516_748.json",
    # V2 runs (depth-first, worse but different configs explored)
    "signal_data_v2_llama_314_748.json",
    "signal_data_v2_llama_516_748.json",
    "signal_data_v2_deepseek_314_748.json",
    "signal_data_v2_deepseek_516_748.json",
    # V1+CB runs (if available)
    "signal_data_v1cb_llama_314_748.json",
    "signal_data_v1cb_llama_516_748.json",
    "signal_data_v1cb_deepseek_314_748.json",
    "signal_data_v1cb_deepseek_516_748.json",
]


def load_all_steps(data_dir: str) -> List[Dict]:
    """Load all steps from all available signal data files."""
    all_steps = []
    for fname in DATA_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        steps = [s for turn in data.get("per_turn_logs", []) for s in turn]
        # Tag each step with its source file for analysis
        for s in steps:
            s["_source"] = fname
        all_steps.extend(steps)
        print(f"  Loaded {len(steps):>6d} steps from {fname}")
    print(f"  Total: {len(all_steps)} steps")
    return all_steps


# ── Efficiency computation ────────────────────────────────────────────────────

def compute_efficiency(accept_length: float, ndt: int) -> float:
    """Throughput proxy: tokens gained per draft token verified.

    efficiency = (accept_length + 1) / ndt

    Higher is better. The +1 accounts for the bonus token from verify
    (even acc=0 produces 1 output token from the target model).
    """
    return (accept_length + 1) / max(ndt, 1)


def find_best_config_per_signal_bucket(steps: List[Dict]) -> Dict:
    """For each signal state, find which config had the best efficiency.

    This is the "oracle" analysis: given signals, which config would
    have been optimal? We use this to create training labels.
    """
    # Group steps by (chosen_topk, chosen_num_steps) and compute
    # average efficiency per config per signal bucket
    configs = defaultdict(list)
    for s in steps:
        key = (s["chosen_topk"], s["chosen_num_steps"])
        eff = compute_efficiency(s["accept_length"], s["chosen_num_draft_tokens"])
        configs[key].append(eff)

    print(f"\n  Config efficiency across all data:")
    print(f"  {'Config (tk,ns)':<15s} {'N':>7s} {'MeanEff':>8s} {'MeanAcc':>8s}")
    sorted_configs = sorted(configs.keys())
    for key in sorted_configs:
        effs = configs[key]
        steps_for_cfg = [s for s in steps
                         if (s["chosen_topk"], s["chosen_num_steps"]) == key]
        mean_acc = sum(s["accept_length"] for s in steps_for_cfg) / len(steps_for_cfg)
        print(f"  {str(key):<15s} {len(effs):>7d} {sum(effs)/len(effs):>8.4f} {mean_acc:>8.3f}")

    return configs


# ── Decision tree training ────────────────────────────────────────────────────

def train_tree(steps: List[Dict], max_depth: int = 5):
    """Train a decision tree to predict the best config from signals.

    Approach: For each step, compute efficiency. Train a regression tree
    to predict efficiency from signals. Then at inference time, evaluate
    the tree for each possible config and pick the best.

    Simpler approach used here: classify each step into the config that
    was used, weighted by efficiency. Steps with high efficiency for a
    config become positive examples for that config.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        print("ERROR: scikit-learn not available. Install with: pip install scikit-learn")
        sys.exit(1)

    # ── Approach 1: Predict efficiency directly (regression) ──────────
    print("\n" + "=" * 70)
    print("  APPROACH 1: Regression — predict efficiency from signals")
    print("=" * 70)

    X = np.array([[s[k] for k in FEATURE_KEYS] for s in steps])
    efficiencies = np.array([
        compute_efficiency(s["accept_length"], s["chosen_num_draft_tokens"])
        for s in steps
    ])
    accept_lengths = np.array([s["accept_length"] for s in steps])

    # Regression tree: predict efficiency
    reg_tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50)
    reg_tree.fit(X, efficiencies)

    cv_scores = cross_val_score(reg_tree, X, efficiencies, cv=5, scoring="r2")
    print(f"  R² (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Also try predicting accept_length directly
    reg_tree_acc = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50)
    reg_tree_acc.fit(X, accept_lengths)
    cv_scores_acc = cross_val_score(reg_tree_acc, X, accept_lengths, cv=5, scoring="r2")
    print(f"  R² for accept_length: {cv_scores_acc.mean():.4f} ± {cv_scores_acc.std():.4f}")

    print(f"\n  Efficiency regression tree (depth={max_depth}):")
    print(export_text(reg_tree, feature_names=FEATURE_KEYS, max_depth=3))

    # ── Approach 2: Classify into config buckets ──────────────────────
    print("=" * 70)
    print("  APPROACH 2: Classification — predict best config from signals")
    print("=" * 70)

    # Create config labels: (topk, num_steps) as string
    config_labels = np.array([
        f"{s['chosen_topk']},{s['chosen_num_steps']}"
        for s in steps
    ])
    unique_configs = sorted(set(config_labels))
    print(f"  Unique configs: {unique_configs}")
    print(f"  Config distribution:")
    for cfg in unique_configs:
        n = sum(1 for c in config_labels if c == cfg)
        print(f"    {cfg}: {n} ({100*n/len(steps):.1f}%)")

    clf_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=30)
    clf_tree.fit(X, config_labels)

    cv_scores_clf = cross_val_score(clf_tree, X, config_labels, cv=5)
    print(f"\n  Accuracy (5-fold CV): {cv_scores_clf.mean():.4f} ± {cv_scores_clf.std():.4f}")

    print(f"\n  Classification tree (depth={max_depth}):")
    print(export_text(clf_tree, feature_names=FEATURE_KEYS, max_depth=3))

    # ── Approach 3: Optimal config assignment ─────────────────────────
    # For each signal bucket, find which config had the BEST efficiency.
    # Then train a tree on (signals → best_config).
    print("=" * 70)
    print("  APPROACH 3: Oracle — what config SHOULD have been used?")
    print("=" * 70)

    # Bucket signals into quantiles and find best config per bucket
    # Use top1_prob and target_top1_prob as 2D grid
    n_buckets = 5
    top1_quantiles = np.quantile(X[:, 0], np.linspace(0, 1, n_buckets + 1))
    target_quantiles = np.quantile(X[:, 1], np.linspace(0, 1, n_buckets + 1))

    print(f"\n  Best config per (top1_prob, target_top1_prob) bucket:")
    print(f"  {'top1_prob':<15s} {'target_top1':<15s} {'N':>5s} {'BestCfg':<12s} {'BestEff':>8s} {'MeanAcc':>8s}")

    oracle_labels = []
    for i in range(len(steps)):
        s = steps[i]
        eff = efficiencies[i]
        oracle_labels.append(config_labels[i])  # fallback: use actual config

    # For each step, check all configs that appeared with similar signals
    # and assign the config that had the highest average efficiency
    # Group by signal similarity (using quantile buckets)
    from collections import defaultdict
    bucket_configs = defaultdict(lambda: defaultdict(list))

    for i in range(len(steps)):
        # Create a bucket key from top1_prob and target_top1_prob (most predictive)
        t1 = int(np.searchsorted(top1_quantiles[1:-1], X[i, 0]))
        tt = int(np.searchsorted(target_quantiles[1:-1], X[i, 1]))
        bucket_key = (t1, tt)
        cfg = config_labels[i]
        bucket_configs[bucket_key][cfg].append(efficiencies[i])

    # For each bucket, find the best config
    best_per_bucket = {}
    for bucket_key, cfg_effs in sorted(bucket_configs.items()):
        best_cfg = None
        best_eff = -1
        for cfg, effs in cfg_effs.items():
            mean_eff = sum(effs) / len(effs)
            if mean_eff > best_eff:
                best_eff = mean_eff
                best_cfg = cfg
        best_per_bucket[bucket_key] = best_cfg
        total_n = sum(len(e) for e in cfg_effs.values())
        mean_acc_bucket = sum(
            accept_lengths[j] for j in range(len(steps))
            if (int(np.searchsorted(top1_quantiles[1:-1], X[j, 0])),
                int(np.searchsorted(target_quantiles[1:-1], X[j, 1]))) == bucket_key
        ) / max(total_n, 1)
        print(f"  [{bucket_key[0]}/{n_buckets-1}]         [{bucket_key[1]}/{n_buckets-1}]          {total_n:>5d} {best_cfg:<12s} {best_eff:>8.4f} {mean_acc_bucket:>8.3f}")

    # Create oracle labels: for each step, use the best config for its bucket
    oracle_labels = []
    for i in range(len(steps)):
        t1 = int(np.searchsorted(top1_quantiles[1:-1], X[i, 0]))
        tt = int(np.searchsorted(target_quantiles[1:-1], X[i, 1]))
        oracle_labels.append(best_per_bucket.get((t1, tt), config_labels[i]))

    oracle_labels = np.array(oracle_labels)

    # Train tree on oracle labels
    oracle_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=30)
    oracle_tree.fit(X, oracle_labels)

    cv_scores_oracle = cross_val_score(oracle_tree, X, oracle_labels, cv=5)
    print(f"\n  Oracle tree accuracy (5-fold CV): {cv_scores_oracle.mean():.4f} ± {cv_scores_oracle.std():.4f}")

    print(f"\n  Oracle tree (depth={max_depth}):")
    print(export_text(oracle_tree, feature_names=FEATURE_KEYS, max_depth=4))

    # ── Feature importance ────────────────────────────────────────────
    print("=" * 70)
    print("  FEATURE IMPORTANCE (across all 3 approaches)")
    print("=" * 70)
    print(f"  {'Feature':<35s} {'Regression':>10s} {'Classif':>10s} {'Oracle':>10s}")
    for j, feat in enumerate(FEATURE_KEYS):
        print(f"  {feat:<35s} {reg_tree.feature_importances_[j]:>10.4f} "
              f"{clf_tree.feature_importances_[j]:>10.4f} "
              f"{oracle_tree.feature_importances_[j]:>10.4f}")

    # ── Export trees ──────────────────────────────────────────────────
    return reg_tree, clf_tree, oracle_tree


def export_tree_as_python(tree, feature_names, output_path):
    """Export decision tree as a Python function for embedding in dynamic_spec.py."""
    from sklearn.tree import export_text

    tree_text = export_text(tree, feature_names=feature_names)

    # Also generate if-else code
    from sklearn.tree import _tree

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        lines = []

        def recurse(node, depth):
            indent = "    " * (depth + 1)
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                lines.append(f"{indent}if signals.{name} <= {threshold:.4f}:")
                recurse(tree_.children_left[node], depth + 1)
                lines.append(f"{indent}else:  # {name} > {threshold:.4f}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                # Leaf node
                if hasattr(tree, 'classes_'):
                    # Classification tree
                    class_idx = tree_.value[node].argmax()
                    prediction = tree.classes_[class_idx]
                    lines.append(f'{indent}return "{prediction}"  # n={int(tree_.n_node_samples[node])}')
                else:
                    # Regression tree
                    prediction = tree_.value[node][0][0]
                    lines.append(f"{indent}return {prediction:.4f}  # n={int(tree_.n_node_samples[node])}")

        lines.append("def tree_predict(signals):")
        recurse(0, 0)
        return "\n".join(lines)

    code = tree_to_code(tree, feature_names)

    with open(output_path, "w") as f:
        f.write("# Auto-generated decision tree policy\n")
        f.write(f"# Feature keys: {feature_names}\n\n")
        f.write(code)
        f.write("\n")

    print(f"\n  Exported tree as Python to: {output_path}")
    return code


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train decision tree for dynamic spec policy")
    parser.add_argument("--data-dir", default=".", help="Directory containing signal_data_*.json files")
    parser.add_argument("--max-depth", type=int, default=5, help="Max tree depth (default: 5)")
    parser.add_argument("--output", default="tree_policy.py", help="Output Python file for tree")
    args = parser.parse_args()

    print("=" * 70)
    print("  DECISION TREE TRAINING FOR DYNAMIC SPECULATIVE DECODING")
    print("=" * 70)

    # Load data
    print("\n  Loading signal data...")
    steps = load_all_steps(args.data_dir)

    if not steps:
        print("  ERROR: No signal data found!")
        sys.exit(1)

    # Filter out steps without accept_length
    steps = [s for s in steps if "accept_length" in s]
    print(f"  Steps with accept_length: {len(steps)}")

    # Analyze config efficiency
    find_best_config_per_signal_bucket(steps)

    # Train trees
    reg_tree, clf_tree, oracle_tree = train_tree(steps, max_depth=args.max_depth)

    # Export the oracle tree (best approach) as Python code
    code = export_tree_as_python(oracle_tree, FEATURE_KEYS, args.output)
    print(f"\n  Generated policy code:\n")
    print(code)


if __name__ == "__main__":
    main()
