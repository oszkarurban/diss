#!/usr/bin/env python3
"""
test_dynamic_adjustment.py — Unit tests for the dynamic speculative decoding
policy engine.  No GPU or running server required.

Run from repo root:
    python test_dynamic_adjustment.py
"""

import os
import sys
from types import SimpleNamespace

# ── Path setup ──────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, "sglang", "python"))

from sglang.srt.speculative.dynamic_spec import (
    AdaptiveSignalNormalizer,
    DynamicSpecPolicy,
    DynamicSpecSignals,
    _INVERTED_SIGNALS,
    _SIGNAL_KEYS,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_args(topk=1, steps=3, ndt=4, topk_max=4, steps_max=5, ndt_max=8):
    return SimpleNamespace(
        speculative_eagle_topk=topk,
        speculative_num_steps=steps,
        speculative_num_draft_tokens=ndt,
        speculative_eagle_topk_max=topk_max,
        speculative_num_steps_max=steps_max,
        speculative_num_draft_tokens_max=ndt_max,
    )


def make_signals(**kw):
    return DynamicSpecSignals(**kw)


def ok(cond, msg):
    tag = "[OK]" if cond else "[FAIL]"
    print(f"    {tag}  {msg}")
    return cond


# ── Tests ───────────────────────────────────────────────────────────────────

def test_signals_dataclass():
    """All 7 fields exist and default to 0.0."""
    s = DynamicSpecSignals()
    all_ok = True
    for key in _SIGNAL_KEYS:
        val = getattr(s, key)
        all_ok &= ok(val == 0.0, f"{key} default = {val}")
    all_ok &= ok(len(_SIGNAL_KEYS) == 7, f"signal count = {len(_SIGNAL_KEYS)}")

    custom = make_signals(draft_entropy=1.5, top1_prob=0.9)
    all_ok &= ok(custom.draft_entropy == 1.5, "custom draft_entropy = 1.5")
    all_ok &= ok(custom.top1_prob == 0.9, "custom top1_prob = 0.9")
    return all_ok


def test_normalizer_warmup():
    """First warmup_steps-1 calls return None; warmup_steps-th returns a dict."""
    warmup = 10
    norm = AdaptiveSignalNormalizer(warmup_steps=warmup)
    all_ok = True

    for i in range(1, warmup):
        sig = make_signals(
            draft_entropy=1.0 + i * 0.1,
            top1_prob=0.5 + i * 0.01,
            top1_minus_top2=0.2 + i * 0.01,
            hidden_norm=80.0 + i,
            target_entropy=1.5 + i * 0.1,
            entropy_gap=0.5 + i * 0.05,
            rolling_accept_rate=0.6 + i * 0.02,
        )
        result = norm.update_and_normalize(sig)
        all_ok &= ok(result is None, f"step {i}: returns None (warmup)")

    # The warmup_steps-th call should return a dict
    sig = make_signals(
        draft_entropy=2.0, top1_prob=0.6, top1_minus_top2=0.3,
        hidden_norm=90.0, target_entropy=2.0, entropy_gap=0.8,
        rolling_accept_rate=0.75,
    )
    result = norm.update_and_normalize(sig)
    all_ok &= ok(isinstance(result, dict), f"step {warmup}: returns dict")
    if result:
        for key in _SIGNAL_KEYS:
            v = result[key]
            all_ok &= ok(
                0.0 <= v <= 1.0,
                f"  {key} = {v:.4f} in [0, 1]",
            )
    return all_ok


def test_normalizer_inversion():
    """Inverted signals: high raw value -> low normalized value."""
    norm = AdaptiveSignalNormalizer(warmup_steps=2)

    # Feed two samples to establish a range
    lo = make_signals(
        draft_entropy=0.5, top1_prob=0.3, top1_minus_top2=0.1,
        hidden_norm=50.0, target_entropy=0.5, entropy_gap=0.1,
        rolling_accept_rate=0.4,
    )
    hi = make_signals(
        draft_entropy=4.0, top1_prob=0.95, top1_minus_top2=0.8,
        hidden_norm=180.0, target_entropy=4.0, entropy_gap=2.0,
        rolling_accept_rate=0.95,
    )
    norm.update_and_normalize(lo)  # warmup step 1 -> None

    # Step 2 = warmup_steps, returns normalized dict
    result = norm.update_and_normalize(hi)
    all_ok = True
    all_ok &= ok(result is not None, "post-warmup returns dict")
    if not result:
        return False

    # For inverted signals: feeding the MAX raw value should give normalized ~0.0
    # (because inversion: 1.0 - 1.0 = 0.0)
    for key in _INVERTED_SIGNALS:
        all_ok &= ok(
            result[key] < 0.1,
            f"{key} (inverted, fed max) = {result[key]:.4f} < 0.1",
        )

    # For non-inverted signals: feeding the MAX should give normalized ~1.0
    non_inverted = set(_SIGNAL_KEYS) - _INVERTED_SIGNALS
    for key in non_inverted:
        all_ok &= ok(
            result[key] > 0.9,
            f"{key} (not inverted, fed max) = {result[key]:.4f} > 0.9",
        )
    return all_ok


def test_confidence_extremes():
    """All-1.0 normalized -> confidence=1.0; all-0.0 -> confidence=0.0."""
    policy = DynamicSpecPolicy(make_args(), warmup_steps=1)
    all_ok = True

    high = {k: 1.0 for k in _SIGNAL_KEYS}
    low = {k: 0.0 for k in _SIGNAL_KEYS}

    c_high = policy.compute_confidence(high)
    c_low = policy.compute_confidence(low)
    all_ok &= ok(abs(c_high - 1.0) < 1e-6, f"all-1.0 -> confidence = {c_high:.6f}")
    all_ok &= ok(abs(c_low - 0.0) < 1e-6, f"all-0.0 -> confidence = {c_low:.6f}")

    mid = {k: 0.5 for k in _SIGNAL_KEYS}
    c_mid = policy.compute_confidence(mid)
    all_ok &= ok(abs(c_mid - 0.5) < 1e-6, f"all-0.5 -> confidence = {c_mid:.6f}")
    return all_ok


def test_map_param():
    """Piecewise-linear _map_param at key confidence values."""
    mp = DynamicSpecPolicy._map_param
    cases = [
        # (confidence, min, start, max, expected)
        (0.0,  1, 3, 5, 1),
        (0.25, 1, 3, 5, 2),
        (0.5,  1, 3, 5, 3),
        (0.75, 1, 3, 5, 4),
        (1.0,  1, 3, 5, 5),
        # topk range where start == min
        (0.0,  1, 1, 4, 1),
        (0.5,  1, 1, 4, 1),
        (1.0,  1, 1, 4, 4),
        # degenerate: all equal
        (0.5,  3, 3, 3, 3),
    ]
    all_ok = True
    for conf, mn, st, mx, expected in cases:
        actual = mp(conf, mn, st, mx)
        all_ok &= ok(
            actual == expected,
            f"conf={conf:.2f}  [{mn},{st},{mx}] -> expected={expected} actual={actual}",
        )
    return all_ok


def test_topk1_constraint():
    """When policy returns topk=1, ndt must equal num_steps+1."""
    # Use args where topk_start=1 and topk_max=1 so topk is always 1
    args = make_args(topk=1, steps=3, ndt=4, topk_max=1, steps_max=5, ndt_max=8)
    policy = DynamicSpecPolicy(args, warmup_steps=1)

    # Feed one warmup sample then one real sample
    policy.get_config(make_signals(
        draft_entropy=1.0, top1_prob=0.5, top1_minus_top2=0.3,
        hidden_norm=100.0, target_entropy=1.0, entropy_gap=0.0,
        rolling_accept_rate=0.7,
    ))
    topk, steps, ndt, conf = policy.get_config(make_signals(
        draft_entropy=0.5, top1_prob=0.8, top1_minus_top2=0.5,
        hidden_norm=120.0, target_entropy=0.5, entropy_gap=0.0,
        rolling_accept_rate=0.9,
    ))
    all_ok = True
    all_ok &= ok(topk == 1, f"topk = {topk} (forced to 1)")
    all_ok &= ok(ndt == steps + 1, f"ndt={ndt} == steps+1={steps + 1}")
    return all_ok


def test_tree_size_constraint():
    """ndt never exceeds the maximum tree size."""
    args = make_args(topk=2, steps=2, ndt=4, topk_max=2, steps_max=2, ndt_max=100)
    policy = DynamicSpecPolicy(args, warmup_steps=1)

    # Feed warmup then a high-confidence sample
    policy.get_config(make_signals(rolling_accept_rate=0.5))
    topk, steps, ndt, _ = policy.get_config(make_signals(
        draft_entropy=0.1, top1_prob=0.99, top1_minus_top2=0.9,
        hidden_norm=200.0, target_entropy=0.1, entropy_gap=0.0,
        rolling_accept_rate=1.0,
    ))
    # topk=2, steps=2 -> tree = 2 + 4 = 6 candidates + 1 root = 7
    max_tree = sum(topk**i for i in range(1, steps + 1)) + 1
    all_ok = True
    all_ok &= ok(ndt <= max_tree, f"ndt={ndt} <= max_tree={max_tree}")
    all_ok &= ok(ndt >= 2, f"ndt={ndt} >= 2")
    return all_ok


def test_full_pipeline():
    """End-to-end: warmup -> high/low/mid confidence -> correct configs."""
    args = make_args(topk=1, steps=3, ndt=4, topk_max=4, steps_max=5, ndt_max=8)
    policy = DynamicSpecPolicy(args, warmup_steps=10)
    all_ok = True

    # Feed warmup samples with varied signals.
    # warmup_steps=10 means step_count < 10 returns None (steps 1-9),
    # step 10 is the FIRST active step (returns a config).
    import random
    random.seed(42)
    warmup_steps = 10
    for i in range(warmup_steps - 1):  # 9 warmup steps
        sig = make_signals(
            draft_entropy=random.uniform(0.3, 4.0),
            top1_prob=random.uniform(0.2, 0.95),
            top1_minus_top2=random.uniform(0.05, 0.8),
            hidden_norm=random.uniform(40.0, 200.0),
            target_entropy=random.uniform(0.3, 4.0),
            entropy_gap=random.uniform(-1.0, 1.0),
            rolling_accept_rate=random.uniform(0.3, 1.0),
        )
        topk, steps, ndt, conf = policy.get_config(sig)
        all_ok &= ok(conf is None, f"warmup step {i+1}: conf=None, config=({topk},{steps},{ndt})")

    # One more varied sample to complete the warmup range
    sig = make_signals(
        draft_entropy=random.uniform(0.3, 4.0),
        top1_prob=random.uniform(0.2, 0.95),
        top1_minus_top2=random.uniform(0.05, 0.8),
        hidden_norm=random.uniform(40.0, 200.0),
        target_entropy=random.uniform(0.3, 4.0),
        entropy_gap=random.uniform(-1.0, 1.0),
        rolling_accept_rate=random.uniform(0.3, 1.0),
    )
    topk, steps, ndt, conf = policy.get_config(sig)
    all_ok &= ok(conf is not None, f"step {warmup_steps}: first active step, conf={conf:.3f}")

    # Now read the normalizer's learned range to construct precise test signals
    n = policy.normalizer

    # HIGH confidence: signals that normalize to ~1.0
    # For inverted signals (entropy): use minimum raw value -> normalizes to 1.0 -> high
    # For non-inverted: use maximum raw value -> normalizes to 1.0 -> high
    high_sig = make_signals(
        draft_entropy=n.running_min["draft_entropy"],
        top1_prob=n.running_max["top1_prob"],
        top1_minus_top2=n.running_max["top1_minus_top2"],
        hidden_norm=n.running_max["hidden_norm"],
        target_entropy=n.running_min["target_entropy"],
        entropy_gap=n.running_min["entropy_gap"],  # abs value used, min abs = high conf
        rolling_accept_rate=n.running_max["rolling_accept_rate"],
    )
    topk_h, steps_h, ndt_h, conf_h = policy.get_config(high_sig)
    all_ok &= ok(conf_h is not None, f"high conf: conf={conf_h}")
    all_ok &= ok(conf_h > 0.7, f"  confidence={conf_h:.3f} > 0.7")
    all_ok &= ok(topk_h >= args.speculative_eagle_topk, f"  topk={topk_h} >= start={args.speculative_eagle_topk}")
    all_ok &= ok(steps_h >= args.speculative_num_steps, f"  steps={steps_h} >= start={args.speculative_num_steps}")
    print(f"    -> config = (topk={topk_h}, steps={steps_h}, ndt={ndt_h})")

    # LOW confidence: signals that normalize to ~0.0
    low_sig = make_signals(
        draft_entropy=n.running_max["draft_entropy"],
        top1_prob=n.running_min["top1_prob"],
        top1_minus_top2=n.running_min["top1_minus_top2"],
        hidden_norm=n.running_min["hidden_norm"],
        target_entropy=n.running_max["target_entropy"],
        entropy_gap=n.running_max["entropy_gap"],
        rolling_accept_rate=n.running_min["rolling_accept_rate"],
    )
    topk_l, steps_l, ndt_l, conf_l = policy.get_config(low_sig)
    all_ok &= ok(conf_l is not None, f"low conf:  conf={conf_l}")
    all_ok &= ok(conf_l < 0.3, f"  confidence={conf_l:.3f} < 0.3")
    all_ok &= ok(topk_l <= args.speculative_eagle_topk, f"  topk={topk_l} <= start={args.speculative_eagle_topk}")
    all_ok &= ok(steps_l <= args.speculative_num_steps, f"  steps={steps_l} <= start={args.speculative_num_steps}")
    print(f"    -> config = (topk={topk_l}, steps={steps_l}, ndt={ndt_l})")

    # MID confidence: midpoint of observed ranges
    mid_sig = make_signals(
        draft_entropy=(n.running_min["draft_entropy"] + n.running_max["draft_entropy"]) / 2,
        top1_prob=(n.running_min["top1_prob"] + n.running_max["top1_prob"]) / 2,
        top1_minus_top2=(n.running_min["top1_minus_top2"] + n.running_max["top1_minus_top2"]) / 2,
        hidden_norm=(n.running_min["hidden_norm"] + n.running_max["hidden_norm"]) / 2,
        target_entropy=(n.running_min["target_entropy"] + n.running_max["target_entropy"]) / 2,
        entropy_gap=(n.running_min["entropy_gap"] + n.running_max["entropy_gap"]) / 2,
        rolling_accept_rate=(n.running_min["rolling_accept_rate"] + n.running_max["rolling_accept_rate"]) / 2,
    )
    topk_m, steps_m, ndt_m, conf_m = policy.get_config(mid_sig)
    all_ok &= ok(conf_m is not None, f"mid conf:  conf={conf_m}")
    all_ok &= ok(0.3 <= conf_m <= 0.7, f"  confidence={conf_m:.3f} in [0.3, 0.7]")
    print(f"    -> config = (topk={topk_m}, steps={steps_m}, ndt={ndt_m})")

    # Verify ordering: high conf should have >= params than low conf
    all_ok &= ok(
        topk_h >= topk_l and steps_h >= steps_l,
        f"  high config >= low config: ({topk_h},{steps_h}) >= ({topk_l},{steps_l})",
    )
    return all_ok


# ── Runner ──────────────────────────────────────────────────────────────────

def main():
    tests = [
        ("DynamicSpecSignals dataclass", test_signals_dataclass),
        ("Normalizer warmup period", test_normalizer_warmup),
        ("Normalizer signal inversion", test_normalizer_inversion),
        ("Confidence extremes (0, 0.5, 1)", test_confidence_extremes),
        ("Piecewise-linear _map_param", test_map_param),
        ("topk=1 constraint (ndt=steps+1)", test_topk1_constraint),
        ("Tree size constraint", test_tree_size_constraint),
        ("Full pipeline (warmup + synthetic)", test_full_pipeline),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'─' * 60}")
        print(f"  TEST: {name}")
        print(f"{'─' * 60}")
        try:
            result = fn()
            if result:
                print(f"  >>> PASS")
                passed += 1
            else:
                print(f"  >>> FAIL")
                failed += 1
        except Exception as e:
            print(f"  >>> ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
