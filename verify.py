from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from tanglecode.anomaly import detect
from tanglecode.braid import (
    order_from_frame,
    reduce_crossings,
    replay_orders,
    word_from_frames,
)
from tanglecode.codec import decode, encode
from tanglecode.data import generate_signals


def run_pipeline(data: np.ndarray, params: dict):
    T, n = data.shape
    initial_order, raw = word_from_frames(data)
    reduced, cancelled = reduce_crossings(raw)
    positions = replay_orders(initial_order, raw, T)

    detection = detect(
        reduced_crossings=reduced,
        positions=positions,
        n=n,
        T=T,
        window=params["window"],
        step=params["step"],
        k=params["k"],
        min_consecutive=params["min_consecutive"],
        baseline_frac=params["baseline_frac"],
        min_baseline_windows=params["min_baseline_windows"],
    )

    tngl_bytes, stats = encode(data)
    dec_n, dec_T, dec_anchor, dec_reduced = decode(tngl_bytes)
    dec_positions = replay_orders(dec_anchor, dec_reduced, dec_T)
    replayed_final = np.argsort(dec_positions[-1]).tolist()
    true_final = order_from_frame(data[-1])

    return {
        "raw": raw,
        "reduced": reduced,
        "cancelled": cancelled,
        "positions": positions,
        "detection": detection,
        "tngl_bytes": tngl_bytes,
        "stats": stats,
        "replayed_final": replayed_final,
        "true_final": true_final,
        "roundtrip_ok": (replayed_final == true_final),
    }


def main():
    print("=" * 70)
    print("  TangleCode Verification Suite")
    print("=" * 70)

    # Final tuned parameters
    params = {
        "window": 200,
        "step": 100,
        "k": 4.0,
        "min_consecutive": 4,
        "baseline_frac": 0.2,
        "min_baseline_windows": 8,
    }

    all_passed = True

    # 1. Build test datasets via data.py
    print("\n[Building Datasets via data.py]")
    clean_data = generate_signals("clean", n=6, T=3000, seed=7)
    drift_data = generate_signals("drift", n=6, T=3000, seed=7)
    stuck_data = generate_signals("stuck", n=6, T=3000, seed=7)
    print(f"  Clean: shape {clean_data.shape}")
    print(f"  Drift: shape {drift_data.shape}")
    print(f"  Stuck: shape {stuck_data.shape}")

    # Run pipelines
    res_clean = run_pipeline(clean_data, params)
    res_drift = run_pipeline(drift_data, params)
    res_stuck = run_pipeline(stuck_data, params)

    # ---------------------------------------------------------
    # Check 1: clean - zero flagged ranges
    # ---------------------------------------------------------
    clean_ranges = res_clean["detection"]["ranges"]
    check1_pass = (len(clean_ranges) == 0)
    status1 = "PASS" if check1_pass else "FAIL"
    print(f"\n1. Clean Dataset Check: [{status1}]")
    print(f"   Expected: 0 flagged ranges")
    print(f"   Actual:   {len(clean_ranges)} flagged ranges -> {clean_ranges}")
    if not check1_pass:
        all_passed = False

    # ---------------------------------------------------------
    # Check 2: drift - range starting between tick 1300 and 1700 with faulty_channel == 4
    # ---------------------------------------------------------
    drift_ranges = res_drift["detection"]["ranges"]
    matching_drift = [
        r for r in drift_ranges
        if 1300 <= r["start_tick"] <= 1700 and r["faulty_channel"] == 4
    ]
    check2_pass = (len(matching_drift) > 0)
    status2 = "PASS" if check2_pass else "FAIL"
    print(f"\n2. Drift Dataset Check: [{status2}]")
    print(f"   Expected: Range starting in [1300, 1700] with faulty_channel == 4")
    print(f"   Actual:   Flagged ranges -> {drift_ranges}")
    if matching_drift:
        print(f"   Matching range found: {matching_drift[0]}")
        print(f"   Fault details: kind='{matching_drift[0]['kind']}', "
              f"rank_before={matching_drift[0]['rank_before']}, rank_after={matching_drift[0]['rank_after']}, "
              f"severity='{matching_drift[0]['severity']}'")
    if not check2_pass:
        all_passed = False

    # ---------------------------------------------------------
    # Check 3: stuck - range starting between tick 1300 and 1700 with faulty_channel == 4
    # ---------------------------------------------------------
    stuck_ranges = res_stuck["detection"]["ranges"]
    matching_stuck = [
        r for r in stuck_ranges
        if 1300 <= r["start_tick"] <= 1700 and r["faulty_channel"] == 4
    ]
    check3_pass = (len(matching_stuck) > 0)
    status3 = "PASS" if check3_pass else "FAIL"
    print(f"\n3. Stuck Dataset Check: [{status3}]")
    print(f"   Expected: Range starting in [1300, 1700] with faulty_channel == 4")
    print(f"   Actual:   Flagged ranges -> {stuck_ranges}")
    if matching_stuck:
        print(f"   Matching range found: {matching_stuck[0]}")
        print(f"   Fault details: kind='{matching_stuck[0]['kind']}', "
              f"rank_before={matching_stuck[0]['rank_before']}, rank_after={matching_stuck[0]['rank_after']}, "
              f"severity='{matching_stuck[0]['severity']}'")
    if not check3_pass:
        all_passed = False

    # ---------------------------------------------------------
    # Check 4: all three - reduced <= raw, cancelled*2 == raw - reduced, and .tngl round trip
    # ---------------------------------------------------------
    print("\n4. Invariants & Codec Round Trip Check:")
    datasets_map = {
        "clean": res_clean,
        "drift": res_drift,
        "stuck": res_stuck,
    }

    check4_all_pass = True
    for name, r in datasets_map.items():
        n_raw = len(r["raw"])
        n_red = len(r["reduced"])
        n_canc = r["cancelled"]
        inv1 = (n_red <= n_raw)
        inv2 = (n_raw - n_red == 2 * n_canc)
        rt_ok = r["roundtrip_ok"]
        pass_name = inv1 and inv2 and rt_ok
        status_name = "PASS" if pass_name else "FAIL"
        print(f"   - {name:5s}: [{status_name}] raw={n_raw}, reduced={n_red}, cancelled={n_canc}, "
              f"reduced<=raw: {inv1}, 2*cancelled==diff: {inv2}, .tngl roundtrip: {rt_ok}")
        if not pass_name:
            check4_all_pass = False

    if not check4_all_pass:
        all_passed = False

    # ---------------------------------------------------------
    # 5. Clean false alarm analysis across 5 other seeds
    # ---------------------------------------------------------
    print("\n[Clean False Alarm Analysis across 5 Other Seeds]")
    test_seeds = [1, 2, 42, 100, 999]
    false_alarm_count = 0
    for seed in test_seeds:
        seed_data = generate_signals("clean", n=6, T=3000, seed=seed)
        seed_res = run_pipeline(seed_data, params)
        seed_ranges = seed_res["detection"]["ranges"]
        has_fa = len(seed_ranges) > 0
        if has_fa:
            false_alarm_count += 1
        fa_tag = "FALSE ALARM" if has_fa else "CLEAN"
        print(f"   Seed {seed:3d}: [{fa_tag:11s}] -> {len(seed_ranges)} ranges ({seed_ranges})")

    print(f"\n   Total false alarms: {false_alarm_count} / {len(test_seeds)} seeds ({false_alarm_count/len(test_seeds)*100:.1f}%)")

    # Final parameters report
    print("\n" + "=" * 70)
    print("  FINAL PARAMETERS:")
    for k, v in params.items():
        print(f"    {k:20s}: {v}")
    print("=" * 70)

    if all_passed:
        print("\nOVERALL RESULT: ALL CHECKS PASSED!\n")
        return 0
    else:
        print("\nOVERALL RESULT: SOME CHECKS FAILED!\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
