#!/usr/bin/env python3
"""
Find the rule that determines which codeword shifts to which syndrome.

Key insight from SAT analysis:
- Syndrome 1 gets 672 codewords
- Syndromes 2,3 get 288 each
- Syndromes 12-15 get 32 each

Question: Is there a simple rule like:
- "Shift to syndrome 1 if bit 0 is 0"
- "Shift to syndrome 2 if bits 0,1 are 1,0"
etc.?

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, Dict, List, Tuple
from collections import defaultdict
from pathlib import Path


def hamming_weight(v: int) -> int:
    return bin(v).count('1')


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def compute_syndrome(v: int, k: int) -> int:
    n = (1 << k) - 1
    syndrome = 0
    for i in range(n):
        if (v >> i) & 1:
            syndrome ^= (i + 1)
    return syndrome


def generate_hamming_code(k: int) -> Set[int]:
    n = (1 << k) - 1
    return {v for v in range(1 << n) if compute_syndrome(v, k) == 0}


def has_circular_run(v: int, n: int, s: int) -> bool:
    bits = format(v, f'0{n}b')
    doubled = bits + bits[:-1]
    return '1' * s in doubled


def get_allowed_vertices(n: int, s: int) -> Set[int]:
    return {v for v in range(1 << n) if not has_circular_run(v, n, s)}


def get_bit(v: int, i: int) -> int:
    return (v >> i) & 1


def analyze_shift_rule(sat_file: str, k: int, s: int):
    """
    Try to find a deterministic rule for shift assignments.
    """
    n = (1 << k) - 1

    sat_centers = set(int(v) for v in np.load(sat_file))
    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    stayed = sat_centers & hamming
    non_hamming = sat_centers - hamming

    print(f"\n{'='*70}")
    print(f"Finding Shift Rule for n={n}, s={s}")
    print(f"{'='*70}")

    # Map each shifted center back to origin and syndrome
    shift_info = {}  # origin -> (shifted, syndrome)

    for c in non_hamming:
        for h in hamming:
            if hamming_distance(c, h) == 1:
                syn = compute_syndrome(c, k)
                shift_info[h] = (c, syn)
                break

    # Group by syndrome
    by_syndrome = defaultdict(list)
    for origin, (shifted, syn) in shift_info.items():
        by_syndrome[syn].append(origin)

    for c in stayed:
        by_syndrome[0].append(c)

    # Analyze what's common within each syndrome group
    print(f"\n--- Bit Pattern Analysis ---")

    for syn in sorted(by_syndrome.keys()):
        group = by_syndrome[syn]
        print(f"\nSyndrome {syn} ({len(group)} codewords):")

        # Analyze bit patterns
        for bit in range(n):
            ones = sum(1 for c in group if get_bit(c, bit) == 1)
            zeros = len(group) - ones
            if ones == 0 or zeros == 0:
                print(f"  Bit {bit:2d}: ALL {'1' if ones > 0 else '0'}")
            elif ones < 10 or zeros < 10:
                print(f"  Bit {bit:2d}: {ones:4d} ones, {zeros:4d} zeros (skewed)")

    # Try to find distinguishing features
    print(f"\n--- Finding Distinguishing Features ---")

    # For codewords that shift to syndrome 1 vs 2 vs 3
    group_1 = by_syndrome.get(1, [])
    group_2 = by_syndrome.get(2, [])
    group_3 = by_syndrome.get(3, [])
    group_0 = by_syndrome.get(0, [])  # stayed

    if group_1 and group_2:
        print(f"\nComparing syndrome 1 ({len(group_1)}) vs syndrome 2 ({len(group_2)}):")

        for bit in range(n):
            ones_1 = sum(1 for c in group_1 if get_bit(c, bit) == 1)
            ones_2 = sum(1 for c in group_2 if get_bit(c, bit) == 1)

            pct_1 = 100 * ones_1 / len(group_1) if group_1 else 0
            pct_2 = 100 * ones_2 / len(group_2) if group_2 else 0

            if abs(pct_1 - pct_2) > 20:
                print(f"  Bit {bit:2d}: syn1 has {pct_1:.1f}% ones, syn2 has {pct_2:.1f}% ones")

    # Check: Does the shift bit tell us something?
    print(f"\n--- Shift Bit Analysis ---")

    for syn in [1, 2, 3, 12, 13, 14, 15]:
        if syn not in by_syndrome:
            continue

        group = by_syndrome[syn]
        shift_bit = syn - 1  # bit position that was flipped

        # What's the value of the shift bit BEFORE shifting?
        ones_at_shift = sum(1 for c in group if get_bit(c, shift_bit) == 1)
        zeros_at_shift = len(group) - ones_at_shift

        print(f"  Syndrome {syn:2d} (shift bit {shift_bit:2d}): {ones_at_shift:4d} had bit=1, {zeros_at_shift:4d} had bit=0")

    # Check: Is there a pattern based on weight?
    print(f"\n--- Weight-Based Pattern ---")

    # For each weight, where do codewords go?
    weight_fate = defaultdict(lambda: defaultdict(int))
    for origin, (shifted, syn) in shift_info.items():
        w = hamming_weight(origin)
        weight_fate[w][syn] += 1

    for c in stayed:
        w = hamming_weight(c)
        weight_fate[w][0] += 1

    for c in hamming - set(shift_info.keys()) - stayed:
        w = hamming_weight(c)
        weight_fate[w]['removed'] += 1

    print(f"\n{'Weight':<8}", end="")
    all_syndromes = sorted(set(s for wf in weight_fate.values() for s in wf.keys() if isinstance(s, int)))
    for syn in all_syndromes:
        print(f"{f's{syn}':<6}", end="")
    print()

    for w in sorted(weight_fate.keys()):
        print(f"{w:<8}", end="")
        for syn in all_syndromes:
            print(f"{weight_fate[w][syn]:<6}", end="")
        print()

    # Key test: Can we PREDICT the shift from properties of the codeword?
    print(f"\n--- Prediction Test ---")
    print("Can we predict shift syndrome from codeword properties?")

    # Simple heuristic: "shift to lowest available syndrome"
    correct_predictions = 0
    total_shifted = 0

    for origin, (shifted, actual_syn) in shift_info.items():
        total_shifted += 1

        # Try to predict: find lowest syndrome where shift is valid
        predicted = None
        for test_syn in [1, 2, 3, 12, 13, 14, 15]:
            test_bit = test_syn - 1
            test_shifted = origin ^ (1 << test_bit)
            if test_shifted in allowed:
                predicted = test_syn
                break

        if predicted == actual_syn:
            correct_predictions += 1

    print(f"  'Lowest valid syndrome' rule: {correct_predictions}/{total_shifted} ({100*correct_predictions/total_shifted:.1f}%)")

    # Try: "shift to lowest syndrome that doesn't conflict with existing centers"
    # This requires knowing the full solution, so it's not truly predictive

    return by_syndrome


def main():
    """Main analysis."""
    sat_file = "/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_12_centers.npy"

    if Path(sat_file).exists():
        analyze_shift_rule(sat_file, k=4, s=12)


if __name__ == "__main__":
    main()
