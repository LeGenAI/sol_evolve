#!/usr/bin/env python3
"""
Analyze the shift pattern in SAT solution.

Key questions:
1. Which Hamming codewords were shifted to become non-Hamming centers?
2. What bit positions were used for shifts?
3. Is there a pattern we can exploit?

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
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


def generate_hamming_code(k: int) -> set:
    n = (1 << k) - 1
    return {v for v in range(1 << n) if compute_syndrome(v, k) == 0}


def analyze_shift_correspondence(sat_centers: set, hamming: set, k: int, n: int):
    """
    For each SAT center not in Hamming, find which Hamming codeword it came from.
    """
    print(f"\n{'='*70}")
    print("Shift Correspondence Analysis")
    print("="*70)

    non_hamming_sat = sat_centers - hamming
    only_hamming = hamming - sat_centers

    print(f"\n|SAT centers| = {len(sat_centers)}")
    print(f"|Hamming| = {len(hamming)}")
    print(f"|SAT ∩ Hamming| = {len(sat_centers & hamming)}")
    print(f"|SAT - Hamming| (shifted centers) = {len(non_hamming_sat)}")
    print(f"|Hamming - SAT| (replaced codewords) = {len(only_hamming)}")

    # For each non-Hamming SAT center, find its "origin" Hamming codeword
    # (the one at distance 1)
    origin_map = {}  # non_hamming_center -> hamming_codeword
    shift_positions = defaultdict(list)  # bit_position -> list of shifts

    for c in non_hamming_sat:
        # Find Hamming codeword at distance 1
        origins = []
        for h in hamming:
            d = hamming_distance(c, h)
            if d == 1:
                # Find which bit was flipped
                diff = c ^ h
                bit_pos = (diff).bit_length() - 1
                origins.append((h, bit_pos))

        if len(origins) == 1:
            h, bit_pos = origins[0]
            origin_map[c] = (h, bit_pos)
            shift_positions[bit_pos].append((h, c))
        elif len(origins) > 1:
            # Multiple possible origins (unlikely for random shifts)
            origin_map[c] = origins

    # Analyze shift positions
    print(f"\n--- Shift Position Distribution ---")
    for pos in sorted(shift_positions.keys()):
        print(f"  Bit {pos}: {len(shift_positions[pos])} shifts")

    # Check if replaced Hamming codewords match origins
    replaced = only_hamming
    origins_used = {origin_map[c][0] for c in non_hamming_sat if isinstance(origin_map.get(c), tuple)}

    print(f"\n--- Origin Analysis ---")
    print(f"Replaced Hamming codewords: {len(replaced)}")
    print(f"Origins of non-Hamming centers: {len(origins_used)}")
    print(f"Match (replaced == origins): {replaced == origins_used}")

    if replaced != origins_used:
        extra_replaced = replaced - origins_used
        missing_origins = origins_used - replaced
        print(f"  Extra replaced: {len(extra_replaced)}")
        print(f"  Missing origins: {len(missing_origins)}")

    # Key insight: Is it a permutation? (1-to-1 correspondence)
    # Each replaced Hamming codeword should have exactly one shifted version
    origin_to_shifted = defaultdict(list)
    for c, info in origin_map.items():
        if isinstance(info, tuple):
            h, _ = info
            origin_to_shifted[h].append(c)

    print(f"\n--- Shift Pattern ---")
    multi_shifted = {h: shifts for h, shifts in origin_to_shifted.items() if len(shifts) > 1}
    print(f"Codewords with multiple shifted versions: {len(multi_shifted)}")

    # Check if it's truly 1-to-1
    if len(origin_to_shifted) == len(non_hamming_sat):
        print("Pattern: Each non-Hamming center comes from a unique Hamming codeword")
    else:
        print(f"Pattern: Some Hamming codewords produced multiple shifted centers")

    return origin_map, shift_positions


def analyze_which_codewords_moved(sat_centers: set, hamming: set, k: int, n: int, s: int):
    """
    Analyze which Hamming codewords stayed vs moved.
    """
    print(f"\n{'='*70}")
    print("Codeword Movement Analysis")
    print("="*70)

    stayed = sat_centers & hamming
    moved = hamming - sat_centers

    # Syndrome distribution of stayed vs moved
    stayed_syn = defaultdict(int)
    for c in stayed:
        stayed_syn[compute_syndrome(c, k)] += 1

    moved_syn = defaultdict(int)
    for c in moved:
        moved_syn[compute_syndrome(c, k)] += 1

    print(f"\nCodewords that stayed in place: {len(stayed)} (all have syndrome 0)")
    print(f"Codewords that were replaced: {len(moved)}")

    # What makes a codeword "need to move"?
    # Hypothesis: Those whose ball intersects forbidden region

    def has_circular_run(v: int, n: int, s: int) -> bool:
        bits = format(v, f'0{n}b')
        doubled = bits + bits[:-1]
        return '1' * s in doubled

    forbidden = {v for v in range(1 << n) if has_circular_run(v, n, s)}

    stayed_with_forbidden_neighbors = 0
    moved_with_forbidden_neighbors = 0

    for c in stayed:
        ball = {c ^ (1 << i) for i in range(n)} | {c}
        if ball & forbidden:
            stayed_with_forbidden_neighbors += 1

    for c in moved:
        ball = {c ^ (1 << i) for i in range(n)} | {c}
        if ball & forbidden:
            moved_with_forbidden_neighbors += 1

    print(f"\n--- Forbidden Neighbor Analysis ---")
    print(f"Stayed with forbidden in ball: {stayed_with_forbidden_neighbors} / {len(stayed)}")
    print(f"Moved with forbidden in ball: {moved_with_forbidden_neighbors} / {len(moved)}")

    # Weight distribution
    stayed_wt = defaultdict(int)
    for c in stayed:
        stayed_wt[hamming_weight(c)] += 1

    moved_wt = defaultdict(int)
    for c in moved:
        moved_wt[hamming_weight(c)] += 1

    print(f"\n--- Weight Distribution ---")
    print(f"{'Weight':<10} {'Stayed':<10} {'Moved':<10}")
    print("-" * 30)
    for w in sorted(set(stayed_wt.keys()) | set(moved_wt.keys())):
        print(f"{w:<10} {stayed_wt[w]:<10} {moved_wt[w]:<10}")


def main():
    """Main analysis."""
    # Load SAT solutions
    for s_val in [12, 11]:
        centers_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s_val}_centers.npy"

        if not Path(centers_file).exists():
            print(f"File not found: {centers_file}")
            continue

        print(f"\n{'#'*70}")
        print(f"# n=15, s={s_val}")
        print(f"{'#'*70}")

        centers_arr = np.load(centers_file)
        sat_centers = set(int(v) for v in centers_arr)

        k = 4
        n = 15
        hamming = generate_hamming_code(k)

        analyze_shift_correspondence(sat_centers, hamming, k, n)
        analyze_which_codewords_moved(sat_centers, hamming, k, n, s_val)


if __name__ == "__main__":
    main()
