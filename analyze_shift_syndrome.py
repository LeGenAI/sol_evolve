#!/usr/bin/env python3
"""
Analyze the relationship between shift patterns and syndromes.

Key observation: Shifting by bit i changes syndrome to i+1 (for Hamming code).
This might explain the shift pattern!

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, Dict
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


def analyze_syndrome_shift_relation(sat_file: str, k: int, s: int):
    """
    Analyze how syndromes relate to shift directions.

    Key insight: If we shift a Hamming codeword c (syndrome 0) by bit i,
    the result c XOR e_i has syndrome i+1.

    This means the SAT solution assigns syndromes to shifted codewords!
    """
    n = (1 << k) - 1

    sat_centers = set(int(v) for v in np.load(sat_file))
    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    stayed = sat_centers & hamming
    non_hamming = sat_centers - hamming

    print(f"\n{'='*70}")
    print(f"Syndrome-Shift Analysis for n={n}, s={s}")
    print(f"{'='*70}")

    print(f"\n|SAT centers| = {len(sat_centers)}")
    print(f"|Stayed (syndrome 0)| = {len(stayed)}")
    print(f"|Shifted (non-zero syndrome)| = {len(non_hamming)}")

    # Analyze syndrome distribution of shifted centers
    syndrome_dist = defaultdict(int)
    for c in non_hamming:
        syn = compute_syndrome(c, k)
        syndrome_dist[syn] += 1

    print(f"\n--- Syndrome Distribution of Non-Hamming Centers ---")
    print(f"Syndrome: Count")
    for syn in sorted(syndrome_dist.keys()):
        print(f"  {syn:2d}: {syndrome_dist[syn]:4d}")

    # Key insight: syndrome i+1 corresponds to shifting by bit i
    print(f"\n--- Interpretation ---")
    print("Syndrome i+1 means 'shifted by bit i':")
    for syn in sorted(syndrome_dist.keys()):
        bit = syn - 1  # bit position that was flipped
        print(f"  Syndrome {syn:2d} → bit {bit:2d} flipped → {syndrome_dist[syn]:4d} codewords")

    # Which codewords got which syndromes?
    print(f"\n--- Who Got Which Syndrome? ---")

    # Map shifted center back to origin
    origin_map = {}  # non_hamming_center -> (origin_hamming, bit_flipped)
    for c in non_hamming:
        for h in hamming:
            if hamming_distance(c, h) == 1:
                diff = c ^ h
                bit = diff.bit_length() - 1
                origin_map[c] = (h, bit)
                break

    # Group by target syndrome
    by_syndrome = defaultdict(list)  # syndrome -> list of (origin, shifted)
    for shifted, (origin, bit) in origin_map.items():
        syn = compute_syndrome(shifted, k)
        by_syndrome[syn].append((origin, shifted, bit))

    # Analyze weight distribution for each syndrome group
    print(f"\n--- Weight Distribution by Target Syndrome ---")
    for syn in sorted(by_syndrome.keys()):
        group = by_syndrome[syn]
        weights = defaultdict(int)
        for origin, shifted, bit in group:
            weights[hamming_weight(origin)] += 1

        print(f"\nSyndrome {syn} (bit {syn-1} flipped): {len(group)} codewords")
        for w in sorted(weights.keys()):
            print(f"  Weight {w}: {weights[w]}")

    # Key question: Is there a RULE for which codeword shifts to which syndrome?
    print(f"\n{'='*70}")
    print("KEY QUESTION: What determines which codeword shifts where?")
    print("="*70)

    # For each Hamming codeword, determine its fate
    codeword_fate = {}  # codeword -> 'stay' or syndrome number
    for c in stayed:
        codeword_fate[c] = 'stay'

    for shifted, (origin, bit) in origin_map.items():
        syn = compute_syndrome(shifted, k)
        codeword_fate[origin] = syn

    # Find codewords that were removed
    removed = hamming - set(codeword_fate.keys())
    for c in removed:
        codeword_fate[c] = 'removed'

    # Analyze by weight
    print(f"\n--- Fate by Codeword Weight ---")
    fate_by_weight = defaultdict(lambda: defaultdict(int))
    for c, fate in codeword_fate.items():
        w = hamming_weight(c)
        fate_by_weight[w][fate] += 1

    weights = sorted(fate_by_weight.keys())
    fates = sorted(set(f for wfates in fate_by_weight.values() for f in wfates.keys()))

    # Header
    header = f"{'Weight':<8}"
    for f in fates:
        if f == 'stay':
            header += f"{'stay':<6}"
        elif f == 'removed':
            header += f"{'rm':<6}"
        else:
            header += f"s{f:<5}"
    print(header)
    print("-" * len(header))

    for w in weights:
        row = f"{w:<8}"
        for f in fates:
            row += f"{fate_by_weight[w][f]:<6}"
        print(row)

    # Check: Is the rule simply "shift to nearest allowed position"?
    print(f"\n--- Nearest Allowed Position Analysis ---")

    for syn in [1, 2, 3]:  # Check first few syndromes
        group = by_syndrome.get(syn, [])
        if not group:
            continue

        print(f"\nSyndrome {syn} group ({len(group)} codewords):")

        # For each shifted codeword, check if this was the "best" choice
        for origin, shifted, bit in group[:3]:  # Just show first 3
            origin_bits = format(origin, f'0{n}b')
            shifted_bits = format(shifted, f'0{n}b')

            # What other options did this codeword have?
            other_options = []
            for b in range(n):
                candidate = origin ^ (1 << b)
                if candidate in allowed:
                    other_options.append((b, candidate, compute_syndrome(candidate, k)))

            print(f"  {origin_bits} → {shifted_bits} (bit {bit}, syn {syn})")
            print(f"    Other valid shifts: {[(b, s) for b, c, s in other_options if b != bit][:5]}")

    return codeword_fate


def main():
    """Main analysis."""
    for s_val in [12, 11]:
        sat_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s_val}_centers.npy"

        if Path(sat_file).exists():
            analyze_syndrome_shift_relation(sat_file, k=4, s=s_val)


if __name__ == "__main__":
    main()
