#!/usr/bin/env python3
"""
Compare SAT Solution vs Hamming Code Structure

Key question: How does the SAT solution differ from Hamming?
- Syndrome distribution
- Weight distribution
- Linearity check
- Overlap percentage

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def has_circular_run(v: int, n: int, s: int) -> bool:
    bits = format(v, f'0{n}b')
    doubled = bits + bits[:-1]
    return '1' * s in doubled


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


def get_allowed_vertices(n: int, s: int) -> set:
    return {v for v in range(1 << n) if not has_circular_run(v, n, s)}


def check_linearity(code: set, sample_size: int = 1000) -> dict:
    """
    Check if code is linear (closed under XOR).
    Returns fraction of pairs whose XOR is in code.
    """
    code_list = list(code)
    n_pairs = min(sample_size, len(code_list) * (len(code_list) - 1) // 2)

    import random
    random.seed(42)

    in_code = 0
    checked = 0

    for _ in range(n_pairs):
        i, j = random.sample(range(len(code_list)), 2)
        xor_val = code_list[i] ^ code_list[j]
        if xor_val in code:
            in_code += 1
        checked += 1

    return {
        'pairs_checked': checked,
        'xor_in_code': in_code,
        'linearity_ratio': in_code / checked if checked > 0 else 0,
        'is_linear': in_code == checked
    }


def analyze_sat_solution(centers_file: str, n: int, s: int, k: int):
    """
    Analyze SAT solution structure.
    """
    print(f"="*70)
    print(f"Analyzing SAT Solution: {centers_file}")
    print(f"n={n}, s={s}, k={k}")
    print(f"="*70)

    # Load SAT centers
    centers_arr = np.load(centers_file)

    # Convert to set of integers (already stored as integers)
    sat_centers = set(int(v) for v in centers_arr)

    print(f"\n|SAT centers| = {len(sat_centers)}")

    # Generate Hamming
    hamming = generate_hamming_code(k)
    print(f"|Hamming| = {len(hamming)}")

    # Compare
    overlap = sat_centers & hamming
    only_sat = sat_centers - hamming
    only_hamming = hamming - sat_centers

    print(f"\nOverlap: {len(overlap)} ({100*len(overlap)/len(sat_centers):.1f}% of SAT)")
    print(f"Only in SAT: {len(only_sat)}")
    print(f"Only in Hamming: {len(only_hamming)}")

    # Syndrome distribution
    print(f"\n--- Syndrome Distribution ---")

    sat_syn = defaultdict(int)
    for c in sat_centers:
        sat_syn[compute_syndrome(c, k)] += 1

    hamming_syn = defaultdict(int)
    for c in hamming:
        hamming_syn[compute_syndrome(c, k)] += 1

    print(f"{'Syndrome':<10} {'SAT':<10} {'Hamming':<10}")
    print("-" * 30)
    all_syns = sorted(set(sat_syn.keys()) | set(hamming_syn.keys()))
    for syn in all_syns[:20]:  # Show top 20
        print(f"{syn:<10} {sat_syn[syn]:<10} {hamming_syn[syn]:<10}")

    # Weight distribution
    print(f"\n--- Weight Distribution ---")

    sat_wt = defaultdict(int)
    for c in sat_centers:
        sat_wt[hamming_weight(c)] += 1

    hamming_wt = defaultdict(int)
    for c in hamming:
        hamming_wt[hamming_weight(c)] += 1

    print(f"{'Weight':<10} {'SAT':<10} {'Hamming':<10}")
    print("-" * 30)
    for w in sorted(set(sat_wt.keys()) | set(hamming_wt.keys())):
        print(f"{w:<10} {sat_wt[w]:<10} {hamming_wt[w]:<10}")

    # Linearity check
    print(f"\n--- Linearity Check ---")
    lin_result = check_linearity(sat_centers)
    print(f"Pairs checked: {lin_result['pairs_checked']}")
    print(f"XOR in code: {lin_result['xor_in_code']}")
    print(f"Linearity ratio: {lin_result['linearity_ratio']:.4f}")
    print(f"Is linear: {lin_result['is_linear']}")

    # Check if SAT centers form a valid perfect partition
    print(f"\n--- Validity Check ---")

    allowed = get_allowed_vertices(n, s)

    # All centers in allowed?
    invalid = sat_centers - allowed
    print(f"Invalid centers (not in allowed): {len(invalid)}")

    # Packing (sample)
    packing_ok = True
    sample_centers = list(sat_centers)[:100]
    for c1, c2 in combinations(sample_centers, 2):
        if hamming_distance(c1, c2) < 3:
            packing_ok = False
            break
    print(f"Packing (sampled): {'OK' if packing_ok else 'FAILED'}")

    # Coverage
    covered = set()
    for c in sat_centers:
        for i in range(n):
            covered.add(c ^ (1 << i))
        covered.add(c)

    covered_allowed = covered & allowed
    print(f"Covered allowed: {len(covered_allowed)} / {len(allowed)}")

    return {
        'sat_centers': sat_centers,
        'hamming': hamming,
        'overlap': overlap,
        'sat_syndrome': dict(sat_syn),
        'hamming_syndrome': dict(hamming_syn),
        'linearity': lin_result
    }


def analyze_non_hamming_centers(sat_centers: set, hamming: set, k: int, n: int, s: int):
    """
    Deeper analysis of centers that are NOT in Hamming.
    """
    print(f"\n{'='*70}")
    print("Analysis of Non-Hamming Centers")
    print("="*70)

    non_hamming = sat_centers - hamming

    print(f"\n|Non-Hamming centers| = {len(non_hamming)}")

    # What syndromes do they have?
    syn_dist = defaultdict(list)
    for c in non_hamming:
        syn = compute_syndrome(c, k)
        syn_dist[syn].append(c)

    print(f"\nSyndrome distribution of non-Hamming centers:")
    for syn in sorted(syn_dist.keys()):
        print(f"  Syndrome {syn}: {len(syn_dist[syn])} centers")

    # For each syndrome, what's the nearest Hamming codeword?
    print(f"\n--- Distance to Hamming codewords ---")

    for syn in sorted(syn_dist.keys())[:5]:  # Sample
        centers_with_syn = syn_dist[syn][:5]
        for c in centers_with_syn:
            # Find nearest Hamming codeword
            min_dist = n + 1
            nearest = None
            for h in hamming:
                d = hamming_distance(c, h)
                if d < min_dist:
                    min_dist = d
                    nearest = h

            print(f"\n  Non-Hamming: {format(c, f'0{n}b')} (syn={syn})")
            print(f"    Nearest Hamming: {format(nearest, f'0{n}b')} at distance {min_dist}")

    # Key question: Are non-Hamming centers at distance 1 from Hamming? (error patterns)
    dist_to_hamming = defaultdict(int)
    for c in non_hamming:
        min_dist = min(hamming_distance(c, h) for h in hamming)
        dist_to_hamming[min_dist] += 1

    print(f"\nDistance from non-Hamming to nearest Hamming:")
    for d in sorted(dist_to_hamming.keys()):
        print(f"  Distance {d}: {dist_to_hamming[d]} centers")


def main():
    """Main analysis."""
    # n=15, s=12
    centers_file = "/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_12_centers.npy"

    if Path(centers_file).exists():
        result = analyze_sat_solution(centers_file, n=15, s=12, k=4)
        analyze_non_hamming_centers(result['sat_centers'], result['hamming'], k=4, n=15, s=12)
    else:
        print(f"File not found: {centers_file}")

    # n=15, s=11
    print("\n\n")
    centers_file_11 = "/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_11_centers.npy"

    if Path(centers_file_11).exists():
        result = analyze_sat_solution(centers_file_11, n=15, s=11, k=4)
        analyze_non_hamming_centers(result['sat_centers'], result['hamming'], k=4, n=15, s=11)
    else:
        print(f"File not found: {centers_file_11}")


if __name__ == "__main__":
    main()
