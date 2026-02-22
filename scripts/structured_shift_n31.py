#!/usr/bin/env python3
"""
Structured Shift Construction for n=31, s=28.

Based on the pattern discovered for n=15, s=12:
- s=12 (n-3): Used bits {0, 1, 2, 11, 12, 13, 14} = {0,1,2} ∪ {n-4,...,n-1}

For n=31, s=28 (also n-3):
- Predicted bits: {0, 1, 2, 27, 28, 29, 30} = {0,1,2} ∪ {n-4,...,n-1}

This dramatically reduces the search space!

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
import json
import subprocess
from typing import Set, Dict
from collections import defaultdict
from pathlib import Path
import time


def has_circular_run(v: int, n: int, s: int) -> bool:
    bits = format(v, f'0{n}b')
    doubled = bits + bits[:-1]
    return '1' * s in doubled


def hamming_weight(v: int) -> int:
    return bin(v).count('1')


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def get_ball(v: int, n: int) -> Set[int]:
    ball = {v}
    for i in range(n):
        ball.add(v ^ (1 << i))
    return ball


def compute_syndrome(v: int, k: int) -> int:
    n = (1 << k) - 1
    syndrome = 0
    for i in range(n):
        if (v >> i) & 1:
            syndrome ^= (i + 1)
    return syndrome


def generate_hamming_code_efficient(k: int) -> Set[int]:
    """Generate Hamming code efficiently via message encoding."""
    n = (1 << k) - 1
    n_k = n - k  # Message length

    # Parity check positions are 1, 2, 4, 8, ...
    parity_positions = {(1 << i) for i in range(k)}

    # Data positions are non-powers of 2
    data_positions = [i for i in range(1, n + 1) if i not in parity_positions]

    codewords = set()

    # Enumerate all 2^(n-k) messages
    for msg in range(1 << n_k):
        codeword = 0

        # Place message bits in data positions
        for bit_idx, pos in enumerate(data_positions):
            if (msg >> bit_idx) & 1:
                codeword |= (1 << (pos - 1))

        # Compute parity bits
        for p in range(k):
            parity_pos = (1 << p)
            parity = 0
            for i in range(1, n + 1):
                if i & parity_pos:
                    if (codeword >> (i - 1)) & 1:
                        parity ^= 1
            if parity:
                codeword |= (1 << (parity_pos - 1))

        codewords.add(codeword)

    return codewords


def count_allowed_vertices(n: int, s: int) -> int:
    """Count allowed vertices without storing them all."""
    count = 0
    for v in range(1 << n):
        if not has_circular_run(v, n, s):
            count += 1
    return count


def is_allowed(v: int, n: int, s: int) -> bool:
    """Check if single vertex is allowed."""
    return not has_circular_run(v, n, s)


def analyze_n31_scale():
    """Analyze the scale of n=31 problem."""
    k = 5
    n = 31
    s = 28

    print("="*70)
    print(f"Scale Analysis for n={n}, s={s}")
    print("="*70)

    # Hamming code size
    hamming_size = 1 << (n - k)  # 2^26 = 67,108,864
    print(f"|Hamming| = 2^{n-k} = {hamming_size:,}")

    # Estimate allowed vertices (sampling)
    print("\nSampling to estimate |Allowed|...")
    sample_size = 100000
    allowed_count = 0
    np.random.seed(42)
    for _ in range(sample_size):
        v = np.random.randint(0, 1 << n)
        if not has_circular_run(v, n, s):
            allowed_count += 1

    estimated_allowed = int(allowed_count / sample_size * (1 << n))
    print(f"Estimated |Allowed| ≈ {estimated_allowed:,}")
    print(f"Expected centers ≈ {estimated_allowed // (n + 1):,}")

    # With restricted bits
    allowed_bits = [0, 1, 2, 27, 28, 29, 30]
    num_choices = len(allowed_bits) + 2  # shifts + stay + remove

    # CNF size estimation
    num_vars = hamming_size * num_choices
    print(f"\n--- CNF Size Estimation (restricted bits) ---")
    print(f"Allowed bits: {allowed_bits}")
    print(f"Choices per codeword: {num_choices}")
    print(f"Variables: ~{num_vars:,}")

    # This is too large for explicit CNF generation!
    print(f"\n*** WARNING: {hamming_size:,} codewords is too large for explicit CNF! ***")
    print("Need alternative approach for n=31.")

    return hamming_size, estimated_allowed


def greedy_syndrome_shift_construction(k: int, s: int, allowed_syndromes: list,
                                       max_iter: int = 100000, verbose: bool = True):
    """
    Greedy construction using syndrome-based shifts.

    Key insight: If lowest-syndrome-first works, we don't need SAT!

    Strategy:
    1. Start with Hamming codewords that are valid and can stay
    2. Process remaining codewords in order
    3. For each, try shifts in syndrome order (1, 2, 3, ..., then high syndromes)
    4. Pick first valid shift that doesn't conflict

    This is O(|Hamming| * |syndromes| * |centers|) which might be tractable.
    """
    n = (1 << k) - 1

    if verbose:
        print("="*70)
        print(f"Greedy Syndrome Shift Construction: n={n}, s={s}")
        print("="*70)

    # Generate Hamming code
    print("Generating Hamming code...")
    hamming = sorted(generate_hamming_code_efficient(k))
    print(f"|Hamming| = {len(hamming)}")

    # Determine valid vs bad codewords
    bad = []
    good = []
    for c in hamming:
        if is_allowed(c, n, s):
            good.append(c)
        else:
            bad.append(c)

    if verbose:
        print(f"|Good Hamming| = {len(good)}")
        print(f"|Bad Hamming| = {len(bad)}")

    # Convert allowed syndromes to bit positions
    shift_bits = [syn - 1 for syn in allowed_syndromes]
    if verbose:
        print(f"Allowed shift bits: {shift_bits}")

    # Start with all good codewords as potential stays
    centers = set()
    shift_map = {}  # codeword -> (action, result)

    # Process codewords: try stay first, then shifts in syndrome order
    print("\nProcessing codewords...")
    processed = 0
    conflicts = 0

    for c in hamming:
        processed += 1
        if processed % 500 == 0 and verbose:
            print(f"  Processed {processed}/{len(hamming)}, centers: {len(centers)}, conflicts: {conflicts}")

        # Try to stay if valid
        if is_allowed(c, n, s):
            # Check if c conflicts with existing centers
            ok = True
            for existing in centers:
                if hamming_distance(c, existing) < 3:
                    ok = False
                    break

            if ok:
                centers.add(c)
                shift_map[c] = ('stay', c)
                continue

        # Try shifts in syndrome order
        placed = False
        for bit in shift_bits:
            shifted = c ^ (1 << bit)

            if not is_allowed(shifted, n, s):
                continue

            # Check conflicts
            ok = True
            for existing in centers:
                if hamming_distance(shifted, existing) < 3:
                    ok = False
                    break

            if ok:
                centers.add(shifted)
                shift_map[c] = ('shift', shifted, bit)
                placed = True
                break

        if not placed:
            conflicts += 1
            shift_map[c] = ('removed', None)

    if verbose:
        print(f"\nFinal: {len(centers)} centers, {conflicts} removed/conflicts")

        # Verify
        print("\nVerifying...")

    # Check packing
    centers_list = list(centers)
    min_dist = float('inf')
    violations = 0
    for i in range(min(len(centers_list), 10000)):  # Sample for large sets
        for j in range(i+1, min(len(centers_list), i+1000)):
            d = hamming_distance(centers_list[i], centers_list[j])
            min_dist = min(min_dist, d)
            if d < 3:
                violations += 1

    if verbose:
        print(f"Sampled min distance: {min_dist}")
        print(f"Sampled packing violations: {violations}")

    # Check covering (sample)
    sample_size = min(10000, len(centers_list) * (n + 1))
    covered_sample = set()
    for c in centers_list[:1000]:
        for neighbor in get_ball(c, n):
            if is_allowed(neighbor, n, s):
                covered_sample.add(neighbor)

    if verbose:
        print(f"Sample covered: {len(covered_sample)}")

    return centers, shift_map


def main():
    """Main entry point."""
    # First, analyze scale
    hamming_size, estimated_allowed = analyze_n31_scale()

    print("\n" + "="*70)
    print("Attempting Greedy Syndrome Shift Construction")
    print("="*70)

    # For n=31, s=28: predict syndromes {1, 2, 3, 28, 29, 30, 31}
    k = 5
    n = 31
    s = 28
    allowed_syndromes = [1, 2, 3, 28, 29, 30, 31]

    print(f"\nTarget: n={n}, s={s}")
    print(f"Predicted allowed syndromes: {allowed_syndromes}")

    # This is still too large... let's try n=15 first to validate the greedy approach
    print("\n" + "="*70)
    print("Validating greedy on n=15, s=12 first...")
    print("="*70)

    centers_15, shift_map_15 = greedy_syndrome_shift_construction(
        k=4, s=12,
        allowed_syndromes=[1, 2, 3, 12, 13, 14, 15],
        verbose=True
    )

    # Compare with SAT solution
    sat_file = "/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_12_centers.npy"
    if Path(sat_file).exists():
        sat_centers = set(int(v) for v in np.load(sat_file))
        print(f"\nComparison with SAT solution:")
        print(f"  Greedy: {len(centers_15)} centers")
        print(f"  SAT: {len(sat_centers)} centers")
        print(f"  Overlap: {len(centers_15 & sat_centers)}")


if __name__ == "__main__":
    main()
