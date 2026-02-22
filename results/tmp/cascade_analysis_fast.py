#!/usr/bin/env python3
"""
Fast cascade analysis for n=31, s=28
Optimized version with limited depth exploration
"""

import numpy as np
from collections import defaultdict
import time

def generate_parity_check_matrix():
    H = np.array([[int(b) for b in format(i, '05b')] for i in range(1, 32)]).T
    return H

def compute_syndrome(v, H):
    return tuple((H @ v) % 2)

def is_hamming_codeword(v, H):
    syn = compute_syndrome(v, H)
    return all(s == 0 for s in syn)

def has_circular_ones(v, s):
    n = len(v)
    doubled = list(v) + list(v)
    for i in range(n):
        if all(doubled[i+j] == 1 for j in range(s)):
            return True
    return False

def get_nearest_codeword(v, H):
    """Get nearest Hamming codeword"""
    syn = compute_syndrome(v, H)
    if all(s == 0 for s in syn):
        return v.copy(), 0
    error_pos = int(''.join(map(str, syn)), 2) - 1
    if 0 <= error_pos < len(v):
        corrected = v.copy()
        corrected[error_pos] = 1 - corrected[error_pos]
        return corrected, 1
    return None, -1

def fast_cascade_analysis():
    """Fast analysis with limited depth"""
    print("=" * 70)
    print("FAST CASCADE ANALYSIS FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    H = generate_parity_check_matrix()
    n = 31
    s = 28

    start_time = time.time()

    # Find initial bad codewords
    print("\n1. Finding bad codewords...")

    bad_codewords = []

    # All-ones
    all_ones = np.ones(n, dtype=int)
    if is_hamming_codeword(all_ones, H):
        bad_codewords.append(all_ones)

    # Weight-28 with contiguous zeros
    for start in range(n):
        v = np.ones(n, dtype=int)
        for offset in range(3):
            v[(start + offset) % n] = 0
        if is_hamming_codeword(v, H):
            bad_codewords.append(v.copy())

    print(f"   Found {len(bad_codewords)} bad codewords")

    # Find uncovered vertices after removing bad
    print("\n2. Finding uncovered allowed vertices...")

    uncovered = []
    for bad in bad_codewords:
        for i in range(n):
            neighbor = bad.copy()
            neighbor[i] = 1 - neighbor[i]
            if not has_circular_ones(neighbor, s):
                uncovered.append(neighbor.copy())

    print(f"   Found {len(uncovered)} uncovered allowed vertices")

    # Analyze ONE uncovered vertex in detail
    print("\n3. Detailed analysis of cascade for one vertex...")

    if uncovered:
        v = uncovered[0]
        v_str = ''.join(map(str, v))
        print(f"   Vertex: {v_str}")

        # Find its nearest codeword
        nearest, dist = get_nearest_codeword(v, H)
        print(f"   Nearest codeword distance: {dist}")
        if nearest is not None:
            nearest_bad = has_circular_ones(nearest, s)
            print(f"   Nearest is bad: {nearest_bad}")

        # Find all distance-2 good codewords
        print("\n   Finding good codewords at distance 2...")
        good_d2 = []
        for i in range(n):
            for j in range(i+1, n):
                d2 = v.copy()
                d2[i] = 1 - d2[i]
                d2[j] = 1 - d2[j]
                if is_hamming_codeword(d2, H) and not has_circular_ones(d2, s):
                    good_d2.append((i, j, d2))

        print(f"   Found {len(good_d2)} good codewords at distance 2")

        if good_d2:
            print("\n   First 5 good d=2 codewords:")
            for i, j, cw in good_d2[:5]:
                cw_str = ''.join(map(str, cw))
                print(f"     Flip ({i},{j}): {cw_str}")

            # For the first conflicting good codeword, analyze ITS neighborhood
            print("\n4. Cascade step: analyzing neighborhood of first conflict...")
            first_conflict = good_d2[0][2]
            fc_str = ''.join(map(str, first_conflict))
            print(f"   Conflicting codeword: {fc_str}")

            # If we remove this good codeword, what becomes uncovered?
            new_uncovered = []
            for i in range(n):
                neighbor = first_conflict.copy()
                neighbor[i] = 1 - neighbor[i]
                if not has_circular_ones(neighbor, s):
                    # Check if covered by another codeword
                    nearest2, dist2 = get_nearest_codeword(neighbor, H)
                    if nearest2 is not None:
                        is_same = np.array_equal(nearest2, first_conflict)
                        if is_same or dist2 > 0:
                            # Not covered by another codeword at distance 0
                            new_uncovered.append(neighbor)

            print(f"   Newly uncovered by removing conflict: {len(new_uncovered)}")

    # Pattern analysis
    print("\n" + "=" * 70)
    print("CASCADE PATTERN ANALYSIS")
    print("=" * 70)

    # Count conflicts for ALL uncovered vertices
    print("\nCounting conflicts for all uncovered vertices...")

    conflict_counts = []
    for idx, v in enumerate(uncovered):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                d2 = v.copy()
                d2[i] = 1 - d2[i]
                d2[j] = 1 - d2[j]
                if is_hamming_codeword(d2, H) and not has_circular_ones(d2, s):
                    count += 1
        conflict_counts.append(count)

        if (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(uncovered)} vertices...")

    print(f"\nConflict count statistics:")
    print(f"   Min: {min(conflict_counts)}")
    print(f"   Max: {max(conflict_counts)}")
    print(f"   Avg: {sum(conflict_counts)/len(conflict_counts):.1f}")
    print(f"   Total unique conflicts: {sum(conflict_counts)} (with overlaps)")

    # Unique conflicting codewords
    print("\nFinding unique conflicting codewords...")

    unique_conflicts = set()
    for v in uncovered:
        for i in range(n):
            for j in range(i+1, n):
                d2 = v.copy()
                d2[i] = 1 - d2[i]
                d2[j] = 1 - d2[j]
                if is_hamming_codeword(d2, H) and not has_circular_ones(d2, s):
                    unique_conflicts.add(tuple(d2))

    print(f"   Unique conflicting good codewords: {len(unique_conflicts)}")

    elapsed = time.time() - start_time
    print(f"\n   Total analysis time: {elapsed:.1f}s")

    # Extrapolation
    print("\n" + "=" * 70)
    print("CASCADE EXTRAPOLATION")
    print("=" * 70)

    print(f"""
    Initial state:
    - Bad codewords: {len(bad_codewords)}
    - Uncovered vertices: {len(uncovered)}
    - Conflicting good codewords (depth 1): {len(unique_conflicts)}

    If we remove these {len(unique_conflicts)} good codewords:
    - Each has ~{n} neighbors
    - ~{n * len(unique_conflicts)} new potential uncovered vertices
    - Some will be forbidden, some covered by others

    Rough cascade estimate:
    - Depth 1: Remove {len(unique_conflicts)} good codewords
    - Depth 2: Remove ~{len(unique_conflicts) * 5} more (estimate)
    - Depth 3: Remove ~{len(unique_conflicts) * 25} more (estimate)
    - ...

    This suggests CASCADE EXPLOSION similar to n=15:
    - Eventually ~90-97% of Hamming codewords removed
    - Replaced by non-Hamming centers
    - Final code is non-linear
    """)

    return len(bad_codewords), len(uncovered), len(unique_conflicts)

if __name__ == "__main__":
    fast_cascade_analysis()
