#!/usr/bin/env python3
"""
Non-Hamming Backbone for Perfect Partition in Λ_n(1^s)

Key Idea: Instead of using Hamming code as backbone, use non-linear perfect codes
(Vasil'ev construction) which may have different bad codeword distribution.

The obstruction in n=31 Hamming case:
- Bad codewords: 1^31, and weight-28 with zeros at {i,i+1,i+2}
- Problem: Some neighbors of bad codewords escape to allowed region

Non-linear perfect codes may:
1. Have bad codewords at different positions
2. Have neighbors that stay within forbidden region
3. Allow successful deletion strategy

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from itertools import combinations, product
from collections import defaultdict
from typing import Set, List, Tuple, Dict


def has_circular_ones(v: int, n: int, s: int) -> bool:
    """Check if integer v (as n-bit string) has s consecutive circular 1s."""
    if v == 0:
        return False
    if v == (1 << n) - 1:  # all ones
        return n >= s

    # Convert to bit positions
    bits = [(v >> i) & 1 for i in range(n)]

    # Check circular runs
    extended = bits + bits  # Double for circular check
    max_run = 0
    current_run = 0
    for b in extended[:2*n]:
        if b == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # For circular, we need to be careful not to double-count
    # Find the actual maximum circular run
    if all(b == 1 for b in bits):
        return n >= s

    # Find runs properly
    runs = []
    i = 0
    while i < n:
        if bits[i] == 1:
            run_start = i
            while i < n and bits[i] == 1:
                i += 1
            runs.append((run_start, i - run_start))
        else:
            i += 1

    if not runs:
        return False

    # Check if first and last runs connect circularly
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][0] + runs[-1][1] == n:
        circular_run = runs[0][1] + runs[-1][1]
        max_run = max(max_run, circular_run)

    max_run = max(r[1] for r in runs) if runs else 0
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][0] + runs[-1][1] == n:
        max_run = max(max_run, runs[0][1] + runs[-1][1])

    return max_run >= s


def get_allowed_vertices(n: int, s: int) -> Set[int]:
    """Get all vertices in Λ_n(1^s)."""
    allowed = set()
    for v in range(1 << n):
        if not has_circular_ones(v, n, s):
            allowed.add(v)
    return allowed


def hamming_weight(v: int) -> int:
    """Count number of 1-bits."""
    return bin(v).count('1')


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return hamming_weight(a ^ b)


def get_ball(v: int, n: int, radius: int = 1) -> Set[int]:
    """Get closed ball of radius r around v in Q_n."""
    ball = {v}
    for i in range(n):
        neighbor = v ^ (1 << i)
        ball.add(neighbor)
    return ball


def construct_hamming_code(k: int) -> Set[int]:
    """Construct Hamming code Ham(2^k - 1, 2^k - k - 1)."""
    n = (1 << k) - 1

    # Parity check matrix H: columns are 1, 2, ..., 2^k-1 in binary
    # A codeword c satisfies Hc = 0

    codewords = set()
    for v in range(1 << n):
        # Check if Hv = 0
        syndrome = 0
        for i in range(n):
            if (v >> i) & 1:
                syndrome ^= (i + 1)  # Column i+1 (1-indexed)
        if syndrome == 0:
            codewords.add(v)

    return codewords


def construct_vasilev_code(k: int, f_type: str = "xor_weight") -> Set[int]:
    """
    Construct Vasil'ev-type non-linear perfect code.

    Vasil'ev (1962) construction:
    Given Hamming code H of length n, construct code of length n+1:
    C = {(c, b) : c ∈ H, b = f(c)} ∪ {(c + e_i, 1-f(c)) : c ∈ H, e_i is unit vector}

    where f: H → {0,1} is any function.

    For our purpose, we want codes that avoid circular s consecutive 1s differently
    than Hamming codes.

    Alternative: Direct construction via switching
    """
    n = (1 << k) - 1
    hamming = construct_hamming_code(k)

    # Simple approach: Switch some codewords
    # If c and c' are at distance 3, we can potentially swap coverage

    # First, let's try a different approach:
    # Construct code by starting from Hamming and applying "switching"

    # Switching operation: If c1, c2, c3, c4 form a "switching class"
    # (specific distance pattern), we can replace some with their complements

    # For now, let's try a simpler modification:
    # Vasil'ev's original construction adds a parity bit

    if f_type == "xor_weight":
        # f(c) = (weight(c) // 2) mod 2
        vasilev = set()
        for c in hamming:
            w = hamming_weight(c)
            parity = (w // 2) % 2
            # Extend to length n+1
            new_c = (c << 1) | parity
            vasilev.add(new_c)
        return vasilev

    elif f_type == "random_switch":
        # Random switching: for each "switchable set", randomly choose configuration
        # This is more complex, implement later
        pass

    return hamming  # fallback


def find_bad_codewords(code: Set[int], n: int, s: int) -> Set[int]:
    """Find codewords that are forbidden (have s consecutive circular 1s)."""
    return {c for c in code if has_circular_ones(c, n, s)}


def analyze_bad_codeword_neighbors(code: Set[int], n: int, s: int) -> Dict:
    """
    Analyze neighbors of bad codewords.

    Key question: Do all neighbors of bad codewords stay in forbidden region?
    If yes, simple deletion works.
    If no, we have the obstruction.
    """
    bad = find_bad_codewords(code, n, s)
    forbidden = set(range(1 << n)) - get_allowed_vertices(n, s)

    analysis = {
        'bad_codewords': bad,
        'bad_count': len(bad),
        'escaping_neighbors': {},  # bad_codeword -> list of allowed neighbors
        'total_escaping': 0,
        'obstruction_details': []
    }

    for c in bad:
        ball = get_ball(c, n)
        escaping = [v for v in ball if v not in forbidden and v != c]
        if escaping:
            analysis['escaping_neighbors'][c] = escaping
            analysis['total_escaping'] += len(escaping)

            # Check distance to other (good) codewords
            for v in escaping:
                nearby_good = [g for g in code if g not in bad and hamming_distance(v, g) <= 2]
                if nearby_good:
                    analysis['obstruction_details'].append({
                        'bad_codeword': c,
                        'escaping_vertex': v,
                        'nearby_good': nearby_good,
                        'distances': [hamming_distance(v, g) for g in nearby_good]
                    })

    return analysis


def try_non_linear_modifications(k: int, s: int) -> Dict:
    """
    Try various non-linear modifications of Hamming code.

    Goal: Find a perfect code where bad codewords have no escaping neighbors,
    or escaping neighbors are not near good codewords.
    """
    n = (1 << k) - 1
    hamming = construct_hamming_code(k)

    results = {
        'n': n,
        's': s,
        'hamming_analysis': analyze_bad_codeword_neighbors(hamming, n, s),
        'modifications_tried': []
    }

    # Modification 1: Coset shift
    # Try different cosets of Hamming code
    print(f"\nAnalyzing cosets for n={n}, s={s}...")

    best_coset = None
    best_escaping = float('inf')

    # Try a sample of cosets (all would be 2^n which is huge)
    sample_size = min(1000, 1 << n)

    for shift in range(sample_size):
        coset = {c ^ shift for c in hamming}
        analysis = analyze_bad_codeword_neighbors(coset, n, s)

        if analysis['total_escaping'] < best_escaping:
            best_escaping = analysis['total_escaping']
            best_coset = shift

            if best_escaping == 0:
                print(f"  Found perfect coset at shift={shift}!")
                results['perfect_coset'] = {
                    'shift': shift,
                    'analysis': analysis
                }
                break

    results['best_coset'] = {
        'shift': best_coset,
        'escaping': best_escaping
    }

    # Modification 2: Single codeword swaps
    # For each bad codeword, try replacing it with a non-codeword at distance 3
    print(f"\n  Best coset: shift={best_coset}, escaping={best_escaping}")

    return results


def analyze_n31_obstruction():
    """
    Detailed analysis of n=31, s=28 obstruction.

    Questions:
    1. How many bad codewords in Hamming(31)?
    2. How many neighbors escape to allowed region?
    3. What is the structure of escaping neighbors?
    """
    print("="*70)
    print("Analyzing n=31, s=28 Obstruction")
    print("="*70)

    k = 5
    n = 31
    s = 28

    # This is too large to enumerate all codewords
    # Let's analyze theoretically

    print(f"\nHamming(31) has 2^26 = {1 << 26:,} codewords")
    print(f"We need to find codewords with circular run ≥ 28")

    # Bad codeword structure:
    # - All ones: 1^31 (weight 31)
    # - Weight 30: one zero → 30 consecutive 1s ≥ 28 ✓
    # - Weight 29: two zeros → if zeros are adjacent or 1 apart, might have 28+ run
    # - Weight 28: three zeros → if zeros are in 3 consecutive positions

    print("\nBad codeword types:")
    print("  - 1^31 (all ones): 1 codeword")
    print("  - Weight 30 (one zero): Need to check which are in Hamming(31)")
    print("  - Weight 29 (two zeros): Need to check pattern")
    print("  - Weight 28 (three zeros): Only if zeros in arc of length 3")

    # For Hamming code, we can characterize bad codewords:
    # 1^n is always in Hamming (syndrome = 1+2+...+(2^k-1) = 0 for all k≥2)

    # Let's count bad codewords for small cases first
    for test_k in [3, 4]:
        test_n = (1 << test_k) - 1
        test_s = test_n - 3
        hamming = construct_hamming_code(test_k)
        bad = find_bad_codewords(hamming, test_n, test_s)
        print(f"\nk={test_k}, n={test_n}, s={test_s}:")
        print(f"  |Hamming| = {len(hamming)}")
        print(f"  |Bad| = {len(bad)}")
        for c in bad:
            print(f"    {bin(c)[2:].zfill(test_n)} (weight {hamming_weight(c)})")

        analysis = analyze_bad_codeword_neighbors(hamming, test_n, test_s)
        print(f"  Total escaping neighbors: {analysis['total_escaping']}")
        if analysis['obstruction_details']:
            print(f"  Obstruction found!")
            for detail in analysis['obstruction_details'][:3]:
                print(f"    Bad: {bin(detail['bad_codeword'])[2:].zfill(test_n)}")
                print(f"    Escaping: {bin(detail['escaping_vertex'])[2:].zfill(test_n)}")
                print(f"    Near good at distances: {detail['distances']}")


def explore_non_hamming_for_n15():
    """
    For n=15, explore non-Hamming perfect codes.

    This is tractable and can reveal structure.
    """
    print("="*70)
    print("Exploring Non-Hamming Backbones for n=15, s=12")
    print("="*70)

    k = 4
    n = 15
    s = 12

    hamming = construct_hamming_code(k)
    allowed = get_allowed_vertices(n, s)
    forbidden = set(range(1 << n)) - allowed

    print(f"\n|V(Λ_15(1^12))| = {len(allowed)}")
    print(f"|Forbidden| = {len(forbidden)}")
    print(f"|Hamming(15)| = {len(hamming)}")

    # Analyze Hamming
    hamming_analysis = analyze_bad_codeword_neighbors(hamming, n, s)
    print(f"\nHamming analysis:")
    print(f"  Bad codewords: {hamming_analysis['bad_count']}")
    print(f"  Escaping neighbors: {hamming_analysis['total_escaping']}")

    # Try to find non-Hamming perfect codes
    # Method: Start from Hamming, apply "switching"

    # A switching set in a perfect code consists of 4 codewords forming a specific pattern
    # We can swap 2 codewords with 2 non-codewords while maintaining perfect property

    # For now, let's explore coset shifts
    print("\nExploring coset shifts...")
    results = try_non_linear_modifications(k, s)

    return results


def explore_switching_for_n15():
    """
    Explore switching operations on Hamming(15).

    Switching: Given 4 codewords c1, c2, c3, c4 with specific distance pattern,
    we can replace them with 4 different vectors to get another perfect code.
    """
    print("="*70)
    print("Exploring Switching Operations for n=15")
    print("="*70)

    k = 4
    n = 15
    s = 12

    hamming = construct_hamming_code(k)

    # Find switching sets
    # A switching set: c1, c2, c3, c4 where:
    # d(c1,c2) = d(c3,c4) = 2 (not in code, since min distance is 3)
    # Actually for perfect codes, switching is more complex

    # Simpler approach: Direct modification
    # Find bad codeword and its "replacement candidates"

    bad = find_bad_codewords(hamming, n, s)
    allowed = get_allowed_vertices(n, s)

    print(f"\nBad codewords in Hamming(15): {len(bad)}")
    for c in bad:
        print(f"  {bin(c)[2:].zfill(n)} (weight {hamming_weight(c)})")

    # For each bad codeword, find potential replacements
    # Replacement must:
    # 1. Be in allowed region
    # 2. Be at distance ≥ 3 from all other (good) codewords
    # 3. Cover the same vertices as bad codeword OR find alternate coverage

    good = hamming - bad

    print(f"\nSearching for replacement candidates...")

    for c in bad:
        print(f"\nBad codeword: {bin(c)[2:].zfill(n)}")

        # Vertices that would be uncovered if we remove c
        uncovered_by_c = get_ball(c, n) & allowed
        print(f"  Would uncover {len(uncovered_by_c)} allowed vertices")

        # Find candidates at distance ≥ 3 from all good codewords
        candidates = []
        for v in allowed:
            if v in hamming:
                continue
            min_dist_to_good = min(hamming_distance(v, g) for g in good)
            if min_dist_to_good >= 3:
                covers = get_ball(v, n) & uncovered_by_c
                if covers:
                    candidates.append((v, min_dist_to_good, len(covers)))

        print(f"  Found {len(candidates)} candidates at distance ≥ 3 from good codewords")

        # Sort by coverage
        candidates.sort(key=lambda x: -x[2])
        for v, d, cov in candidates[:5]:
            print(f"    {bin(v)[2:].zfill(n)}: dist={d}, covers={cov}")


def main():
    """Main analysis."""
    print("Non-Hamming Backbone Analysis for Perfect Partitions")
    print("="*70)

    # First, analyze the known obstruction
    analyze_n31_obstruction()

    print("\n" + "="*70)

    # Then explore for n=15
    explore_non_hamming_for_n15()

    print("\n" + "="*70)

    # Try switching
    explore_switching_for_n15()


if __name__ == "__main__":
    main()
