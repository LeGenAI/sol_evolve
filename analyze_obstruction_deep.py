#!/usr/bin/env python3
"""
Deep Analysis of Obstruction in Λ_n(1^s) Perfect Partition

Key finding: Greedy repair fails because uncovered vertices are too close to good codewords.

Questions to answer:
1. What is the exact structure of the obstruction?
2. Can we find a GLOBAL modification that works?
3. What makes SAT solutions different from Hamming-based?

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, List, Tuple, Dict
from collections import defaultdict
from itertools import combinations


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


def generate_hamming_code(k: int) -> Set[int]:
    n = (1 << k) - 1
    return {v for v in range(1 << n) if compute_syndrome(v, k) == 0}


def get_allowed_vertices(n: int, s: int) -> Set[int]:
    return {v for v in range(1 << n) if not has_circular_run(v, n, s)}


def analyze_obstruction_structure(k: int, s: int):
    """
    Detailed analysis of why local repair fails.
    """
    n = (1 << k) - 1

    print(f"="*70)
    print(f"Obstruction Analysis: n={n}, s={s}")
    print(f"="*70)

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)
    forbidden = set(range(1 << n)) - allowed

    bad = {c for c in hamming if c not in allowed}
    good = hamming - bad

    print(f"\n|Hamming| = {len(hamming)}")
    print(f"|Allowed| = {len(allowed)}")
    print(f"|Bad| = {len(bad)}")
    print(f"|Good| = {len(good)}")

    # Find uncovered vertices after removing bad
    covered_by_good = set()
    for c in good:
        covered_by_good.update(get_ball(c, n) & allowed)

    uncovered = allowed - covered_by_good

    print(f"\nUncovered after removing bad: {len(uncovered)}")

    # For each uncovered vertex, analyze its neighborhood
    print(f"\n--- Uncovered Vertex Analysis ---")

    distance_to_good = defaultdict(list)  # distance -> list of (uncovered, good)

    for v in uncovered:
        # Find nearest good codeword
        nearest_good = []
        for g in good:
            d = hamming_distance(v, g)
            if d <= 3:
                nearest_good.append((g, d))

        nearest_good.sort(key=lambda x: x[1])

        # Which bad codeword was covering this?
        covering_bad = None
        for b in bad:
            if v in get_ball(b, n):
                covering_bad = b
                break

        if len(uncovered) <= 20:  # Only print for small cases
            print(f"\n  Uncovered: {format(v, f'0{n}b')} (weight={hamming_weight(v)})")
            print(f"    Was covered by bad: {format(covering_bad, f'0{n}b') if covering_bad else 'N/A'}")
            print(f"    Nearest good codewords:")
            for g, d in nearest_good[:3]:
                print(f"      {format(g, f'0{n}b')} at distance {d}")

        # Record minimum distance
        if nearest_good:
            min_d = nearest_good[0][1]
            distance_to_good[min_d].append(v)

    print(f"\n--- Distance Distribution ---")
    for d in sorted(distance_to_good.keys()):
        print(f"  Distance {d} to nearest good: {len(distance_to_good[d])} uncovered vertices")

    # The obstruction: if uncovered vertex v has nearest good at distance 2,
    # then ANY center covering v must be at distance ≤ 1 from v,
    # hence at distance ≤ 3 from that good codeword.
    # But we need distance ≥ 3 from ALL good codewords.

    # So the question is: can we find a center c such that
    # d(c, v) ≤ 1 AND d(c, g) ≥ 3 for ALL good g?

    print(f"\n--- Repair Feasibility ---")

    for v in list(uncovered)[:5]:  # Sample
        # Check all potential covering centers
        potential_centers = get_ball(v, n) & allowed

        feasible = []
        infeasible = []

        for c in potential_centers:
            # Check distance to all good
            too_close_to_good = [g for g in good if hamming_distance(c, g) < 3]
            if too_close_to_good:
                infeasible.append((c, too_close_to_good[:3]))  # Keep first 3
            else:
                feasible.append(c)

        print(f"\n  Uncovered: {format(v, f'0{n}b')}")
        print(f"    Potential centers (in ball ∩ allowed): {len(potential_centers)}")
        print(f"    Feasible (d≥3 from all good): {len(feasible)}")
        print(f"    Infeasible: {len(infeasible)}")

        if not feasible and infeasible:
            c, near = infeasible[0]
            print(f"    Example infeasible: {format(c, f'0{n}b')}")
            for g in near:
                d = hamming_distance(c, g)
                print(f"      Too close to {format(g, f'0{n}b')} (d={d})")

    return uncovered, distance_to_good


def find_global_modification(k: int, s: int, max_changes: int = 10):
    """
    Try to find a global modification: swap some good codewords for non-codewords.

    Idea: Instead of just removing bad and adding repair centers,
    we might need to also REMOVE some good codewords to make room for repair.
    """
    n = (1 << k) - 1

    print(f"\n{'='*70}")
    print(f"Global Modification Search: n={n}, s={s}")
    print(f"{'='*70}")

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    bad = {c for c in hamming if c not in allowed}
    good = hamming - bad

    # Find uncovered
    covered_by_good = set()
    for c in good:
        covered_by_good.update(get_ball(c, n) & allowed)
    uncovered = allowed - covered_by_good

    print(f"\nInitial: {len(bad)} bad, {len(uncovered)} uncovered")

    # Strategy: For each uncovered v, find the good codeword(s) that block repair.
    # If we remove those blockers, can we find replacement centers?

    # Find blocking relationships
    blocking = defaultdict(set)  # uncovered -> set of good codewords blocking repair

    for v in uncovered:
        potential_centers = get_ball(v, n) & allowed
        for c in potential_centers:
            for g in good:
                if hamming_distance(c, g) < 3:
                    blocking[v].add(g)

    print(f"\nBlocking analysis:")
    # Which good codewords appear most often as blockers?
    blocker_count = defaultdict(int)
    for v, blockers in blocking.items():
        for g in blockers:
            blocker_count[g] += 1

    top_blockers = sorted(blocker_count.items(), key=lambda x: -x[1])[:10]
    print(f"  Top blockers:")
    for g, count in top_blockers:
        print(f"    {format(g, f'0{n}b')} blocks {count} uncovered vertices")

    # Try removing top blockers and see if repair becomes possible
    print(f"\n--- Testing removal of top blockers ---")

    for num_remove in range(1, min(max_changes + 1, len(top_blockers) + 1)):
        # Remove top num_remove blockers
        to_remove = {g for g, _ in top_blockers[:num_remove]}
        modified_good = good - to_remove

        # Recompute uncovered
        covered = set()
        for c in modified_good:
            covered.update(get_ball(c, n) & allowed)
        new_uncovered = allowed - covered

        print(f"\n  Removing {num_remove} blockers:")
        print(f"    Good remaining: {len(modified_good)}")
        print(f"    Uncovered: {len(new_uncovered)}")

        # Try greedy repair
        repair_centers = set()
        current_centers = modified_good.copy()
        still_uncovered = new_uncovered.copy()

        while still_uncovered:
            # Find best repair center
            best_center = None
            best_coverage = 0

            for c in allowed:
                if c in current_centers:
                    continue
                # Check distance
                if any(hamming_distance(c, g) < 3 for g in current_centers):
                    continue
                # Count coverage
                cov = len(get_ball(c, n) & still_uncovered)
                if cov > best_coverage:
                    best_coverage = cov
                    best_center = c

            if best_center is None:
                break

            repair_centers.add(best_center)
            current_centers.add(best_center)
            still_uncovered -= get_ball(best_center, n)

        if not still_uncovered:
            print(f"    *** SUCCESS! Repair found with {len(repair_centers)} new centers ***")
            print(f"    Total centers: {len(current_centers)}")
            print(f"    Non-Hamming: {len(current_centers - hamming)}")

            # Verify
            verify_covered = set()
            for c in current_centers:
                verify_covered.update(get_ball(c, n) & allowed)

            packing_ok = all(
                hamming_distance(c1, c2) >= 3
                for c1, c2 in combinations(current_centers, 2)
            )

            print(f"    Covers all allowed: {verify_covered == allowed}")
            print(f"    Packing OK: {packing_ok}")

            if verify_covered == allowed and packing_ok:
                return current_centers

        else:
            print(f"    Repair failed: {len(still_uncovered)} still uncovered")

    return None


def main():
    """Main analysis."""
    # n=7
    print("\n" + "="*70)
    print("N=7, S=4 ANALYSIS")
    print("="*70)
    analyze_obstruction_structure(3, 4)
    result7 = find_global_modification(3, 4, max_changes=5)

    # n=15
    print("\n" + "="*70)
    print("N=15, S=12 ANALYSIS")
    print("="*70)
    analyze_obstruction_structure(4, 12)
    result15 = find_global_modification(4, 12, max_changes=10)


if __name__ == "__main__":
    main()
