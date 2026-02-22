#!/usr/bin/env python3
"""
Improved Greedy Syndrome Shift Construction.

Key insight: Simple greedy achieves 2045/2047 on n=15.
Missing 2 centers likely due to ordering.

Improvements:
1. Order codewords by "difficulty" (fewer valid shift options first)
2. Use backtracking for conflicts
3. Try different orderings

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, List, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import random


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


def is_allowed(v: int, n: int, s: int) -> bool:
    return not has_circular_run(v, n, s)


def get_valid_positions(c: int, n: int, s: int, shift_bits: List[int]) -> List[int]:
    """Get all valid positions for codeword c (stay or shifts)."""
    positions = []

    # Can stay?
    if is_allowed(c, n, s):
        positions.append(c)

    # Can shift?
    for bit in shift_bits:
        shifted = c ^ (1 << bit)
        if is_allowed(shifted, n, s):
            positions.append(shifted)

    return positions


def count_valid_options(c: int, n: int, s: int, shift_bits: List[int]) -> int:
    """Count valid placement options for codeword c."""
    return len(get_valid_positions(c, n, s, shift_bits))


def improved_greedy(k: int, s: int, shift_bits: List[int],
                    ordering: str = 'difficulty', seed: int = 42) -> Tuple[Set[int], Dict]:
    """
    Improved greedy with different orderings.

    orderings:
    - 'difficulty': Fewer options first (most constrained first)
    - 'easy': More options first
    - 'random': Random order
    - 'weight': By weight
    """
    n = (1 << k) - 1
    random.seed(seed)

    hamming = list(generate_hamming_code(k))

    # Compute difficulty for each codeword
    difficulties = {}
    for c in hamming:
        difficulties[c] = count_valid_options(c, n, s, shift_bits)

    # Sort by ordering
    if ordering == 'difficulty':
        hamming.sort(key=lambda c: (difficulties[c], c))
    elif ordering == 'easy':
        hamming.sort(key=lambda c: (-difficulties[c], c))
    elif ordering == 'random':
        random.shuffle(hamming)
    elif ordering == 'weight':
        hamming.sort(key=lambda c: (hamming_weight(c), c))

    centers = set()
    placement = {}

    for c in hamming:
        positions = get_valid_positions(c, n, s, shift_bits)

        placed = False
        for pos in positions:
            # Check conflict
            ok = True
            for existing in centers:
                if hamming_distance(pos, existing) < 3:
                    ok = False
                    break

            if ok:
                centers.add(pos)
                placement[c] = pos
                placed = True
                break

        if not placed:
            placement[c] = None  # Removed

    return centers, placement


def verify_coverage(centers: Set[int], n: int, s: int) -> Tuple[int, int, Set[int]]:
    """Verify covering and return statistics."""
    covered = set()
    for c in centers:
        for v in get_ball(c, n):
            if is_allowed(v, n, s):
                covered.add(v)

    # Count total allowed
    total_allowed = sum(1 for v in range(1 << n) if is_allowed(v, n, s))

    uncovered = set(v for v in range(1 << n) if is_allowed(v, n, s) and v not in covered)

    return len(covered), total_allowed, uncovered


def local_search(centers: Set[int], placement: Dict, k: int, s: int,
                 shift_bits: List[int], max_iter: int = 10000) -> Tuple[Set[int], Dict]:
    """
    Local search to improve greedy solution.

    Try to:
    1. Add centers for uncovered vertices
    2. Swap placements to reduce conflicts
    """
    n = (1 << k) - 1
    hamming = set(placement.keys())

    # Find unplaced codewords
    unplaced = [c for c, pos in placement.items() if pos is None]
    print(f"Starting local search with {len(unplaced)} unplaced codewords")

    best_centers = centers.copy()
    best_placement = placement.copy()
    best_unplaced = len(unplaced)

    for iteration in range(max_iter):
        improved = False

        for c in unplaced:
            # Try each valid position
            for pos in get_valid_positions(c, n, s, shift_bits):
                # Find conflicting centers
                conflicts = [e for e in centers if hamming_distance(pos, e) < 3]

                if not conflicts:
                    # Can add directly
                    centers.add(pos)
                    placement[c] = pos
                    unplaced.remove(c)
                    improved = True
                    break

                # Try to move each conflicting center
                for conflict in conflicts:
                    # Find which codeword is at conflict position
                    origin = None
                    for code, p in placement.items():
                        if p == conflict:
                            origin = code
                            break

                    if origin is None:
                        continue

                    # Can we move origin somewhere else?
                    for alt_pos in get_valid_positions(origin, n, s, shift_bits):
                        if alt_pos == conflict:
                            continue

                        # Check if alt_pos is valid
                        ok = True
                        for e in centers:
                            if e == conflict:
                                continue
                            if hamming_distance(alt_pos, e) < 3:
                                ok = False
                                break

                        # Also check against pos
                        if ok and hamming_distance(alt_pos, pos) < 3:
                            ok = False

                        if ok:
                            # Swap!
                            centers.remove(conflict)
                            centers.add(alt_pos)
                            centers.add(pos)
                            placement[origin] = alt_pos
                            placement[c] = pos
                            unplaced.remove(c)
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        current_unplaced = sum(1 for p in placement.values() if p is None)
        if current_unplaced < best_unplaced:
            best_centers = centers.copy()
            best_placement = placement.copy()
            best_unplaced = current_unplaced
            print(f"  Iteration {iteration}: improved to {current_unplaced} unplaced")

        if current_unplaced == 0:
            print(f"  Found complete solution at iteration {iteration}!")
            break

        if not improved:
            break

    return best_centers, best_placement


def main():
    """Test improved greedy."""
    k = 4
    s = 12
    n = 15
    shift_bits = [0, 1, 2, 11, 12, 13, 14]

    print("="*70)
    print(f"Improved Greedy Syndrome Shift Construction: n={n}, s={s}")
    print("="*70)

    # Try different orderings
    orderings = ['difficulty', 'easy', 'random', 'weight']
    best_result = None
    best_count = 0

    for ordering in orderings:
        print(f"\n--- Ordering: {ordering} ---")
        centers, placement = improved_greedy(k, s, shift_bits, ordering=ordering)

        unplaced = sum(1 for p in placement.values() if p is None)
        print(f"Centers: {len(centers)}, Unplaced: {unplaced}")

        if len(centers) > best_count:
            best_count = len(centers)
            best_result = (centers, placement, ordering)

    # Apply local search to best result
    print("\n" + "="*70)
    print("Applying Local Search")
    print("="*70)

    centers, placement, ordering = best_result
    print(f"Starting from best greedy ({ordering}): {len(centers)} centers")

    centers, placement = local_search(centers, placement, k, s, shift_bits)

    # Verify
    print("\n--- Final Verification ---")
    print(f"Centers: {len(centers)}")
    unplaced = sum(1 for p in placement.values() if p is None)
    print(f"Unplaced: {unplaced}")

    # Check packing
    centers_list = list(centers)
    min_dist = float('inf')
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i+1:]:
            d = hamming_distance(c1, c2)
            min_dist = min(min_dist, d)

    print(f"Min distance: {min_dist}")

    # Check covering
    covered, total, uncovered = verify_coverage(centers, n, s)
    print(f"Covered: {covered}/{total}")
    print(f"Uncovered: {len(uncovered)}")

    if len(uncovered) == 0 and min_dist >= 3:
        print("\n*** SUCCESS! Valid perfect partition! ***")
        np.save('improved_greedy_n15_s12_centers.npy', np.array(list(centers)))
        print("Saved to improved_greedy_n15_s12_centers.npy")
    else:
        print("\n*** NOT COMPLETE ***")
        if uncovered:
            print(f"First few uncovered: {list(uncovered)[:5]}")


if __name__ == "__main__":
    main()
