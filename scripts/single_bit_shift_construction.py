#!/usr/bin/env python3
"""
Single-Bit Shift Construction for Perfect Partition

KEY INSIGHT from SAT analysis:
- SAT solutions have IDENTICAL weight distribution as Hamming
- ALL non-Hamming centers are at distance 1 from Hamming codewords
- This means: SAT replaces c ∈ Hamming with c ⊕ e_i for some bit position i

Construction Strategy:
1. Start with Hamming code H
2. For each "problematic" codeword c (bad or blocking repair):
   - Find bit position i such that c ⊕ e_i is:
     a) In allowed region (avoids s consecutive 1s)
     b) Maintains d ≥ 3 from other centers
     c) Covers all vertices that c would cover (within allowed)
3. This is a "switching" operation that preserves perfect code properties

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, List, Tuple, Dict, Optional
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


def find_valid_shift(c: int, n: int, s: int, current_centers: Set[int],
                     allowed: Set[int]) -> Optional[int]:
    """
    Find a bit position i such that c ⊕ (1 << i) is:
    1. In allowed region
    2. Has distance ≥ 3 from all other centers (except c which will be removed)
    3. Covers similar vertices as c

    Returns bit position or None if no valid shift exists.
    """
    other_centers = current_centers - {c}

    for i in range(n):
        shifted = c ^ (1 << i)

        # Check: in allowed region
        if shifted not in allowed:
            continue

        # Check: distance ≥ 3 from all other centers
        min_dist = min((hamming_distance(shifted, o) for o in other_centers), default=n+1)
        if min_dist < 3:
            continue

        # Valid shift found!
        return i

    return None


def single_bit_shift_construction(k: int, s: int, verbose: bool = True) -> Optional[Set[int]]:
    """
    Construct perfect partition using single-bit shifts from Hamming code.

    Strategy:
    1. Identify bad codewords (those with s consecutive 1s)
    2. For each bad codeword, find valid single-bit shift
    3. If direct shift doesn't work, try cascading shifts
    """
    n = (1 << k) - 1

    if verbose:
        print(f"="*70)
        print(f"Single-Bit Shift Construction: n={n}, s={s}")
        print(f"="*70)

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    if verbose:
        print(f"\n|Hamming| = {len(hamming)}")
        print(f"|Allowed| = {len(allowed)}")

    # Find bad codewords
    bad = {c for c in hamming if c not in allowed}
    good = hamming - bad

    if verbose:
        print(f"|Bad codewords| = {len(bad)}")
        print(f"|Good codewords| = {len(good)}")

    if not bad:
        if verbose:
            print("\nNo bad codewords! Hamming code is already valid.")
        return hamming

    # Build initial center set (all good codewords)
    centers = good.copy()

    # Track shifts applied
    shifts_applied = []

    # Process bad codewords
    for c in bad:
        if verbose:
            print(f"\nProcessing bad codeword: {format(c, f'0{n}b')} (weight={hamming_weight(c)})")

        # Find valid shift
        shift_pos = find_valid_shift(c, n, s, centers | {c}, allowed)

        if shift_pos is not None:
            shifted = c ^ (1 << shift_pos)
            if verbose:
                print(f"  Shift by bit {shift_pos}: {format(shifted, f'0{n}b')}")

            # Apply shift
            centers.add(shifted)
            shifts_applied.append((c, shift_pos, shifted))
        else:
            if verbose:
                print(f"  No valid direct shift found!")

            # Try to find cascade: shift a good codeword to make room
            found_cascade = False

            for g in list(centers)[:100]:  # Sample good codewords
                # Check if g is blocking potential shifts of c
                for i in range(n):
                    potential = c ^ (1 << i)
                    if potential not in allowed:
                        continue

                    d_to_g = hamming_distance(potential, g)
                    if d_to_g >= 3:
                        continue  # g is not blocking

                    # g is blocking! Can we shift g?
                    for j in range(n):
                        g_shifted = g ^ (1 << j)
                        if g_shifted not in allowed:
                            continue

                        # Check if g_shifted maintains packing with other centers
                        other = centers - {g}
                        min_dist = min((hamming_distance(g_shifted, o) for o in other), default=n+1)
                        if min_dist < 3:
                            continue

                        # Also check distance from potential (the shifted c)
                        if hamming_distance(g_shifted, potential) < 3:
                            continue

                        # Found cascade!
                        if verbose:
                            print(f"  Cascade: shift {format(g, f'0{n}b')} by bit {j}")
                            print(f"           then shift {format(c, f'0{n}b')} by bit {i}")

                        # Apply cascade
                        centers.remove(g)
                        centers.add(g_shifted)
                        centers.add(potential)
                        shifts_applied.append((g, j, g_shifted))
                        shifts_applied.append((c, i, potential))
                        found_cascade = True
                        break

                    if found_cascade:
                        break
                if found_cascade:
                    break

            if not found_cascade:
                if verbose:
                    print(f"  FAILED: Cannot shift this codeword!")
                return None

    # Verify result
    if verbose:
        print(f"\n--- Verification ---")
        print(f"|Centers| = {len(centers)}")

    # Check all centers in allowed
    invalid = centers - allowed
    if verbose:
        print(f"Invalid centers: {len(invalid)}")

    # Check packing
    packing_violations = 0
    for c1, c2 in combinations(centers, 2):
        if hamming_distance(c1, c2) < 3:
            packing_violations += 1
    if verbose:
        print(f"Packing violations: {packing_violations}")

    # Check covering
    covered = set()
    for c in centers:
        covered.update(get_ball(c, n) & allowed)
    uncovered = allowed - covered

    if verbose:
        print(f"Covered: {len(covered)} / {len(allowed)}")
        print(f"Uncovered: {len(uncovered)}")

    if len(invalid) == 0 and packing_violations == 0 and len(uncovered) == 0:
        if verbose:
            print(f"\n*** SUCCESS! Perfect partition constructed! ***")
            print(f"Total shifts: {len(shifts_applied)}")

            # Analyze overlap with Hamming
            overlap = centers & hamming
            print(f"Overlap with Hamming: {len(overlap)} ({100*len(overlap)/len(centers):.1f}%)")

        return centers

    else:
        if verbose:
            print(f"\n*** FAILED: Not a valid perfect partition ***")
        return None


def iterative_shift_construction(k: int, s: int, max_iterations: int = 1000,
                                  verbose: bool = True) -> Optional[Set[int]]:
    """
    Iterative construction with greedy shift selection.

    Each iteration:
    1. Find uncovered vertices
    2. Find codeword that could cover them (with shift if needed)
    3. Apply shift that minimizes future conflicts
    """
    n = (1 << k) - 1

    if verbose:
        print(f"="*70)
        print(f"Iterative Shift Construction: n={n}, s={s}")
        print(f"="*70)

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    bad = {c for c in hamming if c not in allowed}
    good = hamming - bad

    if verbose:
        print(f"|Good Hamming| = {len(good)}")
        print(f"|Bad Hamming| = {len(bad)}")

    # Start with good codewords
    centers = good.copy()

    # Compute initial coverage
    covered = set()
    for c in centers:
        covered.update(get_ball(c, n) & allowed)
    uncovered = allowed - covered

    if verbose:
        print(f"Initial uncovered: {len(uncovered)}")

    # Track all Hamming codewords and their current state (original or shifted)
    codeword_state = {c: c for c in hamming}  # c -> current position

    iteration = 0
    while uncovered and iteration < max_iterations:
        iteration += 1

        # Find a way to cover some uncovered vertex
        best_action = None
        best_coverage = 0

        for v in list(uncovered)[:50]:  # Sample uncovered
            # Which codewords could cover v?
            # Either the original codeword covering v in Hamming, or a shift of some codeword

            for c in hamming:
                original_ball = get_ball(c, n)
                current_pos = codeword_state[c]

                # Can we shift c to cover v?
                for i in range(n):
                    shifted = c ^ (1 << i)
                    if v not in get_ball(shifted, n):
                        continue  # Doesn't cover v

                    if shifted not in allowed:
                        continue

                    # Check packing with other centers
                    other_centers = {codeword_state[x] for x in hamming if x != c and codeword_state[x] in centers}
                    min_dist = min((hamming_distance(shifted, o) for o in other_centers), default=n+1)
                    if min_dist < 3:
                        continue

                    # Valid shift! How much coverage does it add?
                    new_coverage = len(get_ball(shifted, n) & uncovered)
                    if new_coverage > best_coverage:
                        best_coverage = new_coverage
                        best_action = (c, current_pos, i, shifted)

        if best_action is None:
            if verbose:
                print(f"  Iteration {iteration}: No valid action found!")
            break

        c, current_pos, shift_bit, new_pos = best_action

        if verbose and iteration <= 20:
            print(f"  Iteration {iteration}: Shift {format(c, f'0{n}b')} → {format(new_pos, f'0{n}b')} (covers {best_coverage})")

        # Apply action
        if current_pos in centers:
            centers.remove(current_pos)
        centers.add(new_pos)
        codeword_state[c] = new_pos

        # Update coverage
        covered = set()
        for center in centers:
            covered.update(get_ball(center, n) & allowed)
        uncovered = allowed - covered

    if verbose:
        print(f"\nAfter {iteration} iterations:")
        print(f"  |Centers| = {len(centers)}")
        print(f"  Uncovered: {len(uncovered)}")

    # Verify
    invalid = centers - allowed
    packing_ok = all(hamming_distance(c1, c2) >= 3 for c1, c2 in combinations(centers, 2))

    if len(invalid) == 0 and packing_ok and len(uncovered) == 0:
        if verbose:
            overlap = centers & hamming
            print(f"\n*** SUCCESS! ***")
            print(f"Overlap with Hamming: {len(overlap)} ({100*len(overlap)/len(centers):.1f}%)")
        return centers

    return None


def main():
    """Test constructions."""
    print("Single-Bit Shift Construction Tests")
    print("="*70)

    # Test n=7
    print("\n\n=== N=7, S=4 ===")
    result7 = single_bit_shift_construction(3, 4)

    if not result7:
        print("\nTrying iterative construction...")
        result7 = iterative_shift_construction(3, 4)

    # Test n=15, s=12
    print("\n\n=== N=15, S=12 ===")
    result15_12 = single_bit_shift_construction(4, 12)

    if not result15_12:
        print("\nTrying iterative construction...")
        result15_12 = iterative_shift_construction(4, 12)

    # Test n=15, s=11
    print("\n\n=== N=15, S=11 ===")
    result15_11 = single_bit_shift_construction(4, 11)

    if not result15_11:
        print("\nTrying iterative construction...")
        result15_11 = iterative_shift_construction(4, 11, max_iterations=500)


if __name__ == "__main__":
    main()
