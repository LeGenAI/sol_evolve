#!/usr/bin/env python3
"""
SAT-Guided Cascade Construction

Key insight: The SAT solution shows that ~67-98% of Hamming codewords are shifted.
This suggests we need a GLOBAL optimization, not local repair.

New approach: Use SAT to find shift assignments
- Variables: For each Hamming codeword c, which bit to flip (or stay in place)
- Constraints:
  1. All centers in allowed region
  2. Packing: d >= 3 between all pairs
  3. Covering: every allowed vertex covered

But this is essentially the original SAT problem!

Alternative: Use the SAT SOLUTION as a template to understand the structure,
then try to reconstruct systematically.

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict
from pathlib import Path


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


def analyze_sat_shift_pattern(sat_file: str, k: int, s: int) -> Dict:
    """
    Analyze the shift pattern from SAT solution.

    Returns a dict mapping original Hamming codeword -> (shift_bit, new_position)
    If shift_bit is None, codeword stayed in place.
    """
    n = (1 << k) - 1

    # Load SAT solution
    sat_centers = set(int(v) for v in np.load(sat_file))
    hamming = generate_hamming_code(k)

    # Build shift map
    shift_map = {}

    # For codewords that stayed in place
    stayed = sat_centers & hamming
    for c in stayed:
        shift_map[c] = (None, c)

    # For non-Hamming centers, find their origin
    non_hamming = sat_centers - hamming
    origin_found = set()

    for center in non_hamming:
        # Find Hamming codeword at distance 1
        for h in hamming:
            if hamming_distance(center, h) == 1:
                if h not in origin_found and h not in stayed:
                    # Found origin
                    diff = center ^ h
                    bit_pos = diff.bit_length() - 1
                    shift_map[h] = (bit_pos, center)
                    origin_found.add(h)
                    break

    # Codewords that were removed (all-ones and possibly others)
    removed = hamming - set(shift_map.keys())
    for c in removed:
        shift_map[c] = ('removed', None)

    return shift_map


def reconstruct_from_pattern(shift_map: Dict, k: int, s: int) -> Set[int]:
    """
    Reconstruct solution from shift pattern.
    """
    centers = set()

    for c, (shift_bit, new_pos) in shift_map.items():
        if shift_bit == 'removed':
            continue
        centers.add(new_pos)

    return centers


def verify_solution(centers: Set[int], n: int, s: int) -> Dict:
    """Verify solution."""
    allowed = get_allowed_vertices(n, s)

    # All centers in allowed?
    invalid = centers - allowed

    # Packing
    centers_list = list(centers)
    packing_violations = 0
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i+1:]:
            if hamming_distance(c1, c2) < 3:
                packing_violations += 1

    # Covering
    covered = set()
    for c in centers:
        covered.update(get_ball(c, n) & allowed)
    uncovered = allowed - covered

    return {
        'valid': len(invalid) == 0 and packing_violations == 0 and len(uncovered) == 0,
        'centers': len(centers),
        'invalid': len(invalid),
        'packing_violations': packing_violations,
        'uncovered': len(uncovered)
    }


def generate_shift_cnf(k: int, s: int, output_file: str) -> str:
    """
    Generate CNF for shift assignment problem.

    Variables:
    - For each Hamming codeword c and bit position i: x_{c,i} means "shift c by bit i"
    - x_{c,n} means "don't shift c" (stay in place)
    - x_{c,n+1} means "remove c" (for bad codewords)

    Constraints:
    - Exactly one choice per codeword
    - If shifted position is forbidden, can't choose that shift
    - Packing constraints between possible shifted positions
    - Covering constraints
    """
    n = (1 << k) - 1
    hamming = sorted(generate_hamming_code(k))  # SORTED for consistent ordering!
    allowed = get_allowed_vertices(n, s)

    print(f"Generating shift CNF for n={n}, s={s}")
    print(f"|Hamming| = {len(hamming)}")
    print(f"|Allowed| = {len(allowed)}")

    # Variable mapping
    # var(c, i) = shift c by bit i (i=0..n-1) or don't shift (i=n) or remove (i=n+1)
    var_counter = 1
    var_map = {}  # (c, i) -> var_id

    for c in hamming:
        for i in range(n + 2):  # 0..n-1 for shifts, n for stay, n+1 for remove
            var_map[(c, i)] = var_counter
            var_counter += 1

    clauses = []

    # Constraint 1: Exactly one choice per codeword
    for c in hamming:
        vars_for_c = [var_map[(c, i)] for i in range(n + 2)]

        # At least one
        clauses.append(vars_for_c)

        # At most one (pairwise negations)
        for i in range(len(vars_for_c)):
            for j in range(i + 1, len(vars_for_c)):
                clauses.append([-vars_for_c[i], -vars_for_c[j]])

    # Constraint 2: Shifted position must be in allowed
    for c in hamming:
        for i in range(n):
            shifted = c ^ (1 << i)
            if shifted not in allowed:
                # Can't choose this shift
                clauses.append([-var_map[(c, i)]])

        # If c itself is not in allowed, can't stay
        if c not in allowed:
            clauses.append([-var_map[(c, n)]])

    # Constraint 3: Packing (this is expensive!)
    # For each pair of codewords c1, c2 and each pair of choices,
    # if resulting positions are at distance < 3, exclude

    print("Adding packing constraints (this may take a while)...")

    hamming_list = list(hamming)
    packing_clauses = 0

    for idx1, c1 in enumerate(hamming_list):
        if idx1 % 100 == 0:
            print(f"  Processing codeword {idx1}/{len(hamming_list)}")

        for c2 in hamming_list[idx1 + 1:]:
            # For each pair of choices
            for i1 in range(n + 1):  # n+1 excludes 'remove'
                pos1 = c1 if i1 == n else c1 ^ (1 << i1)
                if pos1 not in allowed:
                    continue

                for i2 in range(n + 1):
                    pos2 = c2 if i2 == n else c2 ^ (1 << i2)
                    if pos2 not in allowed:
                        continue

                    d = hamming_distance(pos1, pos2)
                    if d < 3:
                        # Can't both choose these
                        clauses.append([-var_map[(c1, i1)], -var_map[(c2, i2)]])
                        packing_clauses += 1

    print(f"Added {packing_clauses} packing clauses")

    # Constraint 4: Covering
    # For each allowed vertex v, at least one choice must result in v being covered

    print("Adding covering constraints...")

    covering_clauses = 0

    for v in allowed:
        # Which choices cover v?
        covering_vars = []

        for c in hamming:
            for i in range(n + 1):  # Exclude 'remove'
                pos = c if i == n else c ^ (1 << i)
                if pos not in allowed:
                    continue

                if v in get_ball(pos, n):
                    covering_vars.append(var_map[(c, i)])

        if covering_vars:
            clauses.append(covering_vars)
            covering_clauses += 1
        else:
            print(f"WARNING: Vertex {v} cannot be covered!")

    print(f"Added {covering_clauses} covering clauses")

    # Write CNF
    print(f"\nWriting CNF to {output_file}")
    print(f"Variables: {var_counter - 1}")
    print(f"Clauses: {len(clauses)}")

    with open(output_file, 'w') as f:
        f.write(f"c Shift assignment CNF for n={n}, s={s}\n")
        f.write(f"c Codeword ordering (sorted): {hamming}\n")
        f.write(f"c Variable mapping: var(codeword_idx, choice) = idx * (n+2) + choice + 1\n")
        f.write(f"c where choice = 0..n-1 means shift by bit, n means stay, n+1 means remove\n")
        f.write(f"p cnf {var_counter - 1} {len(clauses)}\n")
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')

    # Also save mapping to file for parsing
    import json
    mapping_file = output_file.replace('.cnf', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump({
            'k': k,
            'n': n,
            's': s,
            'codewords': hamming,
            'choices_per_codeword': n + 2
        }, f)
    print(f"Variable mapping saved to {mapping_file}")

    return output_file


def main():
    """Main entry point."""
    print("SAT-Guided Construction")
    print("="*70)

    # First, analyze existing SAT solutions
    for s_val in [12, 11]:
        sat_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s_val}_centers.npy"

        if Path(sat_file).exists():
            print(f"\n--- Analyzing SAT solution for s={s_val} ---")
            shift_map = analyze_sat_shift_pattern(sat_file, k=4, s=s_val)

            # Count shift types
            stayed = sum(1 for v in shift_map.values() if v[0] is None)
            shifted = sum(1 for v in shift_map.values() if v[0] is not None and v[0] != 'removed')
            removed = sum(1 for v in shift_map.values() if v[0] == 'removed')

            print(f"Stayed in place: {stayed}")
            print(f"Shifted: {shifted}")
            print(f"Removed: {removed}")

            # Reconstruct and verify
            centers = reconstruct_from_pattern(shift_map, k=4, s=s_val)
            result = verify_solution(centers, n=15, s=s_val)
            print(f"Reconstruction valid: {result['valid']}")
            print(f"Details: {result}")

    # Generate shift CNF for n=7 (small enough to be tractable)
    print("\n" + "="*70)
    print("Generating shift CNF for n=7, s=4")
    print("="*70)

    generate_shift_cnf(k=3, s=4, output_file="shift_n7_s4.cnf")


if __name__ == "__main__":
    main()
