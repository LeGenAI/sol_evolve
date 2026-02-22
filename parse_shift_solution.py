#!/usr/bin/env python3
"""
Parse SAT solution for shift assignment problem.

Converts SAT model to actual center set.

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, Dict, Tuple
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


def parse_sat_output(output: str) -> Dict[int, bool]:
    """Parse SAT solver output to get variable assignments."""
    model = {}
    for line in output.split('\n'):
        if line.startswith('v '):
            parts = line[2:].split()
            for p in parts:
                if p == '0':
                    continue
                var = int(p)
                if var > 0:
                    model[var] = True
                else:
                    model[-var] = False
    return model


def reconstruct_centers_from_shift_model(model: Dict[int, bool], k: int, s: int) -> Set[int]:
    """
    Reconstruct center set from SAT model.

    Variable mapping:
    - For codeword index i (0-indexed) and choice j (0 to n+1):
      var = i * (n+2) + j + 1
    - j = 0..n-1: shift by bit j
    - j = n: stay in place
    - j = n+1: remove
    """
    n = (1 << k) - 1
    hamming = sorted(generate_hamming_code(k))
    allowed = get_allowed_vertices(n, s)

    centers = set()
    shift_info = []

    for idx, c in enumerate(hamming):
        # Find which choice was selected
        for j in range(n + 2):
            var = idx * (n + 2) + j + 1
            if model.get(var, False):
                if j < n:
                    # Shift by bit j
                    shifted = c ^ (1 << j)
                    centers.add(shifted)
                    shift_info.append((c, j, shifted))
                elif j == n:
                    # Stay in place
                    centers.add(c)
                    shift_info.append((c, 'stay', c))
                else:
                    # Remove
                    shift_info.append((c, 'remove', None))
                break

    return centers, shift_info


def verify_solution(centers: Set[int], n: int, s: int) -> Dict:
    """Verify solution is a valid perfect partition."""
    allowed = get_allowed_vertices(n, s)

    # All centers in allowed?
    invalid = centers - allowed

    # Packing
    centers_list = list(centers)
    packing_violations = 0
    min_dist = float('inf')
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i+1:]:
            d = hamming_distance(c1, c2)
            min_dist = min(min_dist, d)
            if d < 3:
                packing_violations += 1

    # Covering
    covered = set()
    for c in centers:
        covered.update(get_ball(c, n) & allowed)
    uncovered = allowed - covered

    return {
        'valid': len(invalid) == 0 and packing_violations == 0 and len(uncovered) == 0,
        'centers': len(centers),
        'expected': len(allowed) // (n + 1),
        'invalid': len(invalid),
        'packing_violations': packing_violations,
        'min_distance': min_dist if min_dist < float('inf') else 0,
        'covered': len(covered),
        'total_allowed': len(allowed),
        'uncovered': len(uncovered)
    }


def reconstruct_from_mapping(model: Dict[int, bool], mapping: Dict) -> Tuple[Set[int], list]:
    """
    Reconstruct center set from SAT model using saved mapping.
    """
    codewords = mapping['codewords']
    n = mapping['n']
    choices = mapping['choices_per_codeword']

    centers = set()
    shift_info = []

    for idx, c in enumerate(codewords):
        # Find which choice was selected
        for j in range(choices):
            var = idx * choices + j + 1
            if model.get(var, False):
                if j < n:
                    # Shift by bit j
                    shifted = c ^ (1 << j)
                    centers.add(shifted)
                    shift_info.append((c, j, shifted))
                elif j == n:
                    # Stay in place
                    centers.add(c)
                    shift_info.append((c, 'stay', c))
                else:
                    # Remove
                    shift_info.append((c, 'remove', None))
                break

    return centers, shift_info


def main():
    """Test with n=7, s=4 SAT output."""
    import json
    import subprocess

    # Load mapping
    with open('shift_n7_s4_mapping.json', 'r') as f:
        mapping = json.load(f)

    k = mapping['k']
    n = mapping['n']
    s = mapping['s']

    print("="*70)
    print(f"Solving and parsing shift CNF for n={n}, s={s}")
    print("="*70)

    # Run SAT solver
    result = subprocess.run(
        ['../sat_solvers/cadical/build/cadical', 'shift_n7_s4.cnf'],
        capture_output=True, text=True, timeout=60
    )
    sat_output = result.stdout

    if 'SATISFIABLE' not in sat_output:
        print("UNSATISFIABLE!")
        return

    model = parse_sat_output(sat_output)
    print(f"\nParsed {len(model)} variable assignments")
    true_vars = [v for v, val in model.items() if val]
    print(f"True variables ({len(true_vars)}): {true_vars}")

    centers, shift_info = reconstruct_from_mapping(model, mapping)

    print(f"\n--- Shift Assignments ---")
    hamming = set(mapping['codewords'])
    stayed = sum(1 for _, action, _ in shift_info if action == 'stay')
    shifted = sum(1 for _, action, _ in shift_info if isinstance(action, int))
    removed = sum(1 for _, action, _ in shift_info if action == 'remove')

    print(f"Stayed: {stayed}")
    print(f"Shifted: {shifted}")
    print(f"Removed: {removed}")

    print(f"\n--- Shift Details ---")
    for orig, action, result in shift_info:
        orig_bits = format(orig, f'0{n}b')
        if action == 'stay':
            print(f"  {orig_bits} -> stay")
        elif action == 'remove':
            print(f"  {orig_bits} -> REMOVED")
        else:
            result_bits = format(result, f'0{n}b')
            print(f"  {orig_bits} -> {result_bits} (bit {action})")

    print(f"\n--- Verification ---")
    verification = verify_solution(centers, n, s)
    print(f"Valid: {verification['valid']}")
    print(f"Centers: {verification['centers']} (expected: {verification['expected']})")
    print(f"Invalid centers: {verification['invalid']}")
    print(f"Packing violations: {verification['packing_violations']}")
    print(f"Minimum distance: {verification['min_distance']}")
    print(f"Covered: {verification['covered']} / {verification['total_allowed']}")
    print(f"Uncovered: {verification['uncovered']}")

    if verification['valid']:
        print("\n*** SUCCESS! Valid perfect partition constructed! ***")

        # Compare with Hamming
        overlap = centers & hamming
        non_hamming = centers - hamming
        print(f"\nOverlap with Hamming: {len(overlap)} ({100*len(overlap)/len(centers):.1f}%)")
        print(f"Non-Hamming centers: {len(non_hamming)}")

        # Save centers
        np.save('shift_constructed_n7_s4_centers.npy', np.array(list(centers)))
        print(f"\nSaved to shift_constructed_n7_s4_centers.npy")
    else:
        print("\n*** FAILED: Not a valid perfect partition ***")
        # Debug: show centers
        print(f"\nCenters ({len(centers)}): {sorted(centers)}")


if __name__ == "__main__":
    main()
