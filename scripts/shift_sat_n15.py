#!/usr/bin/env python3
"""
Generate and solve shift assignment SAT for n=15, s=12.

This will be larger but should be tractable:
- |Hamming| = 2048
- Variables: 2048 * (15+2) = 34,816
- Packing clauses: O(2048^2 * 17^2) ~ millions

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
import json
import subprocess
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


def generate_shift_cnf_optimized(k: int, s: int, output_file: str) -> str:
    """
    Generate CNF for shift assignment problem with optimizations.

    Key optimization: Only generate packing clauses for positions that
    actually conflict (distance < 3).
    """
    n = (1 << k) - 1
    hamming = sorted(generate_hamming_code(k))
    allowed = get_allowed_vertices(n, s)

    print(f"Generating shift CNF for n={n}, s={s}")
    print(f"|Hamming| = {len(hamming)}")
    print(f"|Allowed| = {len(allowed)}")

    # Precompute valid positions for each codeword
    valid_choices = {}  # codeword -> list of (choice_idx, result_position)
    for idx, c in enumerate(hamming):
        valid_choices[c] = []
        for i in range(n):
            shifted = c ^ (1 << i)
            if shifted in allowed:
                valid_choices[c].append((i, shifted))
        if c in allowed:
            valid_choices[c].append((n, c))  # stay
        valid_choices[c].append((n + 1, None))  # remove

    print(f"Average valid choices per codeword: {sum(len(v) for v in valid_choices.values()) / len(hamming):.1f}")

    # Variable mapping
    var_counter = 1
    var_map = {}  # (c, choice_idx) -> var_id

    for c in hamming:
        for i in range(n + 2):
            var_map[(c, i)] = var_counter
            var_counter += 1

    clauses = []

    # Constraint 1: Exactly one choice per codeword
    print("Adding choice constraints...")
    for c in hamming:
        vars_for_c = [var_map[(c, i)] for i in range(n + 2)]

        # At least one
        clauses.append(vars_for_c)

        # At most one (pairwise negations)
        for i in range(len(vars_for_c)):
            for j in range(i + 1, len(vars_for_c)):
                clauses.append([-vars_for_c[i], -vars_for_c[j]])

    # Constraint 2: Invalid shifts forbidden
    print("Adding forbidden shift constraints...")
    for c in hamming:
        for i in range(n):
            shifted = c ^ (1 << i)
            if shifted not in allowed:
                clauses.append([-var_map[(c, i)]])
        if c not in allowed:
            clauses.append([-var_map[(c, n)]])  # can't stay

    # Constraint 3: Packing (distance >= 3 between centers)
    print("Adding packing constraints (this may take a while)...")

    # Precompute all possible positions
    position_to_vars = {}  # position -> list of (codeword, choice_idx, var)
    for c in hamming:
        for choice_idx, pos in valid_choices[c]:
            if pos is not None:  # not 'remove'
                if pos not in position_to_vars:
                    position_to_vars[pos] = []
                position_to_vars[pos].append((c, choice_idx, var_map[(c, choice_idx)]))

    packing_clauses = 0
    positions = list(position_to_vars.keys())

    # For each pair of positions at distance < 3
    for i, p1 in enumerate(positions):
        if i % 1000 == 0:
            print(f"  Processing position {i}/{len(positions)}")

        for p2 in positions[i+1:]:
            if hamming_distance(p1, p2) < 3:
                # Add clauses: can't both be chosen
                for _, _, var1 in position_to_vars[p1]:
                    for _, _, var2 in position_to_vars[p2]:
                        clauses.append([-var1, -var2])
                        packing_clauses += 1

    print(f"Added {packing_clauses} packing clauses")

    # Constraint 4: Covering
    print("Adding covering constraints...")
    covering_clauses = 0

    for v in allowed:
        # Which choices cover v?
        covering_vars = []
        for c in hamming:
            for choice_idx, pos in valid_choices[c]:
                if pos is not None and v in get_ball(pos, n):
                    covering_vars.append(var_map[(c, choice_idx)])

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
        f.write(f"p cnf {var_counter - 1} {len(clauses)}\n")
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')

    # Save mapping
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
    """Generate shift CNF for n=15, s=12."""
    print("="*70)
    print("Shift SAT for n=15, s=12")
    print("="*70)

    # This might take a few minutes
    generate_shift_cnf_optimized(k=4, s=12, output_file="shift_n15_s12.cnf")

    print("\n" + "="*70)
    print("CNF generated! Running SAT solver...")
    print("="*70)

    # Run SAT solver with timeout
    try:
        result = subprocess.run(
            ['../sat_solvers/cadical/build/cadical', 'shift_n15_s12.cnf', '-t', '300'],
            capture_output=True, text=True, timeout=600
        )

        if 'SATISFIABLE' in result.stdout:
            print("\n*** SATISFIABLE! ***")

            # Parse solution
            model = {}
            for line in result.stdout.split('\n'):
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

            # Load mapping and reconstruct
            with open('shift_n15_s12_mapping.json', 'r') as f:
                mapping = json.load(f)

            codewords = mapping['codewords']
            n = mapping['n']
            s = mapping['s']
            choices = mapping['choices_per_codeword']

            centers = set()
            stayed = 0
            shifted = 0
            removed = 0

            for idx, c in enumerate(codewords):
                for j in range(choices):
                    var = idx * choices + j + 1
                    if model.get(var, False):
                        if j < n:
                            shifted_pos = c ^ (1 << j)
                            centers.add(shifted_pos)
                            shifted += 1
                        elif j == n:
                            centers.add(c)
                            stayed += 1
                        else:
                            removed += 1
                        break

            print(f"\nStayed: {stayed}")
            print(f"Shifted: {shifted}")
            print(f"Removed: {removed}")
            print(f"Total centers: {len(centers)}")

            # Verify
            allowed = get_allowed_vertices(n, s)
            invalid = centers - allowed
            covered = set()
            for c in centers:
                covered.update(get_ball(c, n) & allowed)
            uncovered = allowed - covered

            print(f"\nInvalid centers: {len(invalid)}")
            print(f"Covered: {len(covered)} / {len(allowed)}")
            print(f"Uncovered: {len(uncovered)}")

            # Check packing
            min_dist = float('inf')
            centers_list = list(centers)
            for i in range(len(centers_list)):
                for j in range(i+1, len(centers_list)):
                    d = hamming_distance(centers_list[i], centers_list[j])
                    min_dist = min(min_dist, d)
            print(f"Minimum distance: {min_dist}")

            if len(invalid) == 0 and len(uncovered) == 0 and min_dist >= 3:
                print("\n*** SUCCESS! Valid perfect partition! ***")
                np.save('shift_constructed_n15_s12_centers.npy', np.array(list(centers)))
                print("Saved to shift_constructed_n15_s12_centers.npy")

        elif 'UNSATISFIABLE' in result.stdout:
            print("\nUNSATISFIABLE!")
        else:
            print("\nTimeout or unknown result")
            print(result.stdout[-2000:])

    except subprocess.TimeoutExpired:
        print("Solver timeout!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
