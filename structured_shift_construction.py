#!/usr/bin/env python3
"""
Structured Shift Construction based on SAT solution patterns.

Key insight from SAT analysis:
- s=12: Only bits 0,1,2,11,12,13,14 used (syndrome pattern!)
- These correspond to syndromes 1,2,3,12,13,14,15

Strategy:
1. Analyze which syndromes (bit positions) are "safe" to shift
2. Apply shifts only in those directions
3. This reduces search space dramatically

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


def analyze_sat_shift_directions(sat_file: str, k: int, s: int) -> Dict:
    """
    Analyze which bit positions are used for shifts in SAT solution.
    Returns statistics about shift directions.
    """
    n = (1 << k) - 1

    sat_centers = set(int(v) for v in np.load(sat_file))
    hamming = generate_hamming_code(k)

    non_hamming = sat_centers - hamming

    # For each non-Hamming center, find shift direction
    shift_counts = defaultdict(int)

    for c in non_hamming:
        for h in hamming:
            if hamming_distance(c, h) == 1:
                diff = c ^ h
                bit_pos = diff.bit_length() - 1
                shift_counts[bit_pos] += 1
                break

    return dict(shift_counts)


def get_valid_shift_directions(n: int, s: int) -> List[int]:
    """
    Determine which bit positions are "safe" for shifts based on s.

    For s=n-3 (e.g., s=12 for n=15):
    - Bits near the "run boundary" are safe
    - Specifically: bits that break or don't create s consecutive 1s
    """
    # From SAT analysis:
    # s=12 (n=15): bits 0,1,2,11,12,13,14
    # s=11 (n=15): bits with specific pattern

    # General pattern: bits that are far from the "middle" of the string
    # or follow a specific modular pattern

    if s == n - 3:
        # For s = n-3, low and high bits are "safe"
        return [0, 1, 2, n-4, n-3, n-2, n-1]
    elif s == n - 4:
        # For s = n-4, alternating pattern
        return [i for i in range(n) if (i % 4) in [0, 1]]
    else:
        # Default: all bits
        return list(range(n))


def structured_shift_sat(k: int, s: int, allowed_bits: List[int], output_file: str):
    """
    Generate shift SAT with restricted bit positions.

    This dramatically reduces the number of variables and clauses.
    """
    n = (1 << k) - 1
    hamming = sorted(generate_hamming_code(k))
    allowed = get_allowed_vertices(n, s)

    print(f"Structured Shift SAT: n={n}, s={s}")
    print(f"|Hamming| = {len(hamming)}")
    print(f"|Allowed| = {len(allowed)}")
    print(f"Allowed shift bits: {allowed_bits}")

    # Choices per codeword: len(allowed_bits) shifts + stay + remove
    num_choices = len(allowed_bits) + 2

    # Variable mapping
    var_counter = 1
    var_map = {}  # (codeword_idx, choice_idx) -> var_id

    for idx, c in enumerate(hamming):
        for choice in range(num_choices):
            var_map[(idx, choice)] = var_counter
            var_counter += 1

    clauses = []

    # Constraint 1: Exactly one choice per codeword
    print("Adding choice constraints...")
    for idx in range(len(hamming)):
        vars_for_c = [var_map[(idx, choice)] for choice in range(num_choices)]
        clauses.append(vars_for_c)  # At least one

        for i in range(len(vars_for_c)):
            for j in range(i + 1, len(vars_for_c)):
                clauses.append([-vars_for_c[i], -vars_for_c[j]])  # At most one

    # Constraint 2: Shifted positions must be in allowed region
    print("Adding forbidden shift constraints...")
    for idx, c in enumerate(hamming):
        for choice_idx, bit in enumerate(allowed_bits):
            shifted = c ^ (1 << bit)
            if shifted not in allowed:
                clauses.append([-var_map[(idx, choice_idx)]])

        stay_choice = len(allowed_bits)
        if c not in allowed:
            clauses.append([-var_map[(idx, stay_choice)]])

    # Precompute positions and their variables
    print("Computing position mappings...")
    pos_to_vars = defaultdict(list)  # position -> list of (codeword_idx, var)

    for idx, c in enumerate(hamming):
        for choice_idx, bit in enumerate(allowed_bits):
            shifted = c ^ (1 << bit)
            if shifted in allowed:
                pos_to_vars[shifted].append((idx, var_map[(idx, choice_idx)]))

        stay_choice = len(allowed_bits)
        if c in allowed:
            pos_to_vars[c].append((idx, var_map[(idx, stay_choice)]))

    print(f"Unique positions: {len(pos_to_vars)}")

    # Constraint 3: Packing
    print("Adding packing constraints...")
    positions = list(pos_to_vars.keys())
    packing_count = 0

    for i, p1 in enumerate(positions):
        if i % 5000 == 0:
            print(f"  Position {i}/{len(positions)}, clauses so far: {len(clauses)}")

        for p2 in positions[i+1:]:
            if hamming_distance(p1, p2) < 3:
                for _, var1 in pos_to_vars[p1]:
                    for _, var2 in pos_to_vars[p2]:
                        clauses.append([-var1, -var2])
                        packing_count += 1

    print(f"Added {packing_count} packing clauses")

    # Constraint 4: Covering
    print("Adding covering constraints...")
    covering_count = 0

    for v in allowed:
        covering_vars = []

        for pos, var_list in pos_to_vars.items():
            if v in get_ball(pos, n):
                for _, var in var_list:
                    covering_vars.append(var)

        if covering_vars:
            # Remove duplicates while preserving order
            covering_vars = list(dict.fromkeys(covering_vars))
            clauses.append(covering_vars)
            covering_count += 1
        else:
            print(f"WARNING: Vertex {v} cannot be covered!")

    print(f"Added {covering_count} covering clauses")

    # Write CNF
    print(f"\nWriting CNF to {output_file}")
    print(f"Variables: {var_counter - 1}")
    print(f"Clauses: {len(clauses)}")

    with open(output_file, 'w') as f:
        f.write(f"c Structured Shift CNF for n={n}, s={s}\n")
        f.write(f"c Allowed bits: {allowed_bits}\n")
        f.write(f"p cnf {var_counter - 1} {len(clauses)}\n")
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')

    # Save mapping
    import json
    mapping_file = output_file.replace('.cnf', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump({
            'k': k,
            'n': n,
            's': s,
            'codewords': hamming,
            'allowed_bits': allowed_bits,
            'choices_per_codeword': num_choices
        }, f)
    print(f"Mapping saved to {mapping_file}")

    return output_file


def verify_solution(centers: Set[int], n: int, s: int) -> Dict:
    """Verify solution is valid."""
    allowed = get_allowed_vertices(n, s)

    invalid = centers - allowed

    centers_list = list(centers)
    packing_violations = 0
    min_dist = float('inf')
    for i, c1 in enumerate(centers_list):
        for c2 in centers_list[i+1:]:
            d = hamming_distance(c1, c2)
            min_dist = min(min_dist, d)
            if d < 3:
                packing_violations += 1

    covered = set()
    for c in centers:
        covered.update(get_ball(c, n) & allowed)
    uncovered = allowed - covered

    return {
        'valid': len(invalid) == 0 and packing_violations == 0 and len(uncovered) == 0,
        'centers': len(centers),
        'invalid': len(invalid),
        'packing_violations': packing_violations,
        'min_distance': min_dist if min_dist < float('inf') else 0,
        'covered': len(covered),
        'total_allowed': len(allowed),
        'uncovered': len(uncovered)
    }


def main():
    """Main entry point."""
    from pathlib import Path
    import subprocess
    import json

    print("="*70)
    print("Structured Shift Construction")
    print("="*70)

    # First, analyze SAT solution patterns
    for s_val in [12, 11]:
        sat_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s_val}_centers.npy"

        if Path(sat_file).exists():
            print(f"\n--- SAT solution shift directions for s={s_val} ---")
            shift_dirs = analyze_sat_shift_directions(sat_file, k=4, s=s_val)

            total = sum(shift_dirs.values())
            print(f"Total shifts: {total}")
            for bit in sorted(shift_dirs.keys()):
                print(f"  Bit {bit}: {shift_dirs[bit]} ({100*shift_dirs[bit]/total:.1f}%)")

    # Generate and solve for n=7, s=4 with restricted bits
    print("\n" + "="*70)
    print("Testing n=7, s=4 with restricted bits")
    print("="*70)

    # For n=7, s=4: based on pattern, bits 0,1,2,4,5,6 might be used
    # But let's use all bits for n=7 since it's small
    allowed_bits_7 = [0, 1, 2, 3, 4, 5, 6]

    structured_shift_sat(k=3, s=4, allowed_bits=allowed_bits_7,
                         output_file="structured_shift_n7_s4.cnf")

    # Solve
    result = subprocess.run(
        ['../sat_solvers/cadical/build/cadical', 'structured_shift_n7_s4.cnf'],
        capture_output=True, text=True, timeout=60
    )

    if 'SATISFIABLE' in result.stdout:
        print("\n*** n=7, s=4 SATISFIABLE! ***")
    else:
        print("\n*** n=7, s=4 result: ***")
        print(result.stdout[-500:])

    # Now try n=15, s=12 with ONLY the bits from SAT analysis
    print("\n" + "="*70)
    print("Testing n=15, s=12 with restricted bits [0,1,2,11,12,13,14]")
    print("="*70)

    allowed_bits_15_12 = [0, 1, 2, 11, 12, 13, 14]

    structured_shift_sat(k=4, s=12, allowed_bits=allowed_bits_15_12,
                         output_file="structured_shift_n15_s12.cnf")

    # Solve with timeout
    print("\nRunning SAT solver...")
    try:
        result = subprocess.run(
            ['../sat_solvers/cadical/build/cadical', 'structured_shift_n15_s12.cnf', '-t', '300'],
            capture_output=True, text=True, timeout=600
        )

        if 'SATISFIABLE' in result.stdout:
            print("\n*** n=15, s=12 SATISFIABLE! ***")

            # Parse and verify
            model = {}
            for line in result.stdout.split('\n'):
                if line.startswith('v '):
                    for p in line[2:].split():
                        if p == '0':
                            continue
                        var = int(p)
                        model[abs(var)] = var > 0

            with open('structured_shift_n15_s12_mapping.json', 'r') as f:
                mapping = json.load(f)

            codewords = mapping['codewords']
            allowed_bits = mapping['allowed_bits']
            num_choices = mapping['choices_per_codeword']
            n = mapping['n']
            s = mapping['s']

            centers = set()
            for idx, c in enumerate(codewords):
                for choice in range(num_choices):
                    var = idx * num_choices + choice + 1
                    if model.get(var, False):
                        if choice < len(allowed_bits):
                            shifted = c ^ (1 << allowed_bits[choice])
                            centers.add(shifted)
                        elif choice == len(allowed_bits):
                            centers.add(c)
                        # else: removed
                        break

            result = verify_solution(centers, n, s)
            print(f"\nVerification: {result}")

            if result['valid']:
                np.save('structured_shift_n15_s12_centers.npy', np.array(list(centers)))
                print("Saved to structured_shift_n15_s12_centers.npy")

        elif 'UNSATISFIABLE' in result.stdout:
            print("\n*** n=15, s=12 UNSATISFIABLE with restricted bits! ***")
            print("Need to use more bit positions.")
        else:
            print("\nSolver output:")
            print(result.stdout[-1000:])

    except subprocess.TimeoutExpired:
        print("Solver timeout!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
