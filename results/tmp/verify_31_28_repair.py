#!/usr/bin/env python3
"""
Re-verify the n=31, s=28 repair possibility.

Key question: Are the 56 allowed uncovered vertices at distance ≥3 from all good Hamming codewords?
If YES, then simple repair IS possible!
"""

import numpy as np
from collections import defaultdict

def generate_parity_check_matrix():
    """Generate H matrix for Hamming(31)"""
    H = np.array([[int(b) for b in format(i, '05b')] for i in range(1, 32)]).T
    return H

def compute_syndrome(v, H):
    """Compute syndrome"""
    return tuple((H @ v) % 2)

def is_hamming_codeword(v, H):
    """Check if v is a Hamming codeword"""
    syn = compute_syndrome(v, H)
    return all(s == 0 for s in syn)

def has_circular_ones(v, s):
    """Check if v has s consecutive 1s in circular form"""
    n = len(v)
    doubled = list(v) + list(v)
    for i in range(n):
        if all(doubled[i+j] == 1 for j in range(s)):
            return True
    return False

def get_nearest_codeword(v, H):
    """Get the nearest Hamming codeword to v"""
    syn = compute_syndrome(v, H)
    if all(s == 0 for s in syn):
        return v.copy(), 0

    # Syndrome gives error position (1-indexed)
    error_pos = int(''.join(map(str, syn)), 2) - 1
    if 0 <= error_pos < 31:
        corrected = v.copy()
        corrected[error_pos] = 1 - corrected[error_pos]
        return corrected, 1
    else:
        return None, -1

def main():
    print("=" * 70)
    print("RE-VERIFICATION: CAN Λ₃₁(1²⁸) BE REPAIRED?")
    print("=" * 70)

    H = generate_parity_check_matrix()
    s = 28
    n = 31

    # Find bad Hamming codewords
    print("\n1. Finding bad Hamming codewords...")

    bad_codewords = []

    # All-ones
    all_ones = np.ones(n, dtype=int)
    if is_hamming_codeword(all_ones, H):
        bad_codewords.append(('all-ones', all_ones))

    # Weight-28: zeros at contiguous positions
    for start in range(n):
        v = np.ones(n, dtype=int)
        for offset in range(3):
            v[(start + offset) % n] = 0
        if is_hamming_codeword(v, H):
            zeros = [(start + offset) % n for offset in range(3)]
            bad_codewords.append((f'zeros-at-{zeros}', v))

    print(f"   Found {len(bad_codewords)} bad codewords")
    for name, v in bad_codewords:
        print(f"   - {name}: {''.join(map(str, v))}")

    # Find allowed vertices in neighborhoods of bad codewords
    print("\n2. Finding allowed vertices that need coverage...")

    uncovered_vertices = []

    for name, bad in bad_codewords:
        for i in range(n):
            neighbor = bad.copy()
            neighbor[i] = 1 - neighbor[i]

            if not has_circular_ones(neighbor, s):
                # This is an allowed vertex
                uncovered_vertices.append({
                    'vertex': neighbor.copy(),
                    'from_bad': name,
                    'flip_pos': i
                })

    print(f"   Found {len(uncovered_vertices)} allowed vertices needing coverage")

    # For each uncovered vertex, find its nearest Hamming codeword
    print("\n3. Analyzing distance to nearest GOOD Hamming codeword...")

    repair_possible = True
    problematic = []

    for uv in uncovered_vertices:
        v = uv['vertex']
        nearest, dist = get_nearest_codeword(v, H)

        if nearest is None:
            print(f"   ERROR: Could not find nearest codeword")
            continue

        # Is nearest codeword bad or good?
        nearest_is_bad = has_circular_ones(nearest, s)

        if dist == 0:
            # v itself is a Hamming codeword
            if nearest_is_bad:
                status = "v is BAD codeword"
            else:
                status = "v is GOOD codeword (can cover itself)"
        elif dist == 1:
            # v is at distance 1 from a Hamming codeword
            if nearest_is_bad:
                status = "nearest is BAD (need to find another covering center)"
                # This is the CRITICAL case: is there a GOOD codeword at distance 2?

                # Check all distance-2 neighbors of v
                has_good_d2 = False
                good_d2_list = []

                for i in range(n):
                    for j in range(i+1, n):
                        d2_neighbor = v.copy()
                        d2_neighbor[i] = 1 - d2_neighbor[i]
                        d2_neighbor[j] = 1 - d2_neighbor[j]

                        if is_hamming_codeword(d2_neighbor, H):
                            if not has_circular_ones(d2_neighbor, s):
                                has_good_d2 = True
                                good_d2_list.append((i, j))

                if has_good_d2:
                    status += f" BUT good codeword at d=2 (conflicts!)"
                    repair_possible = False
                    problematic.append({
                        'vertex': v,
                        'from_bad': uv['from_bad'],
                        'good_d2_positions': good_d2_list[:3]
                    })
                else:
                    status += " and NO good codeword at d=2 (repair OK)"
            else:
                status = "nearest is GOOD (already covered)"

        uv['nearest_dist'] = dist
        uv['nearest_is_bad'] = nearest_is_bad
        uv['status'] = status

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    bad_nearest_count = sum(1 for uv in uncovered_vertices if uv.get('nearest_is_bad', False))
    good_nearest_count = len(uncovered_vertices) - bad_nearest_count

    print(f"\nUncovered vertices with nearest=BAD: {bad_nearest_count}")
    print(f"Uncovered vertices with nearest=GOOD: {good_nearest_count}")

    if repair_possible:
        print("\n*** REPAIR IS POSSIBLE! ***")
        print("All uncovered vertices can be covered by repair centers")
        print("without conflicting with good Hamming codewords.")
    else:
        print("\n*** REPAIR IS NOT POSSIBLE (simple approach) ***")
        print(f"Found {len(problematic)} vertices with good codeword at d=2")
        print("\nProblematic vertices (first 5):")
        for p in problematic[:5]:
            v_str = ''.join(map(str, p['vertex']))
            print(f"  {v_str[:30]}... from {p['from_bad']}")
            print(f"    Good codewords at d=2: {p['good_d2_positions']}")

    # Check if problematic vertices can share a repair center
    if problematic:
        print("\n4. Checking if problematic vertices can share repair centers...")

        # Group by distance
        from itertools import combinations

        groups = []
        remaining = list(range(len(problematic)))

        while remaining:
            group = [remaining[0]]
            for i in remaining[1:]:
                # Check if vertex i is within distance 2 of any vertex in group
                can_join = False
                for j in group:
                    d = np.sum(problematic[i]['vertex'] != problematic[j]['vertex'])
                    if d <= 2:
                        can_join = True
                        break
                if can_join:
                    group.append(i)

            for i in group:
                remaining.remove(i)
            groups.append(group)

        print(f"   Problematic vertices form {len(groups)} groups")
        for gi, group in enumerate(groups[:5]):
            print(f"   Group {gi+1}: {len(group)} vertices")

    # THEORETICAL ANALYSIS
    print("\n" + "=" * 70)
    print("THEORETICAL IMPLICATIONS")
    print("=" * 70)

    print("""
    For n=31, s=28:

    1. SIMPLE REPAIR ANALYSIS:
       - 3 bad Hamming codewords (all-ones + 2 weight-28)
       - 56 allowed vertices need coverage after removing bad codewords
       - Each allowed vertex is at distance 1 from its original bad codeword

    2. THE KEY QUESTION:
       For each allowed vertex v:
       - Its nearest Hamming codeword c has d(v,c) = 0 or 1
       - If c is bad, we need to find a repair center r with d(r,v) ≤ 1
       - r must have d(r, c') ≥ 3 for all good codewords c'

    3. REPAIR OBSTRUCTION:
       If v has a GOOD codeword c' at d(v, c') = 2, then:
       - Any repair center r with d(r,v) ≤ 1 has d(r, c') ≤ 3
       - If d(r, c') < 3, packing is violated

    4. THE NON-LINEAR SOLUTION:
       From n=15 analysis, we know SAT finds codes that are:
       - NOT modifications of Hamming
       - Non-linear (27% XOR closure)
       - Same weight distribution as Hamming

       This suggests n=31 may also have a non-linear solution that:
       - Starts from scratch (not from Hamming)
       - Or uses massive cascade replacement
    """)

    return repair_possible, problematic

if __name__ == "__main__":
    possible, problems = main()
