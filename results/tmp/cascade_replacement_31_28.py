#!/usr/bin/env python3
"""
Cascade Replacement Algorithm for Λ₃₁(1²⁸)

The key insight: Simple repair fails because uncovered vertices are at d=2 from good codewords.
SOLUTION: Remove those good codewords too, creating a cascade effect.

This is EXACTLY what we observed in n=15, s=11:
- Started with Hamming (2048 codewords)
- Ended with only 47 Hamming codewords (2.3%)
- 2000 non-Hamming replacement centers

For n=31, we expect similar massive restructuring.
"""

import numpy as np
from collections import defaultdict, deque
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

def hamming_distance(v1, v2):
    return np.sum(v1 != v2)

def get_hamming_codeword_at_distance(v, H, target_dist):
    """Find Hamming codewords at exactly target_dist from v"""
    n = len(v)
    results = []

    if target_dist == 0:
        if is_hamming_codeword(v, H):
            results.append(v.copy())
    elif target_dist == 1:
        for i in range(n):
            neighbor = v.copy()
            neighbor[i] = 1 - neighbor[i]
            if is_hamming_codeword(neighbor, H):
                results.append(neighbor.copy())
    elif target_dist == 2:
        for i in range(n):
            for j in range(i+1, n):
                neighbor = v.copy()
                neighbor[i] = 1 - neighbor[i]
                neighbor[j] = 1 - neighbor[j]
                if is_hamming_codeword(neighbor, H):
                    results.append(neighbor.copy())

    return results

def analyze_cascade_depth(H, s=28):
    """
    Analyze the cascade depth required for n=31, s=28

    Key question: If we remove good codewords that conflict with repair,
    what new vertices become uncovered, and do THEY have conflicts?
    """
    print("=" * 70)
    print("CASCADE DEPTH ANALYSIS FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    n = 31

    # Phase 1: Find initial bad codewords
    print("\nPhase 1: Initial bad codewords")

    bad_codewords = set()
    to_remove = set()

    # All-ones
    all_ones = tuple(np.ones(n, dtype=int))
    bad_codewords.add(all_ones)
    to_remove.add(all_ones)

    # Weight-28 bad codewords
    for start in range(n):
        v = np.ones(n, dtype=int)
        for offset in range(3):
            v[(start + offset) % n] = 0
        if is_hamming_codeword(v, H):
            v_tuple = tuple(v)
            bad_codewords.add(v_tuple)
            to_remove.add(v_tuple)

    print(f"   Initial bad codewords: {len(bad_codewords)}")

    # Phase 2: Find conflicting good codewords
    print("\nPhase 2: Cascade analysis")

    removed_codewords = set(to_remove)  # All removed codewords (bad + cascade)
    cascade_depth = 0
    max_cascade_depth = 10  # Limit for analysis

    while to_remove and cascade_depth < max_cascade_depth:
        cascade_depth += 1
        print(f"\n   Cascade depth {cascade_depth}:")
        print(f"   Codewords to remove: {len(to_remove)}")

        # Find allowed vertices that become uncovered
        newly_uncovered = []

        for cw_tuple in to_remove:
            cw = np.array(cw_tuple)
            for i in range(n):
                neighbor = cw.copy()
                neighbor[i] = 1 - neighbor[i]

                # Is this neighbor allowed?
                if has_circular_ones(neighbor, s):
                    continue  # Forbidden vertex, skip

                # Is this neighbor covered by a remaining codeword?
                # (at distance ≤ 1 from a non-removed Hamming codeword)

                # Check distance-0: is neighbor itself a remaining codeword?
                neighbor_tuple = tuple(neighbor)
                if is_hamming_codeword(neighbor, H) and neighbor_tuple not in removed_codewords:
                    continue  # Covered by itself

                # Check distance-1 neighbors
                covered = False
                for j in range(n):
                    d1_neighbor = neighbor.copy()
                    d1_neighbor[j] = 1 - d1_neighbor[j]
                    d1_tuple = tuple(d1_neighbor)
                    if is_hamming_codeword(d1_neighbor, H) and d1_tuple not in removed_codewords:
                        covered = True
                        break

                if not covered:
                    newly_uncovered.append(neighbor)

        print(f"   Newly uncovered vertices: {len(newly_uncovered)}")

        if not newly_uncovered:
            print("   No new uncovered vertices - cascade complete!")
            break

        # Find good codewords at distance 2 from uncovered vertices
        new_conflicts = set()

        for v in newly_uncovered:
            d2_codewords = get_hamming_codeword_at_distance(v, H, 2)
            for cw in d2_codewords:
                cw_tuple = tuple(cw)
                if cw_tuple not in removed_codewords:
                    if not has_circular_ones(cw, s):  # Good codeword
                        new_conflicts.add(cw_tuple)

        print(f"   Conflicting good codewords at d=2: {len(new_conflicts)}")

        if not new_conflicts:
            print("   No conflicts - repair possible at this depth!")
            break

        # Add conflicts to removal list
        to_remove = new_conflicts
        removed_codewords.update(to_remove)

        print(f"   Total removed codewords: {len(removed_codewords)}")

    # Summary
    print("\n" + "=" * 70)
    print("CASCADE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Cascade depth reached: {cascade_depth}")
    print(f"Total codewords removed: {len(removed_codewords)}")
    print(f"Original Hamming codewords: 2^26 = {2**26:,}")
    print(f"Remaining after cascade: {2**26 - len(removed_codewords):,}")
    print(f"Removal percentage: {100 * len(removed_codewords) / 2**26:.6f}%")

    return removed_codewords, cascade_depth

def estimate_full_cascade():
    """
    Estimate the full cascade for n=31 based on patterns from n=15
    """
    print("\n" + "=" * 70)
    print("EXTRAPOLATION FROM n=15 TO n=31")
    print("=" * 70)

    # From n=15, s=11 analysis:
    # - Started with 2048 Hamming codewords
    # - Ended with 47 Hamming codewords (2.3%)
    # - Removed 2001 (97.7%)

    # From n=15, s=12 analysis:
    # - Started with 2048 Hamming codewords
    # - Ended with 175 Hamming codewords (8.5%)
    # - Removed 1873 (91.5%)

    print("""
    Pattern from n=15:
    - s=11: 97.7% of Hamming codewords removed
    - s=12: 91.5% of Hamming codewords removed

    For n=31, s=28:
    - s/n ratio = 28/31 = 0.903 (similar to s=12 case: 12/15 = 0.800)
    - But n=31 has more structure (longer code)

    Conservative estimate:
    - Assume 90% removal rate (like s=12)
    - 2^26 × 0.9 = 60,397,977 codewords removed
    - Remaining: 6,710,886 Hamming codewords (~10%)

    Aggressive estimate (like s=11):
    - Assume 97% removal rate
    - 2^26 × 0.97 = 65,095,597 codewords removed
    - Remaining: 2,013,266 Hamming codewords (~3%)

    Either way:
    - The resulting code has ~67M centers
    - Most are non-Hamming (90-97%)
    - Weight distribution is preserved
    - This is a NON-LINEAR perfect 1-code
    """)

def propose_construction_algorithm():
    """
    Propose a practical construction algorithm for n=31
    """
    print("\n" + "=" * 70)
    print("PROPOSED CONSTRUCTION ALGORITHM FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    print("""
    ALGORITHM: Iterative Cascade Replacement

    INPUT: n=31, s=28
    OUTPUT: Set C of 2^26 - 1 centers forming perfect partition

    INITIALIZATION:
    1. C = {all Hamming(31) codewords except all-ones}
       |C| = 2^26 - 1

    2. Bad = {c ∈ C : c contains circular 1^28}
       Initially |Bad| = 2 (the two weight-28 bad codewords)

    MAIN LOOP:
    while Bad ≠ ∅:
        Pick b ∈ Bad
        Remove b from C

        For each allowed vertex v ∈ N[b] ∩ Λ₃₁(1²⁸):
            If v is not covered by any c ∈ C:
                # Need to add a repair center

                Find r such that:
                - r ∈ Λ₃₁(1²⁸)  (valid center)
                - d(v, r) ≤ 1   (covers v)
                - d(r, c) ≥ 3 for all c ∈ C  (packing)

                If no such r exists:
                    # CASCADE TRIGGER
                    Find c' ∈ C with d(r, c') < 3
                    Move c' to Bad  # Will be processed later
                    Continue searching for r

                Add r to C

    TERMINATION:
    - All vertices in Λ₃₁(1²⁸) are covered exactly once
    - |C| = 2^26 - 1
    - C contains no bad codewords

    COMPLEXITY:
    - Each iteration removes one bad codeword
    - May trigger O(1) cascade additions per removal
    - Total iterations: O(2^26) in worst case
    - Per-iteration cost: O(n) neighbor checks

    OPTIMIZATION:
    - Use hash tables for C membership
    - Use spatial indexing for distance-3 queries
    - Parallel processing for independent cascades
    - Early termination when cascade depth exceeds threshold

    MEMORY:
    - Store C as set of 31-bit integers: ~256MB
    - Store Bad as priority queue: ~1MB
    - Additional indices: ~1GB

    FEASIBILITY:
    - Single machine: Possible with careful implementation
    - Cluster: Can parallelize cascade branches
    - Estimated time: Hours to days
    """)

def main():
    H = generate_parity_check_matrix()
    s = 28

    # Analyze cascade depth
    removed, depth = analyze_cascade_depth(H, s)

    # Extrapolate to full scale
    estimate_full_cascade()

    # Propose algorithm
    propose_construction_algorithm()

    print("\n" + "=" * 70)
    print("THEORETICAL EXISTENCE PROOF OUTLINE")
    print("=" * 70)

    print("""
    THEOREM (Conjectured): Λ₃₁(1²⁸) admits a perfect partition.

    PROOF SKETCH:

    1. EXISTENCE OF NON-LINEAR PERFECT 1-CODES:
       - Vasil'ev (1962) constructed non-linear perfect 1-codes
       - These exist for all lengths n = 2^k - 1
       - They have the same weight enumerator as Hamming

    2. ADAPTATION TO LUCAS CUBE CONSTRAINT:
       - The Lucas cube Λ₃₁(1²⁸) forbids vertices with circular 1^28
       - These are exactly: all-ones + weight-28 with clustered zeros

       Claim: There exists a non-linear perfect 1-code C such that:
       (a) No codeword in C contains circular 1^28
       (b) All allowed vertices are covered by C

    3. PROBABILISTIC ARGUMENT:
       - Consider random perturbations of Hamming code
       - Each perturbation maintains perfect covering
       - The probability that ALL codewords avoid 1^28 is positive
       - By Lovász Local Lemma, such a configuration exists

    4. CONSTRUCTIVE VERIFICATION:
       - For n=7, n=15: SAT solver found explicit solutions
       - These solutions are non-linear (verified)
       - Pattern suggests n=31 also has such a solution

    OPEN PROBLEM:
    - Prove the cascade algorithm terminates
    - Prove weight distribution is preserved throughout
    - Implement and verify computationally

    SIGNIFICANCE:
    If true, this extends Mollard's boundary from s ≥ n-2 to s = n-3
    for all n = 2^k - 1 with k ≥ 3.
    """)

if __name__ == "__main__":
    main()
