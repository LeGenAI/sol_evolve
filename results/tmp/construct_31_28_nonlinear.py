#!/usr/bin/env python3
"""
Construct a non-linear perfect 1-code for Λ₃₁(1²⁸)

Strategy: Weight-preserving iterative replacement
- Start with Hamming(31) code
- Iteratively replace bad codewords with valid centers
- Allow cascade replacements to maintain distance ≥ 3
- Preserve weight distribution throughout

Key insight from Λ₁₅(1¹¹):
- Final code may share only ~2% with Hamming
- But weight distribution is preserved
- Non-linear structure (27% XOR closure)
"""

import numpy as np
from collections import defaultdict
import random
import time

def generate_hamming_31():
    """Generate Hamming(31) code via parity check matrix"""
    print("Generating Hamming(31) code...")

    # Parity check matrix H: each column is binary rep of 1..31
    H = np.array([[int(b) for b in format(i, '05b')] for i in range(1, 32)]).T

    # For n=31, we have 2^26 = 67,108,864 codewords
    # This is too large to enumerate directly
    # Instead, use systematic encoding

    # Generator matrix G for systematic Hamming [31,26,3]
    # G = [I_26 | P] where P is 26x5 parity matrix

    # Actually, let's work with syndrome-based approach
    # A codeword c has syndrome H*c = 0

    print("Note: Full enumeration of 2^26 codewords is expensive")
    print("Using syndrome-based approach for membership testing")

    return H

def compute_syndrome(v, H):
    """Compute syndrome of v under parity check matrix H"""
    return tuple((H @ v) % 2)

def is_hamming_codeword(v, H):
    """Check if v is a Hamming codeword (syndrome = 0)"""
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

def hamming_distance(v1, v2):
    """Compute Hamming distance"""
    return np.sum(v1 != v2)

def count_bad_hamming_codewords(H, s, sample_size=10000):
    """Estimate number of bad Hamming codewords by sampling"""
    print(f"\nEstimating bad Hamming codewords for s={s}...")

    bad_count = 0
    total_sampled = 0

    # Sample random codewords
    for _ in range(sample_size):
        # Generate random 26-bit information
        info = np.random.randint(0, 2, 26)

        # Encode using systematic form
        # For Hamming code, parity bits at positions 1,2,4,8,16 (1-indexed)
        # Data bits at other positions

        # Simpler: generate random syndrome-0 vectors
        # Take random 26 bits and compute unique codeword

        # Actually, let's just sample random 31-bit vectors and filter
        v = np.random.randint(0, 2, 31)
        syn = compute_syndrome(v, H)

        # Correct to nearest codeword
        if all(s == 0 for s in syn):
            # v is already a codeword
            codeword = v
        else:
            # Flip the bit indicated by syndrome
            error_pos = int(''.join(map(str, syn)), 2) - 1
            if 0 <= error_pos < 31:
                codeword = v.copy()
                codeword[error_pos] = 1 - codeword[error_pos]
            else:
                continue

        total_sampled += 1
        if has_circular_ones(codeword, s):
            bad_count += 1

    # Estimate total bad codewords
    bad_ratio = bad_count / total_sampled if total_sampled > 0 else 0
    total_codewords = 2**26
    estimated_bad = int(bad_ratio * total_codewords)

    print(f"  Sampled: {total_sampled}, Bad: {bad_count} ({100*bad_ratio:.2f}%)")
    print(f"  Estimated total bad codewords: ~{estimated_bad}")

    return estimated_bad, bad_ratio

def find_bad_codewords_exact(H, s, max_weight=5):
    """Find bad codewords up to a certain weight"""
    print(f"\nFinding bad codewords with weight ≤ {max_weight}...")

    bad_codewords = []

    # Weight 0: all zeros (never bad for s ≥ 1)

    # Weight 31: all ones (always bad for s ≤ 31)
    all_ones = np.ones(31, dtype=int)
    if is_hamming_codeword(all_ones, H):
        bad_codewords.append(('1'*31, 31))
        print(f"  All-ones is a Hamming codeword: YES")

    # Find high-weight bad codewords (weight ≥ 31-3 = 28)
    print(f"\nFinding high-weight bad codewords (weight ≥ 28)...")

    # Weight 28: zeros at exactly 3 positions in a row (circularly)
    for start in range(31):
        v = np.ones(31, dtype=int)
        for offset in range(3):
            v[(start + offset) % 31] = 0

        if is_hamming_codeword(v, H) and has_circular_ones(v, s):
            v_str = ''.join(map(str, v))
            zeros = [(start + offset) % 31 for offset in range(3)]
            bad_codewords.append((v_str, 28, zeros))
            print(f"  Bad: zeros at {zeros}")

    # Weight 27: zeros at exactly 4 positions
    # For s=28, need 28 consecutive 1s, so 3 zeros must be in arc of 3
    print(f"\nFinding weight-27 bad codewords...")
    count_27 = 0
    for start in range(31):
        for extra in range(31):
            if extra in [(start + offset) % 31 for offset in range(4)]:
                continue
            v = np.ones(31, dtype=int)
            for offset in range(3):
                v[(start + offset) % 31] = 0
            v[extra] = 0

            if is_hamming_codeword(v, H) and has_circular_ones(v, s):
                count_27 += 1
                if count_27 <= 5:
                    zeros = sorted([(start + offset) % 31 for offset in range(3)] + [extra])
                    print(f"  Bad weight-27: zeros at {zeros}")

    if count_27 > 5:
        print(f"  ... and {count_27 - 5} more weight-27 bad codewords")

    return bad_codewords

def analyze_neighborhood_conflict(H, s):
    """
    For n=31, s=28, analyze:
    1. How many bad codewords exist
    2. What are the allowed vertices in their neighborhoods
    3. What's the conflict structure with good codewords
    """
    print("\n" + "=" * 70)
    print("NEIGHBORHOOD CONFLICT ANALYSIS FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    # Find all weight-28+ bad codewords
    all_ones = np.ones(31, dtype=int)

    bad_codewords = []

    # All-ones (weight 31)
    if is_hamming_codeword(all_ones, H):
        bad_codewords.append(all_ones)

    # Weight 28: zeros in contiguous arc of 3
    for start in range(31):
        v = np.ones(31, dtype=int)
        for offset in range(3):
            v[(start + offset) % 31] = 0
        if is_hamming_codeword(v, H):
            bad_codewords.append(v.copy())

    print(f"\nBad codewords found: {len(bad_codewords)}")

    # For each bad codeword, find allowed neighbors
    total_allowed_uncovered = 0
    conflict_info = []

    for bad in bad_codewords:
        bad_str = ''.join(map(str, bad))
        wt = np.sum(bad)

        # Find neighbors
        allowed_neighbors = []
        forbidden_neighbors = []

        for i in range(31):
            neighbor = bad.copy()
            neighbor[i] = 1 - neighbor[i]

            if has_circular_ones(neighbor, s):
                forbidden_neighbors.append(i)
            else:
                allowed_neighbors.append(i)

        # For each allowed neighbor, find distance to nearest GOOD Hamming codeword
        neighbor_conflicts = []
        for i in allowed_neighbors:
            neighbor = bad.copy()
            neighbor[i] = 1 - neighbor[i]

            # Find nearest Hamming codeword
            syn = compute_syndrome(neighbor, H)
            if all(s == 0 for s in syn):
                # neighbor is a codeword
                nearest_dist = 0
                nearest_codeword = neighbor
            else:
                # nearest codeword is at distance 1
                error_pos = int(''.join(map(str, syn)), 2) - 1
                nearest_codeword = neighbor.copy()
                if 0 <= error_pos < 31:
                    nearest_codeword[error_pos] = 1 - nearest_codeword[error_pos]
                nearest_dist = 1

            # Is nearest codeword bad?
            nearest_is_bad = has_circular_ones(nearest_codeword, s)

            neighbor_conflicts.append({
                'flip_pos': i,
                'nearest_dist': nearest_dist,
                'nearest_is_bad': nearest_is_bad
            })

        # Count conflicts: allowed neighbors whose nearest GOOD codeword is at distance 2
        conflicts_d2 = sum(1 for nc in neighbor_conflicts
                          if nc['nearest_dist'] == 1 and not nc['nearest_is_bad'])

        conflict_info.append({
            'bad': bad_str,
            'weight': wt,
            'allowed_neighbors': len(allowed_neighbors),
            'forbidden_neighbors': len(forbidden_neighbors),
            'conflicts_d2': conflicts_d2
        })

        total_allowed_uncovered += len(allowed_neighbors)

    print("\nPer-bad-codeword analysis:")
    for info in conflict_info[:10]:
        print(f"  {info['bad'][:20]}... (wt {info['weight']})")
        print(f"    Allowed neighbors: {info['allowed_neighbors']}")
        print(f"    At d=2 from good: {info['conflicts_d2']}")

    if len(conflict_info) > 10:
        print(f"  ... and {len(conflict_info) - 10} more bad codewords")

    # Total analysis
    total_conflicts_d2 = sum(info['conflicts_d2'] for info in conflict_info)
    print(f"\n" + "=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"Total bad codewords: {len(bad_codewords)}")
    print(f"Total allowed vertices needing coverage: {total_allowed_uncovered}")
    print(f"Total at d=2 from good Hamming: {total_conflicts_d2}")

    return conflict_info

def propose_nonlinear_construction(H, s):
    """
    Propose a non-linear construction based on insights from n=15
    """
    print("\n" + "=" * 70)
    print("PROPOSED NON-LINEAR CONSTRUCTION FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    print("""
    STRATEGY: Weight-Preserving Cascade Replacement

    Based on analysis of Λ₁₅(1¹¹) and Λ₁₅(1¹²):
    - SAT solutions share only 2-8% with Hamming
    - But weight distribution is PRESERVED
    - Non-linear structure emerges from cascade replacements

    ALGORITHM:

    1. INITIALIZE:
       C = Hamming(31) code (2^26 codewords)
       Bad = {c ∈ C : c contains circular 1^28}

    2. ITERATIVE REPLACEMENT:
       while Bad ∩ C ≠ ∅:
           Pick b ∈ Bad ∩ C
           For each allowed vertex v ∈ N[b] ∩ Λ₃₁(1²⁸):
               Find replacement center r:
                 - r ∈ Λ₃₁(1²⁸) (no circular 1^28)
                 - d(r, c) ≥ 3 for all c ∈ C \ {b}
                 - d(v, r) ≤ 1
                 - weight(r) chosen to preserve distribution
               If no such r exists:
                   Trigger CASCADE: remove conflicting c' from C
                   Add c' to replacement queue
           Remove b from C
           Add replacement(s) to C

    3. WEIGHT BALANCING:
       After each iteration, check weight distribution
       If imbalanced, prioritize replacements that restore balance

    4. TERMINATION:
       C contains no bad codewords
       All allowed vertices covered exactly once
       Distance ≥ 3 between all centers

    KEY INSIGHT:
    The cascade process naturally produces a non-linear code because:
    - Replacements break XOR closure
    - But weight distribution is a GLOBAL constraint
    - The algorithm finds a local minimum satisfying all constraints

    COMPUTATIONAL CHALLENGES FOR n=31:
    - 2^26 ≈ 67M codewords to track
    - ~2^31 ≈ 2B vertices to cover
    - Need efficient data structures (hashing, spatial indexing)

    FEASIBILITY:
    - Full SAT encoding: ~10^9 variables, likely infeasible
    - Iterative local search: May work with good heuristics
    - Hybrid: Start with Hamming, use SAT for local repairs
    """)

    return True

def estimate_computational_cost():
    """Estimate computational resources needed for n=31"""
    print("\n" + "=" * 70)
    print("COMPUTATIONAL COST ESTIMATE FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    n = 31
    k = 5  # n = 2^k - 1
    s = 28

    # Vertex count
    total_vertices = 2**n

    # Estimate forbidden vertices
    # A vertex is forbidden if it contains circular 1^28
    # This means at most 3 zeros, and they must be in a contiguous arc

    # Count allowed vertices
    # Weight w vertex is allowed if its zeros don't all fit in arc of length n-s = 3

    allowed_estimate = 0
    for w in range(n+1):
        if w <= n - s:  # weight ≤ 3 → all zeros fit in arc of 3 → forbidden
            continue
        # weight > 3: need to count arrangements where zeros DON'T all fit in arc of 3
        # This is complex; estimate
        from math import comb
        total_w = comb(n, w)
        # Rough estimate: most are allowed for w > 5
        allowed_estimate += total_w * 0.9 if w > 5 else total_w * 0.1

    print(f"n = {n}, s = {s}")
    print(f"Total vertices: 2^{n} = {total_vertices:,}")
    print(f"Estimated allowed vertices: ~{int(allowed_estimate):,}")

    # Center count
    center_count = 2**(n-k) - 1  # Remove all-ones
    print(f"Expected centers: 2^{n-k} - 1 = {center_count:,}")

    # Ball size
    avg_ball_size = n + 1  # Each ball covers ~32 vertices on average
    print(f"Average ball size: {avg_ball_size}")

    # Verification: centers * ball_size ≈ allowed_vertices
    print(f"Centers × Ball size: {center_count * avg_ball_size:,}")

    # SAT encoding size
    # Variables: one per (center, vertex) pair
    sat_vars = center_count * allowed_estimate / 1000  # Sparse, not all pairs
    print(f"\nSAT encoding estimate:")
    print(f"  Variables: ~{sat_vars/1e6:.0f}M (sparse encoding)")
    print(f"  Clauses: ~{sat_vars * 10 / 1e6:.0f}M")
    print(f"  Status: LIKELY INFEASIBLE for direct SAT")

    # Iterative approach
    print(f"\nIterative local search estimate:")
    print(f"  Bad codewords: ~{3 + 30} (weight 31 + weight 28)")
    print(f"  Cascade depth: Unknown, but n=15 required ~2000 replacements")
    print(f"  Proportionally for n=31: ~{2000 * (2**26 / 2**11):.0f} replacements")
    print(f"  Status: CHALLENGING but potentially feasible")

    return {
        'n': n,
        'vertices': total_vertices,
        'centers': center_count,
        'sat_feasible': False,
        'iterative_feasible': 'Maybe'
    }

def main():
    print("=" * 70)
    print("NON-LINEAR PERFECT CODE CONSTRUCTION FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    # Generate parity check matrix
    H = generate_hamming_31()
    s = 28

    # Analyze bad codewords
    bad_codewords = find_bad_codewords_exact(H, s)

    # Analyze neighborhood conflicts
    conflict_info = analyze_neighborhood_conflict(H, s)

    # Propose construction
    propose_nonlinear_construction(H, s)

    # Estimate computational cost
    cost = estimate_computational_cost()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    EXISTENCE CONJECTURE:
    A non-linear perfect 1-code C exists for Λ₃₁(1²⁸) with:
    - |C| = 2^26 - 1 = 67,108,863 centers
    - Weight distribution = Hamming weight distribution - {weight 31}
    - All centers avoid circular 1^28
    - C shares only ~2% with Hamming(31)

    PROOF APPROACH:
    1. THEORETICAL: Prove existence via probabilistic method
       - Show density of valid configurations is positive
       - Use Lovász Local Lemma for cascade termination

    2. CONSTRUCTIVE: Develop iterative algorithm
       - Start with Hamming, cascade-replace bad codewords
       - Use weight-preservation as invariant
       - May require sophisticated heuristics

    3. COMPUTATIONAL: Hybrid SAT/local search
       - Decompose problem into tractable subproblems
       - Use SAT for local repair, greedy for global structure

    NEXT STEPS:
    1. Implement iterative replacement with random sampling
    2. Test on intermediate cases (n=15 with different seeds)
    3. Develop theoretical bounds on cascade depth
    4. If successful, attempt n=31 with cluster computing
    """)

if __name__ == "__main__":
    main()
