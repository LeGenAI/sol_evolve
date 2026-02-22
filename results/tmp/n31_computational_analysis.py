#!/usr/bin/env python3
"""
Computational Analysis for n=31, s=28 Perfect Partition

Analyzes why direct SAT approaches fail and what alternatives exist.

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import defaultdict
from fractions import Fraction

# ============================================================================
# Scale Analysis
# ============================================================================

def analyze_n31_scale():
    """Analyze the computational scale of n=31, s=28"""

    print("=" * 70)
    print("COMPUTATIONAL ANALYSIS: Λ₃₁(1²⁸)")
    print("=" * 70)

    n, s = 31, 28

    # Total binary strings
    total_2n = 2 ** n  # 2^31 ≈ 2.1 billion

    # Hamming code parameters
    hamming_size = 2 ** (n - 5)  # 2^26 ≈ 67 million codewords

    print(f"\n1. SCALE PARAMETERS:")
    print(f"   n = {n}, s = {s} (s = n-3)")
    print(f"   |{{0,1}}^n| = 2^{n} = {total_2n:,}")
    print(f"   |Ham(31,26)| = 2^{n-5} = {hamming_size:,}")

    # Forbidden vertex estimate
    # A vertex is forbidden iff it has ≥28 consecutive 1s (circularly)
    # This happens when max gap between zeros ≥ 28

    # For weight w, need to count binary strings with ≥28 consecutive 1s
    # Exact computation is complex, but we can estimate:

    # Upper bound: vertices with weight ≥ 28 can have forbidden patterns
    # C(31,28) + C(31,29) + C(31,30) + C(31,31)
    from math import comb
    high_weight_count = sum(comb(31, w) for w in range(28, 32))

    print(f"\n2. FORBIDDEN REGION ESTIMATE:")
    print(f"   Vertices with weight ≥ 28: {high_weight_count:,}")
    print(f"   Max possible forbidden ≈ {high_weight_count:,}")
    print(f"   (Actual forbidden ≤ this, due to gap constraint)")

    # Expected vertices in Λ₃₁(1²⁸)
    expected_vertices = total_2n - high_weight_count
    print(f"\n3. EXPECTED |V(Λ₃₁(1²⁸))|:")
    print(f"   Lower bound: 2^31 - {high_weight_count:,} = {expected_vertices:,}")

    # Expected centers
    # For uniform ball size n+1 = 32:
    expected_centers_uniform = expected_vertices / 32
    print(f"\n4. EXPECTED CENTER COUNT:")
    print(f"   If all balls have size 32: |C| ≈ {expected_centers_uniform:,.0f}")
    print(f"   Hamming code has: {hamming_size:,} codewords")
    print(f"   Ratio: Hamming / Expected ≈ {hamming_size / expected_centers_uniform:.4f}")

    # SAT encoding size
    print(f"\n5. SAT ENCODING COMPLEXITY:")

    # Variables: one per valid center (≈ expected_vertices if all can be centers)
    # But we filter by ball size, so fewer
    print(f"   Center variables: ≈ {expected_vertices:,} (worst case)")

    # Auxiliary variables for at-most-one encoding
    # Sequential counter uses n-1 aux vars per constraint
    # Covering constraints: ≈ expected_vertices constraints
    # Average constraint size: ≈ 32 (ball size)
    aux_per_constraint = 31  # n-1 for sequential counter
    total_aux = expected_vertices * aux_per_constraint

    print(f"   Auxiliary variables: ≈ {total_aux:,}")
    print(f"   Total variables: ≈ {expected_vertices + total_aux:,}")

    # Clauses
    # Covering: ≈ expected_vertices * ball_size clauses
    covering_clauses = expected_vertices * 32
    # Packing: for each center pair at distance < 3
    # This is the dominant factor
    # Each center has ≈ n + C(n,2) neighbors at distance ≤ 2
    neighbors_per_center = n + n * (n - 1) // 2
    packing_clauses = expected_centers_uniform * neighbors_per_center / 2

    print(f"   Covering clauses: ≈ {covering_clauses:,}")
    print(f"   Packing clauses: ≈ {packing_clauses:,.0f}")
    print(f"   Total clauses: ≈ {covering_clauses + packing_clauses:,.0f}")

    # Memory estimate
    bytes_per_variable = 8  # rough estimate
    bytes_per_clause = 16
    memory_vars = (expected_vertices + total_aux) * bytes_per_variable
    memory_clauses = (covering_clauses + packing_clauses) * bytes_per_clause

    print(f"\n6. MEMORY REQUIREMENTS:")
    print(f"   Variable storage: ≈ {memory_vars / 1e9:.1f} GB")
    print(f"   Clause storage: ≈ {memory_clauses / 1e9:.1f} GB")
    print(f"   Total estimated: ≈ {(memory_vars + memory_clauses) / 1e9:.1f} GB")

    return expected_vertices, hamming_size


def analyze_hamming_repair_approach():
    """Analyze the Hamming code repair approach"""

    print("\n" + "=" * 70)
    print("HAMMING REPAIR APPROACH ANALYSIS")
    print("=" * 70)

    n, s = 31, 28

    # Hamming code Ham(31, 26)
    hamming_size = 2 ** 26

    print(f"\n1. HAMMING CODE STRUCTURE:")
    print(f"   Ham(31, 26): {hamming_size:,} codewords")
    print(f"   Minimum distance: 3")
    print(f"   Covering radius: 1")
    print(f"   All codewords have balls of size 32 in F_2^31")

    # Bad codewords analysis
    # A codeword is "bad" if it has ≥28 consecutive 1s
    # This is a small fraction

    # For a random 31-bit string:
    # P(≥28 consecutive 1s) ≈ n * (1/2)^28 ≈ 31 / 2^28 ≈ 1.15e-7

    prob_bad = 31 / (2 ** 28)
    expected_bad = hamming_size * prob_bad

    print(f"\n2. BAD CODEWORD ESTIMATE:")
    print(f"   P(codeword is bad) ≈ n / 2^{s} = {prob_bad:.2e}")
    print(f"   Expected bad codewords ≈ {expected_bad:.1f}")

    # But this is just an estimate - actual structure matters
    # The Hamming code is linear, so bad codewords depend on generator matrix

    print(f"\n3. REPAIR CHALLENGE:")
    print(f"   Each bad codeword covers 32 vertices")
    print(f"   Dropping k bad codewords leaves 32k vertices uncovered")
    print(f"   Need replacement centers at distance ≥3 from all remaining")

    # Local SAT approach
    print(f"\n4. LOCAL SAT RESULTS:")
    print(f"   drop_radius=2: No droppable codewords (all essential)")
    print(f"   drop_radius=3, max_drops=50: UNSAT in 0.01s")
    print(f"   Interpretation: Local repair within radius 3 is insufficient")

    # Why local repair fails
    print(f"\n5. WHY LOCAL REPAIR FAILS:")
    print(f"   - Hamming code is rigid: pairwise distance exactly 3")
    print(f"   - Dropping any codeword breaks perfect covering")
    print(f"   - Replacement must come from non-Hamming vertices")
    print(f"   - Non-Hamming replacements conflict with many existing centers")
    print(f"   - Global restructuring needed, not local repair")


def propose_alternatives():
    """Propose alternative approaches for n=31"""

    print("\n" + "=" * 70)
    print("ALTERNATIVE APPROACHES")
    print("=" * 70)

    print("""
1. CONSTRUCTION FROM SMALLER SOLUTIONS:

   If we have a perfect partition for Λ₁₅(1¹²), can we construct one for Λ₃₁(1²⁸)?

   Observation: n=15 and n=31 are both 2^k - 1 (k=4 and k=5)

   Potential approach: Use recursive construction
   - Λ₃₁(1²⁸) might embed multiple copies of Λ₁₅(1¹²) structure
   - Centers in smaller graph might lift to centers in larger graph

2. PROBABILISTIC EXISTENCE PROOF:

   Show that random selection of ≈2^26 vertices with distance ≥3
   covers all of Λ₃₁(1²⁸) with positive probability.

   Challenge: Proving covering is difficult
   - Need to show every vertex is within distance 1 of some center
   - Probabilistic arguments work for packing, harder for covering

3. ALGEBRAIC CONSTRUCTION:

   Use the algebraic structure of Hamming codes:
   - Ham(31, 26) is the null space of a 5×31 parity check matrix
   - Perfect partition centers form a different algebraic structure
   - Find a modification that avoids forbidden patterns

   Idea: Coset shifting
   - Instead of Ham(31, 26), use Ham(31, 26) + v for carefully chosen v
   - Choose v to avoid forbidden patterns while maintaining covering

4. SAT WITH SYMMETRY BREAKING:

   Use symmetry breaking to reduce search space:
   - Fix some centers (e.g., 0 is always a center)
   - Use cyclic symmetry of Λₙ(1ˢ)
   - Add lexicographic ordering constraints

5. INCREMENTAL SAT:

   Build solution incrementally:
   - Start with partial solution (e.g., low-weight centers)
   - Add centers one by one, checking feasibility
   - Backtrack when stuck, guided by conflict analysis
    """)


def main():
    expected_vertices, hamming_size = analyze_n31_scale()
    analyze_hamming_repair_approach()
    propose_alternatives()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print(f"""
The n=31, s=28 case presents significant computational challenges:

1. SCALE: ~2 billion vertices, ~67 million potential centers
2. MEMORY: Estimated 100+ GB for full SAT encoding
3. LOCAL REPAIR: Proven insufficient (UNSAT with reasonable constraints)

RECOMMENDED PATH FORWARD:

A. For EXISTENCE proof (non-constructive):
   - Probabilistic argument or counting lemma
   - Algebraic structure theorem

B. For CONSTRUCTIVE proof:
   - Recursive construction from n=15 solution
   - Coset-based modification of Hamming code
   - Distributed/parallel SAT solving

C. For PRACTICAL computation:
   - Decompose into smaller subproblems
   - Use symmetry and structure extensively
   - Consider quantum-assisted optimization

The fact that n=7 and n=15 cases admit non-Hamming perfect partitions
suggests that n=31 might also have such solutions, but finding them
requires more sophisticated techniques than local repair.
    """)


if __name__ == "__main__":
    main()
