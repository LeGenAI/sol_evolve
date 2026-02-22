#!/usr/bin/env python3
"""
Quick Analysis of Lambda_15(1^12) - optimized for speed
"""

import numpy as np
from collections import Counter
import time

def has_circular_consecutive_ones(n, bits, s):
    """Check if n-bit number has s consecutive 1s in circular fashion."""
    doubled = (n << bits) | n
    mask = (1 << s) - 1
    for i in range(bits):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def generate_lambda_vertices(n, s):
    """Generate all vertices of Lambda_n(1^s)."""
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def hamming_weight(x):
    return bin(x).count('1')

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def main():
    n, s = 15, 12

    print(f"{'='*60}")
    print(f"Quick Analysis of Lambda_{n}(1^{s})")
    print(f"{'='*60}\n")

    # Generate vertices
    print("1. Generating vertices...")
    t0 = time.time()
    vertices = generate_lambda_vertices(n, s)
    V = len(vertices)
    print(f"   |V| = {V} (generated in {time.time()-t0:.2f}s)")

    # Divisibility check
    print(f"\n2. Divisibility Analysis")
    print(f"   |V| mod 16 = {V % 16}")
    if V % 16 == 0:
        print(f"   ✓ Perfect partition possible: {V // 16} centers")
    else:
        print(f"   ✗ No exact partition with ball size 16")
        remainder = V % 16
        print(f"   Need {V // 16} balls of 16 + handle {remainder} extra")

    # Degree analysis (sample-based for speed)
    print(f"\n3. Degree Distribution (full analysis)")
    vertices_set = set(vertices)
    t0 = time.time()

    degrees = []
    for v in vertices:
        deg = 0
        for i in range(n):
            if (v ^ (1 << i)) in vertices_set:
                deg += 1
        degrees.append(deg)

    print(f"   Computed in {time.time()-t0:.2f}s")
    degree_counts = Counter(degrees)
    print(f"   Distribution: {dict(sorted(degree_counts.items()))}")

    is_regular = len(degree_counts) == 1
    print(f"   Is {n}-regular: {is_regular}")

    if not is_regular:
        max_deg = max(degrees)
        min_deg = min(degrees)
        print(f"   Min degree: {min_deg}, Max degree: {max_deg}")

        # Find non-max degree vertices
        non_max_count = sum(1 for d in degrees if d < max_deg)
        print(f"   Vertices with degree < {max_deg}: {non_max_count}")

        # Ball sizes depend on degrees
        min_ball = min_deg + 1
        max_ball = max_deg + 1
        print(f"   Ball sizes range: [{min_ball}, {max_ball}]")

    # Weight distribution
    print(f"\n4. Weight Distribution")
    weights = [hamming_weight(v) for v in vertices]
    weight_counts = Counter(weights)
    print(f"   {dict(sorted(weight_counts.items()))}")

    # Theoretical bounds
    print(f"\n5. Theoretical Bounds for Perfect Partition")
    min_ball_size = min(degrees) + 1 if degrees else 1
    max_ball_size = max(degrees) + 1 if degrees else n + 1
    lower_bound = (V + max_ball_size - 1) // max_ball_size
    upper_bound = (V + min_ball_size - 1) // min_ball_size
    print(f"   Lower bound on centers: ceil({V}/{max_ball_size}) = {lower_bound}")
    print(f"   Upper bound on centers: ceil({V}/{min_ball_size}) = {upper_bound}")

    # Save data
    print(f"\n6. Saving data...")
    np.save('lambda_15_12_vertices.npy', np.array(vertices))
    np.save('lambda_15_12_degrees.npy', np.array(degrees))
    print(f"   Saved vertices and degrees to results/tmp/")

    # Strategy
    print(f"\n{'='*60}")
    print("STRATEGY RECOMMENDATION")
    print(f"{'='*60}")
    print(f"""
Problem Scale:
- {V:,} vertices
- Need ~{lower_bound:,} to {upper_bound:,} centers
- SAT variables: ~{V} (one per vertex for center selection)
- SAT clauses: ~{V * (V-1) // 2} distance constraints + {V} covering constraints

Approach Options:
1. GREEDY + LOCAL SEARCH (Fast feasibility check)
   - O(V²) time but practical
   - May find good solution or prove it's hard

2. ILP with Gurobi/CPLEX (Optimal with proof)
   - Can handle ~100K binary variables
   - Best for proving optimality

3. SAT with incremental solving (Exact)
   - Start with k={lower_bound}, increase if UNSAT
   - Use symmetry breaking and learned clauses

4. HYBRID: Greedy → SAT repair
   - Use greedy solution to warm-start SAT
   - Focus SAT on uncovered vertices only

Recommended: Start with fast greedy, then ILP/SAT for verification
""")

if __name__ == "__main__":
    main()
