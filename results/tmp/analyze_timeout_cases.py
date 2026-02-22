#!/usr/bin/env python3
"""
Analyze why Λ₁₅(1¹⁰) and Λ₁₅(1⁹) timed out.
Look for structural differences from successful cases (s=11, s=12).

Author: Jae-Hyun Baek (Claude Code as "human" analyst)
Date: 2025-11-25
"""

from collections import Counter
import numpy as np

def has_circular_consecutive_ones(n, bits, s):
    doubled = (n << bits) | n
    mask = (1 << s) - 1
    for i in range(bits):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming_weight(x):
    return bin(x).count('1')

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def generate_lambda_vertices(n, s):
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball_size(center, n, vertices_set):
    ball = 1  # center itself
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball += 1
    return ball

def analyze_lambda(n, s):
    """Comprehensive analysis of Λₙ(1ˢ)."""
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)
    V = len(vertices)

    print(f"\n{'='*60}")
    print(f"Analysis of Λ_{n}(1^{s})")
    print(f"{'='*60}")

    print(f"\n1. Basic Statistics:")
    print(f"   |V| = {V}")

    # Degree distribution
    degrees = [sum(1 for i in range(n) if (v ^ (1 << i)) in vertices_set) for v in vertices]
    deg_dist = Counter(degrees)
    print(f"   Degree distribution: {dict(sorted(deg_dist.items()))}")

    # Ball size distribution
    ball_sizes = [get_ball_size(v, n, vertices_set) for v in vertices]
    ball_dist = Counter(ball_sizes)
    print(f"   Ball size distribution: {dict(sorted(ball_dist.items()))}")

    avg_ball = sum(ball_sizes) / len(ball_sizes)
    print(f"   Average ball size: {avg_ball:.2f}")

    # Modularity analysis
    print(f"\n2. Modularity Analysis:")
    for ball_size in sorted(ball_dist.keys()):
        print(f"   |V| mod {ball_size} = {V % ball_size}")

    # Estimate centers needed
    min_centers = V // max(ball_sizes)
    max_centers = V // min(ball_sizes)
    print(f"\n3. Center Count Estimates:")
    print(f"   Min centers (if all balls size {max(ball_sizes)}): {min_centers}")
    print(f"   Max centers (if all balls size {min(ball_sizes)}): {max_centers}")
    print(f"   Likely range: [{min_centers}, {max_centers}]")

    # Weight distribution
    weights = [hamming_weight(v) for v in vertices]
    weight_dist = Counter(weights)
    print(f"\n4. Vertex Weight Distribution:")
    for w in sorted(weight_dist.keys()):
        print(f"   w={w}: {weight_dist[w]} vertices")

    # Check low-weight vertex coverage
    low_weight_vertices = [v for v in vertices if hamming_weight(v) <= 2]
    print(f"\n5. Low-Weight Vertex Analysis:")
    print(f"   Vertices with weight <= 2: {len(low_weight_vertices)}")

    for v in low_weight_vertices[:10]:  # First 10
        ball_size = get_ball_size(v, n, vertices_set)
        # Count how many potential centers cover this vertex
        covers = sum(1 for c in vertices if hamming_distance(c, v) <= 1)
        print(f"   {bin(v)[2:].zfill(n)}: ball_size={ball_size}, covered_by={covers} vertices")

    # Check if any vertex has limited coverage options
    min_coverage_options = float('inf')
    hardest_vertex = None
    for v in vertices:
        options = sum(1 for c in vertices if hamming_distance(c, v) <= 1)
        if options < min_coverage_options:
            min_coverage_options = options
            hardest_vertex = v

    print(f"\n6. Coverage Bottleneck Analysis:")
    print(f"   Vertex with fewest coverage options: {bin(hardest_vertex)[2:].zfill(n)}")
    print(f"   Coverage options: {min_coverage_options}")

    # Distribution of coverage options
    coverage_counts = [sum(1 for c in vertices if hamming_distance(c, v) <= 1) for v in vertices]
    coverage_dist = Counter(coverage_counts)
    print(f"   Coverage options distribution: {dict(sorted(coverage_dist.items()))}")

    # SAT difficulty indicator
    print(f"\n7. SAT Difficulty Indicators:")

    # More uniform ball sizes = easier
    ball_variance = np.var(ball_sizes)
    print(f"   Ball size variance: {ball_variance:.2f}")

    # Check constraint density
    # Covering: one clause per vertex with |ball_membership| literals
    # Packing: one clause per close pair
    close_pairs = 0
    for i, v in enumerate(vertices):
        for u in vertices[i+1:]:
            if hamming_distance(v, u) < 3:
                close_pairs += 1
    print(f"   Close pairs (distance < 3): {close_pairs}")

    clause_estimate = V + close_pairs  # Rough estimate
    var_estimate = V
    ratio = clause_estimate / var_estimate
    print(f"   Clause/Variable ratio estimate: {ratio:.2f}")

    return {
        'n': n, 's': s, 'V': V,
        'deg_dist': dict(deg_dist),
        'ball_dist': dict(ball_dist),
        'avg_ball': avg_ball,
        'close_pairs': close_pairs,
        'ball_variance': ball_variance
    }

def main():
    print("="*60)
    print("Timeout Case Analysis for Λ₁₅(1ˢ)")
    print("="*60)

    # Analyze all relevant cases
    cases = [
        (15, 12),  # SAT - reference
        (15, 11),  # SAT - reference
        (15, 10),  # TIMEOUT
        (15, 9),   # TIMEOUT
        (15, 8),   # Unknown
        (7, 4),    # SAT - reference
        (7, 3),    # UNSAT - reference
    ]

    results = []
    for n, s in cases:
        result = analyze_lambda(n, s)
        results.append(result)

    # Comparative summary
    print("\n" + "="*60)
    print("COMPARATIVE SUMMARY")
    print("="*60)

    print(f"\n{'Case':<15} | {'|V|':>8} | {'Avg Ball':>10} | {'Ball Var':>10} | {'Close Pairs':>12}")
    print("-"*70)

    for r in results:
        print(f"Λ_{r['n']}(1^{r['s']:<2})" + f" | {r['V']:>8} | {r['avg_ball']:>10.2f} | {r['ball_variance']:>10.2f} | {r['close_pairs']:>12}")

    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    # Compare SAT vs UNSAT patterns
    sat_cases = [(15,12), (15,11), (7,4)]
    for r in results:
        key = (r['n'], r['s'])
        status = "SAT" if key in sat_cases else "UNSAT" if key == (7,3) else "?"
        mod_check = r['V'] % max(r['ball_dist'].keys())
        print(f"\nΛ_{r['n']}(1^{r['s']}): {status}")
        print(f"  |V| = {r['V']}, mod max_ball = {mod_check}")
        print(f"  Ball sizes: {sorted(r['ball_dist'].keys())}")

if __name__ == "__main__":
    main()
