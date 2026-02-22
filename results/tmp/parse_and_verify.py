#!/usr/bin/env python3
"""
Parse SAT solution and verify the perfect partition
"""

import numpy as np
from collections import Counter
import subprocess

def has_circular_consecutive_ones(n, bits, s):
    doubled = (n << bits) | n
    mask = (1 << s) - 1
    for i in range(bits):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def hamming_weight(x):
    return bin(x).count('1')

def generate_lambda_vertices(n, s):
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball(center, n, vertices_set):
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

def main():
    n, s = 15, 12
    num_vertices = 32707

    print(f"{'='*60}")
    print(f"Parsing and Verifying Lambda_{n}(1^{s}) Perfect Partition")
    print(f"{'='*60}\n")

    # Generate vertices
    print("Generating vertices...")
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)
    print(f"|V| = {len(vertices)}")

    # Run solver and capture output
    print("\nRunning SAT solver...")
    result = subprocess.run(
        ["/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical", "lambda_15_12.cnf"],
        capture_output=True,
        text=True,
        timeout=300
    )

    # Parse solution from stdout
    print("Parsing solution...")
    centers_indices = []

    for line in result.stdout.split('\n'):
        if line.startswith('v '):
            parts = line.split()[1:]
            for p in parts:
                if p == '0':
                    continue
                var = int(p)
                # Only consider original vertex variables (1 to num_vertices)
                if var > 0 and var <= num_vertices:
                    centers_indices.append(var - 1)  # Convert to 0-indexed

    # Convert indices to actual vertices
    centers = [vertices[i] for i in centers_indices]
    print(f"Centers found: {len(centers)}")

    # Verification
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    # Check pairwise distances
    print("\n1. Checking pairwise distances...")
    min_dist = float('inf')
    violations = 0
    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            d = hamming_distance(c1, c2)
            if d < 3:
                violations += 1
            if d < min_dist:
                min_dist = d

    print(f"   Min pairwise distance: {min_dist}")
    print(f"   Distance violations (< 3): {violations}")

    # Check coverage
    print("\n2. Checking coverage...")
    covered = set()
    overlaps = 0
    ball_sizes = []

    for c in centers:
        ball = get_ball(c, n, vertices_set)
        ball_sizes.append(len(ball))
        for v in ball:
            if v in covered:
                overlaps += 1
            covered.add(v)

    uncovered = vertices_set - covered
    print(f"   Covered: {len(covered)}/{len(vertices)}")
    print(f"   Uncovered: {len(uncovered)}")
    print(f"   Overlaps: {overlaps}")

    # Ball size distribution
    print("\n3. Ball size distribution:")
    ball_dist = Counter(ball_sizes)
    print(f"   {dict(sorted(ball_dist.items()))}")

    # Total ball coverage
    total_ball = sum(ball_sizes)
    print(f"\n4. Coverage accounting:")
    print(f"   Sum of ball sizes: {total_ball}")
    print(f"   Expected (|V|): {len(vertices)}")
    print(f"   Match: {total_ball == len(vertices)}")

    # Weight distribution of centers
    print("\n5. Weight distribution of centers:")
    weights = [hamming_weight(c) for c in centers]
    weight_dist = Counter(weights)
    print(f"   {dict(sorted(weight_dist.items()))}")

    # Final verdict
    is_perfect = (len(uncovered) == 0) and (overlaps == 0) and (min_dist >= 3)

    print(f"\n{'='*60}")
    if is_perfect:
        print("✓ PERFECT PARTITION VERIFIED!")
        print(f"{'='*60}")
        print(f"\nLambda_{n}(1^{s}) admits a perfect partition with {len(centers)} centers.")

        # Save results
        print("\nSaving results...")
        np.save('lambda_15_12_centers.npy', np.array(centers))

        with open('lambda_15_12_centers.txt', 'w') as f:
            f.write(f"Perfect Partition of Lambda_{n}(1^{s})\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total vertices: {len(vertices)}\n")
            f.write(f"Number of centers: {len(centers)}\n")
            f.write(f"Ball sizes: {dict(sorted(ball_dist.items()))}\n\n")
            f.write("Centers (binary representation):\n")
            for i, c in enumerate(sorted(centers)):
                ball = get_ball(c, n, vertices_set)
                f.write(f"{i+1:4d}. {bin(c)[2:].zfill(n)}  (ball size: {len(ball)})\n")

        print("   Saved to lambda_15_12_centers.npy")
        print("   Saved to lambda_15_12_centers.txt")

        # Summary table for paper
        print(f"\n{'='*60}")
        print("SUMMARY FOR PAPER")
        print(f"{'='*60}")
        print(f"""
\\begin{{theorem}}
The Generalized Lucas cube $\\Lambda_{{15}}(1^{{12}})$ admits a perfect partition
with {len(centers)} centers.
\\end{{theorem}}

Key statistics:
- $|V(\\Lambda_{{15}}(1^{{12}}))| = {len(vertices)}$
- Number of centers: {len(centers)}
- Ball sizes: {dict(sorted(ball_dist.items()))}
- Min pairwise distance: {min_dist}
""")

    else:
        print("✗ NOT A PERFECT PARTITION")
        print(f"{'='*60}")
        print(f"Issues found:")
        if len(uncovered) > 0:
            print(f"  - {len(uncovered)} uncovered vertices")
        if overlaps > 0:
            print(f"  - {overlaps} overlapping vertices")
        if min_dist < 3:
            print(f"  - Min distance {min_dist} < 3")

if __name__ == "__main__":
    main()
