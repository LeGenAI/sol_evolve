#!/usr/bin/env python3
"""
Optimized Greedy Algorithm for Lambda_15(1^12) Perfect Partition
"""

import numpy as np
from collections import defaultdict
import time

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

def get_ball(center, n, vertices_set):
    """Get closed ball of radius 1 around center."""
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

def analyze_non_regular_vertices(vertices, n):
    """Analyze vertices with degree < n (non-regular)."""
    vertices_set = set(vertices)

    non_regular = []
    for v in vertices:
        deg = sum(1 for i in range(n) if (v ^ (1 << i)) in vertices_set)
        if deg < n:
            non_regular.append((v, deg))

    print(f"\n{'='*60}")
    print(f"Non-Regular Vertices Analysis")
    print(f"{'='*60}")
    print(f"Total non-regular: {len(non_regular)}")

    # Group by degree
    by_degree = defaultdict(list)
    for v, deg in non_regular:
        by_degree[deg].append(v)

    for deg in sorted(by_degree.keys()):
        vlist = by_degree[deg]
        print(f"\nDegree {deg} ({len(vlist)} vertices, ball size {deg+1}):")

        # Analyze weight distribution
        weights = [hamming_weight(v) for v in vlist]
        weight_counts = defaultdict(int)
        for w in weights:
            weight_counts[w] += 1
        print(f"  Weight distribution: {dict(sorted(weight_counts.items()))}")

        # Show first few
        for v in vlist[:5]:
            binary = bin(v)[2:].zfill(n)
            print(f"    {binary} (weight {hamming_weight(v)})")
        if len(vlist) > 5:
            print(f"    ... and {len(vlist)-5} more")

    return non_regular

def optimized_greedy(vertices, n, min_dist=3):
    """
    Greedy with optimization:
    1. Prefer low-degree vertices as centers (smaller balls → more centers needed → better coverage control)
    2. Use efficient data structures
    """
    vertices_set = set(vertices)

    # Precompute degrees and sort by degree (ascending)
    print("\nPrecomputing degrees...")
    vertex_degrees = {}
    for v in vertices:
        deg = sum(1 for i in range(n) if (v ^ (1 << i)) in vertices_set)
        vertex_degrees[v] = deg

    # Sort vertices by degree (prefer low degree as centers)
    sorted_vertices = sorted(vertices, key=lambda v: vertex_degrees[v])

    print("Running optimized greedy...")
    centers = []
    covered = set()

    # Track which vertices are "blocked" (within distance < min_dist of a center)
    blocked = set()

    t0 = time.time()

    for v in sorted_vertices:
        if v in covered:
            continue
        if v in blocked:
            continue

        # Check if v is valid (far enough from all centers)
        # Since we track blocked vertices, no need to check all centers

        # Add v as center
        centers.append(v)

        # Cover all vertices in ball
        ball = get_ball(v, n, vertices_set)
        covered.update(ball)

        # Block vertices within distance < min_dist
        for u in vertices_set - blocked:
            if hamming_distance(v, u) < min_dist:
                blocked.add(u)

        if len(centers) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {len(centers)} centers, {len(covered)}/{len(vertices)} covered, {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\nGreedy completed in {total_time:.1f}s")

    return centers, covered, blocked

def verify_partition(centers, vertices, n):
    """Verify the partition properties."""
    vertices_set = set(vertices)

    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    # Check center distances
    min_dist = float('inf')
    violations = 0
    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            d = hamming_distance(c1, c2)
            if d < 3:
                violations += 1
            if d < min_dist:
                min_dist = d

    print(f"Center count: {len(centers)}")
    print(f"Min pairwise distance: {min_dist}")
    print(f"Distance violations (< 3): {violations}")

    # Check coverage
    covered = set()
    ball_sizes = []
    overlaps = 0

    for c in centers:
        ball = get_ball(c, n, vertices_set)
        ball_sizes.append(len(ball))
        for v in ball:
            if v in covered:
                overlaps += 1
            covered.add(v)

    uncovered = vertices_set - covered
    print(f"Covered: {len(covered)}/{len(vertices)}")
    print(f"Uncovered: {len(uncovered)}")
    print(f"Overlaps: {overlaps}")

    # Ball size distribution
    from collections import Counter
    ball_dist = Counter(ball_sizes)
    print(f"Ball sizes: {dict(sorted(ball_dist.items()))}")

    is_perfect = (len(uncovered) == 0) and (overlaps == 0) and (violations == 0)
    print(f"\n{'✓ PERFECT PARTITION!' if is_perfect else '✗ Not a perfect partition'}")

    if not is_perfect and len(uncovered) > 0:
        print(f"\nUncovered vertices:")
        for v in list(uncovered)[:10]:
            print(f"  {bin(v)[2:].zfill(n)}")
        if len(uncovered) > 10:
            print(f"  ... and {len(uncovered)-10} more")

    return is_perfect, uncovered

def main():
    n, s = 15, 12

    print(f"{'='*60}")
    print(f"Lambda_{n}(1^{s}) Perfect Partition Search")
    print(f"{'='*60}")

    # Generate vertices
    print("\nGenerating vertices...")
    vertices = generate_lambda_vertices(n, s)
    print(f"|V| = {len(vertices)}")

    # Analyze non-regular vertices
    non_regular = analyze_non_regular_vertices(vertices, n)

    # Run optimized greedy
    centers, covered, blocked = optimized_greedy(vertices, n, min_dist=3)

    # Verify
    is_perfect, uncovered = verify_partition(centers, vertices, n)

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    np.save('greedy_centers.npy', np.array(centers))
    if len(uncovered) > 0:
        np.save('uncovered.npy', np.array(list(uncovered)))
    print(f"Centers saved to greedy_centers.npy")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Vertices: {len(vertices)}")
    print(f"Centers found: {len(centers)}")
    print(f"Covered: {len(covered)}")
    print(f"Uncovered: {len(uncovered)}")
    print(f"Perfect partition: {'YES' if is_perfect else 'NO'}")

    if not is_perfect:
        print(f"\nNext steps:")
        print(f"1. Analyze uncovered vertices")
        print(f"2. Try local search to improve greedy solution")
        print(f"3. Use SAT/ILP for exact solution")

if __name__ == "__main__":
    main()
