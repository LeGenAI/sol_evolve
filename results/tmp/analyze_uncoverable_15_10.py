#!/usr/bin/env python3
"""
Analyze the uncoverable vertices in Lambda_15(1^10).

If there are vertices that cannot be covered by ANY valid center,
then a perfect partition is IMPOSSIBLE.

This would be a significant theoretical result!

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter

def has_circular_consecutive_ones(n_val, bits, s):
    """Check if n_val has s consecutive 1s in circular bit representation."""
    doubled = (n_val << bits) | n_val
    mask = (1 << s) - 1
    for i in range(bits):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def generate_lambda_vertices(n, s):
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball(center, n, vertices_set):
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

def to_binary(v, n):
    return format(v, f'0{n}b')

def main():
    print("=" * 70)
    print("Analysis of Uncoverable Vertices in Lambda_15(1^10)")
    print("=" * 70)

    n, s = 15, 10

    # Generate Lambda_15(1^10)
    print("\nGenerating Lambda_15(1^10) vertices...")
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)
    print(f"|V(Lambda_15(1^10))| = {len(vertices)}")

    # For each vertex v, check if there exists ANY valid center c such that v in B(c)
    # A valid center must:
    # 1. Be in Lambda_15(1^10) (no 10+ consecutive 1s)
    # 2. Have v in its ball (v = c or d(v,c) = 1)

    print("\nFinding uncoverable vertices...")
    uncoverable = []

    for v in vertices:
        can_be_covered = False

        # Check if v can be a center itself
        # (center covers itself)
        if not has_circular_consecutive_ones(v, n, s):
            can_be_covered = True
        else:
            # Check neighbors of v (distance 1)
            for i in range(n):
                neighbor = v ^ (1 << i)
                # neighbor is a potential center if:
                # 1. neighbor is in Lambda_15(1^10)
                # 2. v would be in neighbor's ball
                if neighbor in vertices_set:
                    # neighbor is valid and v is in B(neighbor)
                    can_be_covered = True
                    break

        if not can_be_covered:
            uncoverable.append(v)

    print(f"\nUncoverable vertices: {len(uncoverable)}")

    if uncoverable:
        print(f"\n{'='*70}")
        print("ANALYSIS OF UNCOVERABLE VERTICES")
        print("=" * 70)

        # These are vertices that:
        # 1. Are in Lambda_15(1^10) (have < 10 consecutive 1s)
        # 2. Cannot be a center themselves (would need to be - but they're in the graph)
        # 3. All their neighbors are NOT in Lambda_15(1^10) (have >= 10 consecutive 1s)

        print("\nThis should be IMPOSSIBLE because:")
        print("  - If v is in Lambda_15(1^10), it can always be its own center")
        print("  - v is automatically covered by B(v)")

        print("\nLet me re-verify...")

        # Re-check: for each vertex, can IT cover itself?
        really_uncoverable = []
        for v in vertices:
            # v is in Lambda_15(1^10), so v is a valid center
            # B(v) always contains v
            # So v is always covered by choosing v as a center
            # UNLESS there's some other constraint...

            # Actually, the issue is: v might NOT be a valid center
            # if it violates the s=10 constraint

            # Wait - if v is in Lambda_15(1^10), by definition it has < 10 consecutive 1s
            # So v IS a valid center

            # The "uncoverable" must be something else...
            ball = get_ball(v, n, vertices_set)
            ball_size = len(ball)

            # Find all potential centers that can cover v
            potential_centers = []
            for c in vertices_set:
                c_ball = get_ball(c, n, vertices_set)
                if v in c_ball:
                    potential_centers.append(c)

            if not potential_centers:
                really_uncoverable.append(v)
                print(f"  {to_binary(v, n)} - ball size {ball_size}, no potential centers!")

        print(f"\nRe-verified uncoverable: {len(really_uncoverable)}")

        if len(really_uncoverable) == 0:
            print("\nGOOD: Every vertex CAN be covered by some center.")
            print("The greedy approach failed due to packing conflicts, not coverability.")

    else:
        print("\nEvery vertex can be covered - perfect partition MAY exist.")

    # Additional analysis: check the constraint more carefully
    print("\n" + "=" * 70)
    print("CONSTRAINT ANALYSIS")
    print("=" * 70)

    # For a perfect partition, we need:
    # 1. Every vertex covered exactly once
    # 2. Centers at pairwise distance >= 3

    # The number of centers needed:
    # Sum of 1/|B(c)| over all vertices = number of centers (if perfect)

    print("\nExpected number of centers:")
    ball_sizes = {}
    for v in vertices:
        ball = get_ball(v, n, vertices_set)
        ball_sizes[v] = len(ball)

    expected_centers = sum(1.0 / ball_sizes[v] for v in vertices)
    print(f"  Sum of 1/|B(v)| = {expected_centers:.2f}")

    # Distribution of ball sizes
    size_dist = Counter(ball_sizes.values())
    print(f"\nBall size distribution:")
    for size in sorted(size_dist.keys()):
        count = size_dist[size]
        contribution = count * (1.0 / size)
        print(f"  Size {size}: {count} vertices, contributes {contribution:.2f} centers")

    # Check: is 32527 divisible by any common ball size?
    print(f"\n|V| = {len(vertices)}")
    print(f"If all balls had size 16: would need {len(vertices)/16:.2f} centers")
    print(f"Actual expected: {expected_centers:.2f}")

    # The key question: does there exist a selection of ~2033 vertices
    # such that their balls partition V?

    # Let's check: what's the min centers needed?
    print(f"\nTheoretical bounds:")
    print(f"  Lower bound (all size 16): {len(vertices) / 16:.2f}")
    print(f"  Upper bound (all size 14): {len(vertices) / 14:.2f}")
    print(f"  Expected (weighted): {expected_centers:.2f}")

    # Check if the problem is related to packing
    print("\n" + "=" * 70)
    print("PACKING ANALYSIS")
    print("=" * 70)

    # How many vertices can be at distance >= 3 from each other?
    # This is the independence number of the graph where edges connect
    # vertices at distance < 3

    # A quick estimate: random sampling
    print("\nEstimating maximum independent set (centers at distance >= 3)...")

    import random
    random.seed(42)

    best_count = 0
    for trial in range(10):
        selected = []
        candidates = list(vertices)
        random.shuffle(candidates)

        for c in candidates:
            ok = True
            for s in selected:
                if hamming_distance(c, s) < 3:
                    ok = False
                    break
            if ok:
                selected.append(c)

        if len(selected) > best_count:
            best_count = len(selected)

    print(f"  Random greedy: ~{best_count} independent centers")
    print(f"  Need: ~{expected_centers:.0f} centers")

    if best_count < expected_centers:
        print(f"\n  WARNING: Packing constraint may be too restrictive!")
        print(f"  Can fit {best_count} centers but need {expected_centers:.0f}")
    else:
        print(f"\n  Packing seems feasible (can fit {best_count} >= {expected_centers:.0f})")

if __name__ == "__main__":
    main()
