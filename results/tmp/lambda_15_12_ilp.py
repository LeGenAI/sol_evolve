#!/usr/bin/env python3
"""
ILP Model for Lambda_15(1^12) Perfect Partition
Using PuLP (open-source) or Gurobi if available
"""

import numpy as np
from collections import defaultdict
import time
import sys

def has_circular_consecutive_ones(n, bits, s):
    doubled = (n << bits) | n
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

def build_ilp_model(vertices, n, k_target=None):
    """
    Build ILP model for perfect partition.

    Variables:
    - x[v] = 1 if vertex v is a center

    Constraints:
    1. Sum of x[v] = k (if k_target specified) or minimize
    2. For each vertex u: sum_{v: u in ball(v)} x[v] = 1 (exact cover)
    3. For each pair (u,v) with d(u,v) < 3: x[u] + x[v] <= 1 (packing)
    """
    try:
        import pulp
    except ImportError:
        print("PuLP not installed. Install with: pip install pulp")
        sys.exit(1)

    print(f"Building ILP model...")
    print(f"  Vertices: {len(vertices)}")

    vertices_set = set(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    idx_to_v = {i: v for i, v in enumerate(vertices)}

    # Precompute balls
    print("  Computing balls...")
    balls = {}  # center_idx -> list of vertex indices in ball
    ball_membership = defaultdict(list)  # vertex_idx -> list of center indices whose ball contains it

    for i, v in enumerate(vertices):
        ball = get_ball(v, n, vertices_set)
        ball_indices = [v_to_idx[u] for u in ball]
        balls[i] = ball_indices
        for j in ball_indices:
            ball_membership[j].append(i)

    # Precompute close pairs (distance < 3)
    print("  Computing close pairs...")
    close_pairs = []
    for i, v in enumerate(vertices):
        # Only check vertices with index > i to avoid duplicates
        for j in range(i + 1, len(vertices)):
            u = vertices[j]
            if hamming_distance(v, u) < 3:
                close_pairs.append((i, j))

        if i % 5000 == 0:
            print(f"    Processed {i}/{len(vertices)} vertices, {len(close_pairs)} close pairs so far")

    print(f"  Total close pairs: {len(close_pairs)}")

    # Create model
    print("  Creating PuLP model...")
    model = pulp.LpProblem("LambdaPerfectPartition", pulp.LpMinimize)

    # Variables
    x = {}
    for i in range(len(vertices)):
        x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')

    # Objective: minimize number of centers
    model += pulp.lpSum(x[i] for i in range(len(vertices))), "MinCenters"

    # Constraint: exact cover (each vertex covered exactly once)
    print("  Adding covering constraints...")
    for j in range(len(vertices)):
        centers_covering_j = ball_membership[j]
        model += pulp.lpSum(x[i] for i in centers_covering_j) == 1, f"Cover_{j}"

    # Constraint: packing (close pairs can't both be centers)
    print("  Adding packing constraints...")
    for i, j in close_pairs:
        model += x[i] + x[j] <= 1, f"Pack_{i}_{j}"

    # Optional: fix k if target specified
    if k_target:
        model += pulp.lpSum(x[i] for i in range(len(vertices))) == k_target, "FixK"

    print(f"  Model built:")
    print(f"    Variables: {len(x)}")
    print(f"    Covering constraints: {len(vertices)}")
    print(f"    Packing constraints: {len(close_pairs)}")

    return model, x, idx_to_v, balls

def solve_and_extract(model, x, idx_to_v, vertices, n, time_limit=3600):
    """Solve the ILP and extract centers."""
    import pulp

    print(f"\nSolving ILP (time limit: {time_limit}s)...")
    t0 = time.time()

    # Try different solvers
    solver = None
    try:
        # Try Gurobi first (fastest)
        solver = pulp.GUROBI(msg=1, timeLimit=time_limit)
        print("  Using Gurobi solver")
    except:
        try:
            # Try CBC (open-source)
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit)
            print("  Using CBC solver")
        except:
            solver = pulp.PULP_CBC_CMD(msg=1)
            print("  Using default solver")

    status = model.solve(solver)
    solve_time = time.time() - t0

    print(f"\nSolve completed in {solve_time:.1f}s")
    print(f"Status: {pulp.LpStatus[status]}")

    if status == pulp.LpStatusOptimal or status == pulp.LpStatusNotSolved:
        # Extract centers
        centers = []
        for i, var in x.items():
            if var.varValue and var.varValue > 0.5:
                centers.append(idx_to_v[i])

        print(f"Centers found: {len(centers)}")

        # Verify
        vertices_set = set(vertices)
        covered = set()
        for c in centers:
            ball = get_ball(c, n, vertices_set)
            covered.update(ball)

        print(f"Covered: {len(covered)}/{len(vertices)}")

        return centers, status
    else:
        print("No solution found")
        return None, status

def main():
    n, s = 15, 12

    print(f"{'='*60}")
    print(f"ILP for Lambda_{n}(1^{s}) Perfect Partition")
    print(f"{'='*60}\n")

    # Generate vertices
    print("Generating vertices...")
    vertices = generate_lambda_vertices(n, s)
    print(f"|V| = {len(vertices)}")

    # Build and solve ILP
    model, x, idx_to_v, balls = build_ilp_model(vertices, n)

    # Solve with 1 hour time limit
    centers, status = solve_and_extract(model, x, idx_to_v, vertices, n, time_limit=3600)

    if centers:
        print(f"\n{'='*60}")
        print("SOLUTION FOUND")
        print(f"{'='*60}")
        print(f"Number of centers: {len(centers)}")

        # Save solution
        np.save('ilp_centers.npy', np.array(centers))
        print(f"Centers saved to ilp_centers.npy")

        # Verify solution
        vertices_set = set(vertices)
        covered = set()
        ball_sizes = []

        for c in centers:
            ball = get_ball(c, n, vertices_set)
            ball_sizes.append(len(ball))
            for v in ball:
                if v in covered:
                    print(f"WARNING: Overlap detected at {bin(v)[2:].zfill(n)}")
                covered.add(v)

        uncovered = vertices_set - covered
        print(f"\nVerification:")
        print(f"  Covered: {len(covered)}/{len(vertices)}")
        print(f"  Uncovered: {len(uncovered)}")
        print(f"  Ball sizes: {dict(sorted(set((s, ball_sizes.count(s)) for s in ball_sizes)))}")

        if len(uncovered) == 0:
            print("\n✓ PERFECT PARTITION FOUND!")

            # Show some centers
            print(f"\nFirst 10 centers:")
            for c in sorted(centers)[:10]:
                print(f"  {bin(c)[2:].zfill(n)}")

if __name__ == "__main__":
    main()
