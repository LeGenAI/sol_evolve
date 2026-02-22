#!/usr/bin/env python3
"""
Systematic SAT Search for Perfect Partitions of Λₙ(1ˢ)

Explores all combinations to determine existence of perfect partitions:
- Λ₇(1ˢ) for s = 1, 2, 3 (s=4 already known)
- Λ₁₅(1ˢ) for s = 1, 2, ..., 11 (s=12 just discovered!)

Author: Jae-Hyun Baek (with Claude Code assistance)
Date: 2025-11-25
"""

import numpy as np
from collections import defaultdict, Counter
import time
import subprocess
import os
import json

# ============================================================
# Core Functions
# ============================================================

def has_circular_consecutive_ones(n, bits, s):
    """Check if n-bit number has s consecutive 1s in circular form."""
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
    """Generate all vertices of Λₙ(1ˢ)."""
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball(center, n, vertices_set):
    """Get closed ball of radius 1 around center."""
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

# ============================================================
# SAT Encoder
# ============================================================

class SATEncoder:
    def __init__(self, vertices, n):
        self.vertices = vertices
        self.n = n
        self.num_vertices = len(vertices)
        self.vertices_set = set(vertices)
        self.v_to_idx = {v: i for i, v in enumerate(vertices)}
        self.clauses = []
        self.num_vars = len(vertices)
        self.aux_var_counter = self.num_vars + 1

        # Precompute balls
        self.balls = {}
        self.ball_membership = defaultdict(list)

        for v in vertices:
            ball = get_ball(v, n, self.vertices_set)
            self.balls[v] = ball
            for u in ball:
                self.ball_membership[u].append(v)

    def var(self, v):
        return self.v_to_idx[v] + 1

    def new_aux_var(self):
        var = self.aux_var_counter
        self.aux_var_counter += 1
        return var

    def add_clause(self, literals):
        self.clauses.append(literals)

    def at_least_one(self, literals):
        self.add_clause(literals)

    def at_most_one_pairwise(self, literals):
        for i in range(len(literals)):
            for j in range(i + 1, len(literals)):
                self.add_clause([-literals[i], -literals[j]])

    def at_most_one_sequential(self, literals):
        if len(literals) <= 4:
            return self.at_most_one_pairwise(literals)

        n = len(literals)
        s = [self.new_aux_var() for _ in range(n - 1)]

        self.add_clause([-literals[0], s[0]])
        self.add_clause([literals[0], -s[0]])

        for i in range(1, n - 1):
            self.add_clause([-s[i], s[i-1], literals[i]])
            self.add_clause([-s[i-1], s[i]])
            self.add_clause([-literals[i], s[i]])
            self.add_clause([-s[i-1], -literals[i]])

        self.add_clause([-s[-1], -literals[-1]])

    def exactly_one(self, literals):
        self.at_least_one(literals)
        self.at_most_one_sequential(literals)

    def encode(self):
        # Covering constraints
        for u in self.vertices:
            centers = self.ball_membership[u]
            literals = [self.var(c) for c in centers]
            self.exactly_one(literals)

        # Packing constraints
        for i, u in enumerate(self.vertices):
            u_var = self.var(u)
            for j in range(i + 1, self.num_vertices):
                v = self.vertices[j]
                if hamming_distance(u, v) < 3:
                    v_var = self.var(v)
                    self.add_clause([-u_var, -v_var])

    def write_dimacs(self, filename):
        with open(filename, 'w') as f:
            f.write(f"p cnf {self.aux_var_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

# ============================================================
# Solution Parser and Verifier
# ============================================================

def parse_solution(stdout, num_vertices, vertices):
    """Parse SAT solver stdout and extract centers."""
    centers_indices = []

    for line in stdout.split('\n'):
        if line.startswith('v '):
            parts = line.split()[1:]
            for p in parts:
                if p == '0':
                    continue
                var = int(p)
                if var > 0 and var <= num_vertices:
                    centers_indices.append(var - 1)

    return [vertices[i] for i in centers_indices]

def verify_solution(centers, vertices, n):
    """Verify the solution is a perfect partition."""
    vertices_set = set(vertices)

    # Check pairwise distances
    min_dist = float('inf')
    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            d = hamming_distance(c1, c2)
            if d < min_dist:
                min_dist = d

    if min_dist < 3:
        return False, f"Min distance {min_dist} < 3"

    # Check coverage
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

    if overlaps > 0:
        return False, f"{overlaps} overlapping vertices"

    uncovered = vertices_set - covered
    if len(uncovered) > 0:
        return False, f"{len(uncovered)} uncovered vertices"

    ball_dist = Counter(ball_sizes)
    return True, {
        "centers": len(centers),
        "min_dist": min_dist,
        "ball_sizes": dict(sorted(ball_dist.items()))
    }

# ============================================================
# Main Search Function
# ============================================================

def search_lambda(n, s, solver_path, timeout=600):
    """
    Search for perfect partition of Λₙ(1ˢ).

    Returns:
        dict with keys:
        - status: "SAT", "UNSAT", "TIMEOUT", "ERROR"
        - vertices: number of vertices
        - centers: number of centers (if SAT)
        - ball_sizes: distribution (if SAT)
        - time: solve time
    """
    result = {
        "n": n,
        "s": s,
        "status": "ERROR",
        "vertices": 0,
        "time": 0
    }

    print(f"\n{'='*60}")
    print(f"Searching Λ_{n}(1^{s})")
    print(f"{'='*60}")

    # Generate vertices
    t0 = time.time()
    vertices = generate_lambda_vertices(n, s)
    result["vertices"] = len(vertices)

    print(f"|V| = {len(vertices)}")

    if len(vertices) == 0:
        result["status"] = "EMPTY"
        return result

    # For s >= n, all 2^n strings are valid
    if s >= n:
        print(f"s >= n: Full hypercube (trivial case)")
        result["status"] = "TRIVIAL"
        return result

    # Encode as SAT
    print("Encoding SAT...")
    encoder = SATEncoder(vertices, n)
    encoder.encode()

    cnf_file = f"lambda_{n}_{s}.cnf"
    encoder.write_dimacs(cnf_file)

    print(f"Variables: {encoder.aux_var_counter - 1}")
    print(f"Clauses: {len(encoder.clauses)}")

    # Run solver
    print(f"Running CaDiCaL (timeout: {timeout}s)...")
    try:
        proc = subprocess.run(
            [solver_path, cnf_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        solve_time = time.time() - t0
        result["time"] = round(solve_time, 2)

        # Check result
        if proc.returncode == 10:  # SAT
            print("SAT! Parsing solution...")
            centers = parse_solution(proc.stdout, len(vertices), vertices)

            is_valid, info = verify_solution(centers, vertices, n)

            if is_valid:
                result["status"] = "SAT"
                result["centers"] = info["centers"]
                result["ball_sizes"] = info["ball_sizes"]
                result["min_dist"] = info["min_dist"]
                print(f"✓ VERIFIED: {info['centers']} centers, ball sizes {info['ball_sizes']}")

                # Save solution
                np.save(f"lambda_{n}_{s}_centers.npy", np.array(centers))

            else:
                result["status"] = "VERIFY_FAILED"
                result["error"] = info
                print(f"✗ Verification failed: {info}")

        elif proc.returncode == 20:  # UNSAT
            result["status"] = "UNSAT"
            print("UNSAT: No perfect partition exists")

        else:
            result["status"] = "ERROR"
            result["error"] = f"Solver returned {proc.returncode}"
            print(f"Error: Solver returned {proc.returncode}")

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["time"] = timeout
        print(f"TIMEOUT after {timeout}s")

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        print(f"Error: {e}")

    return result

# ============================================================
# Main
# ============================================================

def main():
    solver_path = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"

    if not os.path.exists(solver_path):
        print(f"Solver not found: {solver_path}")
        return

    # Define search space
    experiments = []

    # Λ₇(1ˢ) for s = 1, 2, 3 (s=4 already known: SAT with 15 centers)
    for s in [3, 2, 1]:
        experiments.append((7, s))

    # Λ₁₅(1ˢ) for s = 11, 10, ..., 1 (s=12 just discovered: SAT with 2047 centers)
    for s in range(11, 0, -1):
        experiments.append((15, s))

    # Results storage
    results = []

    # Known results
    known = {
        (7, 4): {"status": "SAT", "centers": 15, "vertices": 99, "ball_sizes": {6: 3, 7: 6, 8: 6}},
        (15, 12): {"status": "SAT", "centers": 2047, "vertices": 32707, "ball_sizes": {14: 11, 15: 23, 16: 2013}}
    }

    print("="*60)
    print("SYSTEMATIC SEARCH FOR PERFECT PARTITIONS OF Λₙ(1ˢ)")
    print("="*60)
    print("\nKnown results:")
    for (n, s), info in known.items():
        print(f"  Λ_{n}(1^{s}): {info['status']} - {info.get('centers', 'N/A')} centers")

    print(f"\nRunning {len(experiments)} experiments...")

    for n, s in experiments:
        result = search_lambda(n, s, solver_path, timeout=3600)  # 1 hour timeout
        results.append(result)

        # Save intermediate results
        with open("lambda_search_results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print("\n### Λ₇(1ˢ) Family ###")
    print(f"{'s':>3} | {'|V|':>8} | {'Status':>10} | {'Centers':>8} | {'Ball Sizes'}")
    print("-"*60)

    # Print Λ₇ results
    print(f"  4 | {99:>8} | {'SAT':>10} | {15:>8} | {6: 3, 7: 6, 8: 6}")  # Known
    for r in results:
        if r["n"] == 7:
            centers = r.get("centers", "-")
            ball_sizes = r.get("ball_sizes", "-")
            print(f"{r['s']:>3} | {r['vertices']:>8} | {r['status']:>10} | {str(centers):>8} | {ball_sizes}")

    print("\n### Λ₁₅(1ˢ) Family ###")
    print(f"{'s':>3} | {'|V|':>8} | {'Status':>10} | {'Centers':>8} | {'Ball Sizes'}")
    print("-"*60)

    # Print Λ₁₅ results
    print(f" 12 | {32707:>8} | {'SAT':>10} | {2047:>8} | {14: 11, 15: 23, 16: 2013}")  # Known
    for r in results:
        if r["n"] == 15:
            centers = r.get("centers", "-")
            ball_sizes = r.get("ball_sizes", "-")
            print(f"{r['s']:>3} | {r['vertices']:>8} | {r['status']:>10} | {str(centers):>8} | {ball_sizes}")

    # Save final results
    all_results = {
        "known": known,
        "new_results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("lambda_search_complete.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to lambda_search_complete.json")

if __name__ == "__main__":
    main()
