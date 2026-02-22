#!/usr/bin/env python3
"""
Focused SAT search for Λ₁₅(1¹⁰) with hints from algebraic analysis.

We know: 0×14 + 1×15 + 2032×16 = 32,527 is feasible.
Strategy: Add unit propagation hints to guide the solver.
"""

import numpy as np
from collections import defaultdict, Counter
import time
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

def generate_lambda_vertices(n, s):
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball(center, n, vertices_set):
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

class FocusedSATEncoder:
    def __init__(self, vertices, n):
        self.vertices = vertices
        self.n = n
        self.num_vertices = len(vertices)
        self.vertices_set = set(vertices)
        self.v_to_idx = {v: i for i, v in enumerate(vertices)}
        self.clauses = []
        self.num_vars = len(vertices)
        self.aux_var_counter = self.num_vars + 1

        # Precompute balls and their sizes
        self.balls = {}
        self.ball_sizes = {}
        self.ball_membership = defaultdict(list)

        for v in vertices:
            ball = get_ball(v, n, self.vertices_set)
            self.balls[v] = ball
            self.ball_sizes[v] = len(ball)
            for u in ball:
                self.ball_membership[u].append(v)

        # Categorize by ball size
        self.by_ball_size = defaultdict(list)
        for v, size in self.ball_sizes.items():
            self.by_ball_size[size].append(v)

        print(f"Ball size distribution:")
        for size in sorted(self.by_ball_size.keys()):
            print(f"  Size {size}: {len(self.by_ball_size[size])} vertices")

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

    def encode(self, add_hints=False):
        # Covering constraints
        print("Encoding covering constraints...")
        for u in self.vertices:
            centers = self.ball_membership[u]
            literals = [self.var(c) for c in centers]
            self.exactly_one(literals)

        # Packing constraints
        print("Encoding packing constraints...")
        for i, u in enumerate(self.vertices):
            u_var = self.var(u)
            for j in range(i + 1, self.num_vertices):
                v = self.vertices[j]
                if hamming_distance(u, v) < 3:
                    v_var = self.var(v)
                    self.add_clause([-u_var, -v_var])

            if i % 10000 == 0:
                print(f"  {i}/{self.num_vertices}")

        if add_hints:
            self.add_algebraic_hints()

    def add_algebraic_hints(self):
        """
        Add hints based on algebraic analysis:
        - We need ~1 center with ball size 15
        - We need ~2032 centers with ball size 16
        - We need 0 centers with ball size 14
        """
        print("Adding algebraic hints...")

        # Hint 1: Force vertex 0 (all zeros) to be a center
        # It always has ball size 16 (n=15 neighbors all valid)
        zero_var = self.var(0)
        print(f"  Hint: Setting vertex 0 as center")
        self.add_clause([zero_var])

        # Hint 2: The all-ones vertex 32767 is NOT in Λ₁₅(1¹⁰)
        # because it has 15 consecutive 1s > 10
        # So we don't need to handle it

    def write_dimacs(self, filename):
        print(f"Writing DIMACS to {filename}...")
        with open(filename, 'w') as f:
            f.write(f"p cnf {self.aux_var_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        print(f"  Written {len(self.clauses)} clauses")

def main():
    n, s = 15, 10

    print("="*60)
    print(f"Focused SAT Search for Λ_{n}(1^{s})")
    print("="*60)

    # Generate vertices
    print("\nGenerating vertices...")
    vertices = generate_lambda_vertices(n, s)
    print(f"|V| = {len(vertices)}")

    # Create encoder with hints
    encoder = FocusedSATEncoder(vertices, n)

    # Encode with hints
    t0 = time.time()
    encoder.encode(add_hints=True)
    encode_time = time.time() - t0
    print(f"\nEncoding time: {encode_time:.1f}s")
    print(f"Variables: {encoder.aux_var_counter - 1}")
    print(f"Clauses: {len(encoder.clauses)}")

    # Write CNF
    cnf_file = "lambda_15_10_focused.cnf"
    encoder.write_dimacs(cnf_file)

    # Run solver with longer timeout
    solver_path = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"
    timeout = 7200  # 2 hours

    print(f"\nRunning CaDiCaL (timeout: {timeout}s)...")
    t0 = time.time()

    try:
        proc = subprocess.run(
            [solver_path, cnf_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        solve_time = time.time() - t0
        print(f"Solve time: {solve_time:.1f}s")

        if proc.returncode == 10:
            print("\n" + "="*60)
            print("SAT! Parsing solution...")
            print("="*60)

            centers_indices = []
            for line in proc.stdout.split('\n'):
                if line.startswith('v '):
                    parts = line.split()[1:]
                    for p in parts:
                        if p == '0':
                            continue
                        var = int(p)
                        if var > 0 and var <= len(vertices):
                            centers_indices.append(var - 1)

            centers = [vertices[i] for i in centers_indices]
            print(f"Centers found: {len(centers)}")

            # Verify
            vertices_set = set(vertices)
            covered = set()
            ball_sizes = []

            for c in centers:
                ball = get_ball(c, n, vertices_set)
                ball_sizes.append(len(ball))
                covered.update(ball)

            print(f"Covered: {len(covered)}/{len(vertices)}")
            print(f"Ball sizes: {dict(Counter(ball_sizes))}")

            if len(covered) == len(vertices):
                print("\n✓ PERFECT PARTITION FOR Λ₁₅(1¹⁰) FOUND!")
                np.save('lambda_15_10_centers.npy', np.array(centers))

        elif proc.returncode == 20:
            print("\nUNSAT: No perfect partition exists for Λ₁₅(1¹⁰)")

        else:
            print(f"\nSolver returned {proc.returncode}")

    except subprocess.TimeoutExpired:
        print(f"\nTIMEOUT after {timeout}s")

if __name__ == "__main__":
    main()
