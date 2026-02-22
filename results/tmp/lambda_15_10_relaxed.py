#!/usr/bin/env python3
"""
Relaxed SAT search for Λ₁₅(1¹⁰) with ball size range 10-16.

Key change from focused version: allow centers with ball sizes 10-16
instead of only 14-16. This gives more flexibility for the solver.

Author: Jae-Hyun Baek
Date: 2025-11-26
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

class RelaxedSATEncoder:
    def __init__(self, vertices, n, min_ball_size=10, max_ball_size=16):
        self.vertices = vertices
        self.n = n
        self.num_vertices = len(vertices)
        self.vertices_set = set(vertices)
        self.v_to_idx = {v: i for i, v in enumerate(vertices)}
        self.clauses = []

        # Precompute balls
        self.balls = {}
        self.ball_sizes = {}
        self.ball_membership = defaultdict(list)

        for v in vertices:
            ball = get_ball(v, n, self.vertices_set)
            self.balls[v] = ball
            self.ball_sizes[v] = len(ball)
            for u in ball:
                self.ball_membership[u].append(v)

        # Filter centers by ball size
        self.valid_centers = [v for v in vertices
                              if min_ball_size <= self.ball_sizes[v] <= max_ball_size]

        print(f"Total vertices: {len(vertices)}")
        print(f"Valid centers (ball size {min_ball_size}-{max_ball_size}): {len(self.valid_centers)}")

        # Ball size distribution
        self.by_ball_size = defaultdict(list)
        for v in self.valid_centers:
            self.by_ball_size[self.ball_sizes[v]].append(v)

        print(f"Ball size distribution (valid centers only):")
        for size in sorted(self.by_ball_size.keys()):
            print(f"  Size {size}: {len(self.by_ball_size[size])} vertices")

        # Variable mapping: only valid centers get variables
        self.center_to_var = {c: i + 1 for i, c in enumerate(self.valid_centers)}
        self.num_vars = len(self.valid_centers)
        self.aux_var_counter = self.num_vars + 1

    def var(self, center):
        return self.center_to_var.get(center)

    def new_aux_var(self):
        var = self.aux_var_counter
        self.aux_var_counter += 1
        return var

    def add_clause(self, literals):
        self.clauses.append(literals)

    def at_most_one_sequential(self, literals):
        if len(literals) <= 1:
            return
        if len(literals) <= 4:
            for i in range(len(literals)):
                for j in range(i + 1, len(literals)):
                    self.add_clause([-literals[i], -literals[j]])
            return

        n = len(literals)
        s = [self.new_aux_var() for _ in range(n - 1)]

        self.add_clause([-literals[0], s[0]])
        for i in range(1, n - 1):
            self.add_clause([-s[i-1], s[i]])
            self.add_clause([-literals[i], s[i]])
        self.add_clause([-s[-1], -literals[-1]])

        for i in range(n - 1):
            self.add_clause([-literals[i+1], -s[i]])

    def exactly_one(self, literals):
        if not literals:
            print("WARNING: Empty covering - UNSAT condition!")
            self.add_clause([])  # Empty clause = UNSAT
            return
        self.add_clause(literals)  # At least one
        self.at_most_one_sequential(literals)  # At most one

    def encode(self):
        # Covering constraints: each vertex must be covered by exactly one valid center
        print("Encoding covering constraints...")
        uncoverable = 0

        for i, u in enumerate(self.vertices):
            # Find valid centers that can cover u
            potential_centers = self.ball_membership[u]
            valid_covering = [c for c in potential_centers if c in self.center_to_var]

            if not valid_covering:
                uncoverable += 1
                print(f"  Vertex {u} ({bin(u)}) cannot be covered by any valid center!")
            else:
                literals = [self.var(c) for c in valid_covering]
                self.exactly_one(literals)

            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i}/{self.num_vertices}")

        if uncoverable > 0:
            print(f"WARNING: {uncoverable} vertices cannot be covered - problem is UNSAT")
            return False

        # Packing constraints: centers at distance < 3 cannot both be selected
        print("Encoding packing constraints...")
        packing_count = 0

        valid_set = set(self.valid_centers)
        for i, u in enumerate(self.valid_centers):
            u_var = self.var(u)

            for j in range(i + 1, len(self.valid_centers)):
                v = self.valid_centers[j]
                if hamming_distance(u, v) < 3:
                    v_var = self.var(v)
                    self.add_clause([-u_var, -v_var])
                    packing_count += 1

            if i % 5000 == 0 and i > 0:
                print(f"  {i}/{len(self.valid_centers)}, {packing_count} packing clauses")

        print(f"  Total packing clauses: {packing_count}")
        return True

    def add_hints(self):
        """Add hints to guide solver"""
        # Vertex 0 (all zeros) typically has ball size 16 (or close)
        if 0 in self.center_to_var:
            print(f"Adding hint: vertex 0 as center (ball size {self.ball_sizes[0]})")
            self.add_clause([self.var(0)])

    def write_dimacs(self, filename, include_hints=False):
        with open(filename, 'w') as f:
            # Header with statistics as comments
            f.write(f"c Λ₁₅(1¹⁰) Perfect Partition SAT Encoding\n")
            f.write(f"c Vertices: {self.num_vertices}\n")
            f.write(f"c Valid centers: {len(self.valid_centers)}\n")
            f.write(f"c Ball size range: 10-16\n")
            f.write(f"p cnf {self.aux_var_counter - 1} {len(self.clauses)}\n")

            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

        print(f"Written to {filename}:")
        print(f"  Variables: {self.aux_var_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

        # Also write hints file
        if include_hints:
            hints_file = filename.replace('.cnf', '_hints.txt')
            with open(hints_file, 'w') as f:
                # Hint: high-ball-size centers are good choices
                for size in sorted(self.by_ball_size.keys(), reverse=True)[:3]:
                    for c in self.by_ball_size[size][:10]:
                        f.write(f"{self.var(c)}\n")
            print(f"  Hints written to {hints_file}")


def main():
    n, s = 15, 10
    min_ball, max_ball = 10, 16

    print("=" * 60)
    print(f"Relaxed SAT Search for Λ_{n}(1^{s})")
    print(f"Ball size range: {min_ball}-{max_ball}")
    print("=" * 60)

    # Generate vertices
    print("\nGenerating vertices...")
    vertices = generate_lambda_vertices(n, s)
    print(f"|V(Λ_{n}(1^{s}))| = {len(vertices)}")

    # Create encoder
    encoder = RelaxedSATEncoder(vertices, n, min_ball_size=min_ball, max_ball_size=max_ball)

    # Encode
    print("\nEncoding to CNF...")
    t0 = time.time()
    feasible = encoder.encode()
    encode_time = time.time() - t0
    print(f"Encoding time: {encode_time:.1f}s")

    if not feasible:
        print("\nProblem is UNSAT due to uncoverable vertices")
        return

    # Add hints
    encoder.add_hints()

    # Write CNF
    cnf_file = "lambda_15_10_relaxed.cnf"
    encoder.write_dimacs(cnf_file, include_hints=True)

    # Report expected center count
    # Sum of 1/ball_size over all vertices should equal number of centers
    expected = sum(1.0 / encoder.ball_sizes[v] for v in vertices)
    print(f"\nExpected number of centers: {expected:.1f}")

    print(f"\nTo run solver:")
    print(f"  /Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical {cnf_file}")


if __name__ == "__main__":
    main()
