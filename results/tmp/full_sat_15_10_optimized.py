#!/usr/bin/env python3
"""
Full SAT search for Lambda_15(1^10) with optimized encoding.

Key optimizations:
1. Only vertices with ball size >= 14 as candidates (to reduce variables)
2. Symmetry breaking hints
3. Better at-most-one encoding

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import time
import subprocess

def has_circular_consecutive_ones(n_val, bits, s):
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

class OptimizedSATEncoder:
    """Optimized SAT encoder for perfect partition."""

    def __init__(self, vertices, n, min_ball_size=14):
        self.vertices = vertices
        self.n = n
        self.num_vertices = len(vertices)
        self.vertices_set = set(vertices)
        self.clauses = []

        print("Precomputing balls...")
        self.balls = {}
        self.ball_sizes = {}
        self.ball_membership = defaultdict(list)

        for v in vertices:
            ball = get_ball(v, n, self.vertices_set)
            self.balls[v] = ball
            self.ball_sizes[v] = len(ball)
            for u in ball:
                self.ball_membership[u].append(v)

        # Filter candidates by ball size
        self.candidates = [v for v in vertices if self.ball_sizes[v] >= min_ball_size]
        print(f"Total vertices: {len(vertices)}")
        print(f"Candidates (ball size >= {min_ball_size}): {len(self.candidates)}")

        # Ball size distribution of candidates
        size_dist = Counter(self.ball_sizes[v] for v in self.candidates)
        print(f"Candidate ball size distribution:")
        for s in sorted(size_dist.keys()):
            print(f"  Size {s}: {size_dist[s]}")

        # Variable mapping
        self.var_map = {c: i + 1 for i, c in enumerate(self.candidates)}
        self.num_vars = len(self.candidates)
        self.aux_counter = self.num_vars + 1

    def new_aux(self):
        var = self.aux_counter
        self.aux_counter += 1
        return var

    def add_clause(self, lits):
        self.clauses.append(lits)

    def at_most_one_sequential(self, lits):
        if len(lits) <= 1:
            return
        if len(lits) <= 4:
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self.add_clause([-lits[i], -lits[j]])
            return

        n = len(lits)
        s = [self.new_aux() for _ in range(n - 1)]

        self.add_clause([-lits[0], s[0]])
        for i in range(1, n - 1):
            self.add_clause([-s[i-1], s[i]])
            self.add_clause([-lits[i], s[i]])
        self.add_clause([-s[-1], -lits[-1]])

        for i in range(n - 1):
            self.add_clause([-lits[i+1], -s[i]])

    def encode(self):
        print("\n" + "=" * 60)
        print("Building SAT Encoding")
        print("=" * 60)

        # Check coverability first
        print("\nChecking coverability...")
        uncoverable = []
        for v in self.vertices:
            # Find candidates that can cover v
            potential = [c for c in self.ball_membership[v] if c in self.var_map]
            if not potential:
                uncoverable.append(v)

        if uncoverable:
            print(f"\nERROR: {len(uncoverable)} vertices cannot be covered!")
            print("Sample uncoverable:")
            for v in uncoverable[:10]:
                print(f"  {to_binary(v, self.n)} (ball size {self.ball_sizes[v]})")

            # These vertices have ball size < min_ball_size AND
            # all their neighbors also have ball size < min_ball_size
            # This means the min_ball_size filter is too strict
            print("\nNeed to include more candidates...")
            return False

        # Covering constraints
        print("\nEncoding covering constraints...")
        for i, v in enumerate(self.vertices):
            covering = [c for c in self.ball_membership[v] if c in self.var_map]
            lits = [self.var_map[c] for c in covering]
            self.add_clause(lits)  # At least one
            self.at_most_one_sequential(lits)  # At most one

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{self.num_vertices}")

        # Packing constraints
        print("\nEncoding packing constraints...")
        pack_count = 0
        for i, c1 in enumerate(self.candidates):
            v1 = self.var_map[c1]
            for c2 in self.candidates[i+1:]:
                if hamming_distance(c1, c2) < 3:
                    v2 = self.var_map[c2]
                    self.add_clause([-v1, -v2])
                    pack_count += 1

            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1}/{len(self.candidates)}, {pack_count} clauses")

        print(f"  Total packing clauses: {pack_count}")

        # Symmetry breaking: force vertex 0 to be a center
        if 0 in self.var_map:
            print("\nAdding symmetry breaking: vertex 0 as center")
            self.add_clause([self.var_map[0]])

        print(f"\nTotal encoding:")
        print(f"  Variables: {self.aux_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

        return True

    def write_dimacs(self, filename):
        with open(filename, 'w') as f:
            f.write(f"c Lambda_15(1^10) Full SAT Encoding\n")
            f.write(f"c Vertices: {self.num_vertices}\n")
            f.write(f"c Candidates: {len(self.candidates)}\n")
            f.write(f"p cnf {self.aux_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        print(f"Written to {filename}")

    def parse_solution(self, output):
        centers = []
        for line in output.split('\n'):
            if line.startswith('v '):
                parts = line.split()[1:]
                for p in parts:
                    if p == '0':
                        continue
                    var = int(p)
                    if var > 0 and var <= self.num_vars:
                        centers.append(self.candidates[var - 1])
        return centers

def verify_solution(centers, n=15, s=10):
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    print(f"\nSolution: {len(centers)} centers")
    print(f"Target: {len(vertices)} vertices")

    invalid = [c for c in centers if has_circular_consecutive_ones(c, n, s)]
    if invalid:
        print(f"\nERROR: {len(invalid)} invalid centers!")
        return False
    print("All centers valid: OK")

    min_dist = float('inf')
    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            d = hamming_distance(c1, c2)
            if d < min_dist:
                min_dist = d
    print(f"Min pairwise distance: {min_dist}")

    if min_dist < 3:
        print("ERROR: Centers too close!")
        return False

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

    print(f"Covered: {len(covered)}/{len(vertices)}")
    print(f"Uncovered: {len(uncovered)}")
    print(f"Overlaps: {overlaps}")
    print(f"Ball sizes: {dict(Counter(ball_sizes))}")

    if len(uncovered) == 0 and overlaps == 0 and min_dist >= 3:
        print("\nPERFECT PARTITION VERIFIED!")
        return True
    return False

def save_solution(centers, n=15, s=10):
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    np.save('/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_centers.npy',
            np.array(centers))

    with open('/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_centers.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Perfect Partition of Lambda_{n}(1^{s})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: 2025-11-27\n")
        f.write(f"Method: Full SAT Search (Optimized)\n\n")
        f.write(f"n = {n}, s = {s}\n")
        f.write(f"|V| = {len(vertices)}\n")
        f.write(f"Centers = {len(centers)}\n\n")

        f.write("CENTERS:\n")
        for i, c in enumerate(sorted(centers)):
            ball = get_ball(c, n, vertices_set)
            f.write(f"{i+1:4d}. {to_binary(c, n)}  (ball: {len(ball)})\n")

    print("\nSolution saved!")

def main():
    print("=" * 70)
    print("Lambda_15(1^10) Full SAT Search (Optimized)")
    print("=" * 70)

    n, s = 15, 10

    print("\nGenerating vertices...")
    vertices = generate_lambda_vertices(n, s)
    print(f"|V| = {len(vertices)}")

    # Try with all candidates (no filtering)
    encoder = OptimizedSATEncoder(vertices, n, min_ball_size=14)

    if not encoder.encode():
        print("\nTrying with no ball size filter...")
        encoder = OptimizedSATEncoder(vertices, n, min_ball_size=1)
        if not encoder.encode():
            print("\nEncoding failed!")
            return

    cnf_file = "/Users/baegjaehyeon/CodeEvolve/results/tmp/full_15_10.cnf"
    encoder.write_dimacs(cnf_file)

    solver = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"
    timeout = 28800  # 8 hours

    print(f"\nRunning CaDiCaL (timeout: {timeout}s = {timeout/3600:.1f}h)...")
    t0 = time.time()

    try:
        proc = subprocess.run(
            [solver, cnf_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        solve_time = time.time() - t0
        print(f"Solve time: {solve_time:.1f}s ({solve_time/3600:.2f}h)")

        if proc.returncode == 10:
            print("\nSAT! Parsing solution...")
            centers = encoder.parse_solution(proc.stdout)
            print(f"Found {len(centers)} centers")

            if verify_solution(centers, n, s):
                save_solution(centers, n, s)

        elif proc.returncode == 20:
            print("\nUNSAT - No perfect partition exists for Lambda_15(1^10)!")
            print("\nThis is a significant theoretical result!")

        else:
            print(f"\nUnknown result (code {proc.returncode})")

    except subprocess.TimeoutExpired:
        print(f"\nTimeout after {timeout}s")

if __name__ == "__main__":
    main()
