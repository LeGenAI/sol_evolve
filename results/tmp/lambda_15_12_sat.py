#!/usr/bin/env python3
"""
SAT Encoding for Lambda_15(1^12) Perfect Partition
Based on the same approach used for Lambda_7(1^4) in lucas.tex

Variables:
- x_v = 1 iff vertex v is a center

Constraints:
1. EXACT COVER: Each vertex must be in exactly one ball
   For each vertex u: sum_{v: u in B(v)} x_v = 1

2. PACKING (implicit in exact cover + distance):
   Centers must be at distance >= 3
   For each pair (u,v) with d(u,v) < 3: x_u + x_v <= 1

Note: We'll use cardinality constraints for "exactly one" encoding.
"""

import numpy as np
from collections import defaultdict
import time
import subprocess
import os

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
    """Get closed ball of radius 1."""
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

class SATEncoder:
    def __init__(self, vertices, n):
        self.vertices = vertices
        self.n = n
        self.num_vertices = len(vertices)
        self.vertices_set = set(vertices)
        self.v_to_idx = {v: i for i, v in enumerate(vertices)}
        self.clauses = []
        self.num_vars = len(vertices)  # x_1, ..., x_|V|
        self.aux_var_counter = self.num_vars + 1

        # Precompute balls
        print("Precomputing balls...")
        self.balls = {}  # vertex -> list of vertices in ball
        self.ball_membership = defaultdict(list)  # vertex -> list of centers whose ball contains it

        for v in vertices:
            ball = get_ball(v, n, self.vertices_set)
            self.balls[v] = ball
            for u in ball:
                self.ball_membership[u].append(v)

    def var(self, v):
        """Get SAT variable for vertex v being a center."""
        return self.v_to_idx[v] + 1  # SAT variables are 1-indexed

    def new_aux_var(self):
        """Create a new auxiliary variable."""
        var = self.aux_var_counter
        self.aux_var_counter += 1
        return var

    def add_clause(self, literals):
        """Add a clause (disjunction of literals)."""
        self.clauses.append(literals)

    def at_least_one(self, literals):
        """At least one of the literals must be true."""
        self.add_clause(literals)

    def at_most_one_pairwise(self, literals):
        """At most one (pairwise encoding) - O(n^2) clauses."""
        for i in range(len(literals)):
            for j in range(i + 1, len(literals)):
                self.add_clause([-literals[i], -literals[j]])

    def at_most_one_sequential(self, literals):
        """
        At most one (sequential counter encoding) - O(n) auxiliary variables, O(n) clauses.
        More efficient for large n.
        """
        if len(literals) <= 4:
            return self.at_most_one_pairwise(literals)

        # Create auxiliary variables s_1, ..., s_{n-1}
        # s_i means "at least one of x_1, ..., x_i is true"
        n = len(literals)
        s = [self.new_aux_var() for _ in range(n - 1)]

        # s_1 <-> x_1
        self.add_clause([-literals[0], s[0]])
        self.add_clause([literals[0], -s[0]])

        for i in range(1, n - 1):
            # s_i -> (s_{i-1} or x_i)
            self.add_clause([-s[i], s[i-1], literals[i]])
            # (s_{i-1} or x_i) -> s_i
            self.add_clause([-s[i-1], s[i]])
            self.add_clause([-literals[i], s[i]])
            # If s_{i-1} is true, x_i must be false
            self.add_clause([-s[i-1], -literals[i]])

        # For the last literal
        self.add_clause([-s[-1], -literals[-1]])

    def exactly_one(self, literals):
        """Exactly one of the literals must be true."""
        self.at_least_one(literals)
        self.at_most_one_sequential(literals)

    def encode_covering_constraints(self):
        """Each vertex must be covered by exactly one center's ball."""
        print("Encoding covering constraints...")
        for i, u in enumerate(self.vertices):
            # Get all potential centers whose ball contains u
            centers = self.ball_membership[u]
            literals = [self.var(c) for c in centers]
            self.exactly_one(literals)

            if i % 5000 == 0 and i > 0:
                print(f"  Processed {i}/{self.num_vertices} vertices")

    def encode_packing_constraints(self):
        """
        Centers must be at pairwise distance >= 3.

        Note: With the exact-one covering constraint, packing is somewhat implicit.
        If two vertices u, v with d(u,v) < 3 are both centers, their balls overlap,
        violating the exact-one constraint.

        However, we still add explicit packing constraints to help the solver.
        """
        print("Encoding packing constraints...")
        count = 0

        for i, u in enumerate(self.vertices):
            u_var = self.var(u)
            # Check vertices with index > i (avoid duplicates)
            for j in range(i + 1, self.num_vertices):
                v = self.vertices[j]
                if hamming_distance(u, v) < 3:
                    v_var = self.var(v)
                    self.add_clause([-u_var, -v_var])
                    count += 1

            if i % 5000 == 0 and i > 0:
                print(f"  Processed {i}/{self.num_vertices} vertices, {count} packing clauses")

        print(f"  Total packing clauses: {count}")

    def encode(self):
        """Generate complete SAT encoding."""
        self.encode_covering_constraints()
        self.encode_packing_constraints()

        print(f"\nEncoding complete:")
        print(f"  Variables: {self.aux_var_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

    def write_dimacs(self, filename):
        """Write CNF in DIMACS format."""
        print(f"\nWriting DIMACS to {filename}...")
        with open(filename, 'w') as f:
            f.write(f"p cnf {self.aux_var_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        print(f"  Written {len(self.clauses)} clauses")

    def parse_solution(self, solution_file):
        """Parse SAT solver output and extract centers."""
        centers = []
        with open(solution_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('s '):
                    if 'UNSATISFIABLE' in line:
                        return None
                elif line.startswith('v '):
                    parts = line.split()[1:]
                    for p in parts:
                        if p == '0':
                            continue
                        var = int(p)
                        if var > 0 and var <= self.num_vertices:
                            # This is a center variable (positive = true)
                            centers.append(self.vertices[var - 1])
        return centers

def run_solver(cnf_file, output_file, solver_path, timeout=3600):
    """Run SAT solver on CNF file."""
    print(f"\nRunning solver: {solver_path}")
    print(f"  Input: {cnf_file}")
    print(f"  Output: {output_file}")
    print(f"  Timeout: {timeout}s")

    try:
        result = subprocess.run(
            [solver_path, cnf_file, output_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        print(f"  Solver exit code: {result.returncode}")
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"  Solver timed out after {timeout}s")
        return -1
    except Exception as e:
        print(f"  Error: {e}")
        return -1

def verify_solution(centers, vertices, n):
    """Verify the solution."""
    vertices_set = set(vertices)

    print(f"\nVerifying solution with {len(centers)} centers...")

    # Check pairwise distances
    min_dist = float('inf')
    for i, c1 in enumerate(centers):
        for c2 in centers[i+1:]:
            d = hamming_distance(c1, c2)
            if d < min_dist:
                min_dist = d
    print(f"  Min pairwise distance: {min_dist}")

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

    uncovered = vertices_set - covered
    print(f"  Covered: {len(covered)}/{len(vertices)}")
    print(f"  Uncovered: {len(uncovered)}")
    print(f"  Overlaps: {overlaps}")

    from collections import Counter
    ball_dist = Counter(ball_sizes)
    print(f"  Ball sizes: {dict(sorted(ball_dist.items()))}")

    is_perfect = (len(uncovered) == 0) and (overlaps == 0) and (min_dist >= 3)

    if is_perfect:
        print(f"\n✓ PERFECT PARTITION VERIFIED!")
    else:
        print(f"\n✗ Not a perfect partition")
        if len(uncovered) > 0:
            print(f"  Uncovered: {[bin(v)[2:].zfill(n) for v in list(uncovered)[:5]]}...")

    return is_perfect

def main():
    n, s = 15, 12

    print(f"{'='*60}")
    print(f"SAT Encoding for Lambda_{n}(1^{s}) Perfect Partition")
    print(f"{'='*60}\n")

    # Generate vertices
    print("Generating vertices...")
    vertices = generate_lambda_vertices(n, s)
    print(f"|V| = {len(vertices)}")

    # Create encoder
    encoder = SATEncoder(vertices, n)

    # Generate encoding
    t0 = time.time()
    encoder.encode()
    encode_time = time.time() - t0
    print(f"\nEncoding time: {encode_time:.1f}s")

    # Write CNF
    cnf_file = "lambda_15_12.cnf"
    encoder.write_dimacs(cnf_file)

    # Find solver
    solver_candidates = [
        "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical",
        "cadical",
        "/usr/local/bin/cadical"
    ]

    solver_path = None
    for candidate in solver_candidates:
        if os.path.exists(candidate) or os.system(f"which {candidate} > /dev/null 2>&1") == 0:
            solver_path = candidate
            break

    if not solver_path:
        print("\nNo SAT solver found. Please run manually:")
        print(f"  cadical {cnf_file} solution.txt")
        return

    # Run solver
    output_file = "lambda_15_12_solution.txt"
    t0 = time.time()
    exit_code = run_solver(cnf_file, output_file, solver_path, timeout=7200)  # 2 hour timeout
    solve_time = time.time() - t0

    print(f"\nSolve time: {solve_time:.1f}s")

    if exit_code == 10:  # SATISFIABLE
        print("\n" + "="*60)
        print("SAT - Solution found!")
        print("="*60)

        centers = encoder.parse_solution(output_file)
        if centers:
            print(f"Centers found: {len(centers)}")

            # Verify
            is_perfect = verify_solution(centers, vertices, n)

            if is_perfect:
                # Save solution
                np.save('sat_centers.npy', np.array(centers))

                # Save human-readable version
                with open('lambda_15_12_centers.txt', 'w') as f:
                    f.write(f"Perfect Partition of Lambda_15(1^12)\n")
                    f.write(f"="*50 + "\n")
                    f.write(f"Total vertices: {len(vertices)}\n")
                    f.write(f"Number of centers: {len(centers)}\n\n")
                    f.write("Centers:\n")
                    for c in sorted(centers):
                        f.write(f"  {bin(c)[2:].zfill(n)}\n")

                print(f"\nResults saved to:")
                print(f"  sat_centers.npy")
                print(f"  lambda_15_12_centers.txt")

    elif exit_code == 20:  # UNSATISFIABLE
        print("\n" + "="*60)
        print("UNSAT - No perfect partition exists!")
        print("="*60)

    else:
        print(f"\nSolver returned unexpected code: {exit_code}")

if __name__ == "__main__":
    main()
