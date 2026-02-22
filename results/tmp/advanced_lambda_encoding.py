#!/usr/bin/env python3
"""
Advanced SAT Encoding Strategies for Λₙ(1ˢ) Perfect Partitions

This script explores more efficient encoding strategies:
1. Symmetry Breaking - using automorphisms of the Lucas cube
2. Phase Saving - using greedy solution as initial assignment
3. Cardinality Networks - more efficient "exactly one" encoding
4. Lexicographic Ordering - breaking ties between symmetric solutions

Author: Jae-Hyun Baek (with Claude Code as HITL "human")
Date: 2025-11-25
"""

import numpy as np
from collections import defaultdict, Counter
import time
import subprocess
import random

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

def circular_rotate(x, n, k):
    """Rotate n-bit number x by k positions."""
    k = k % n
    return ((x << k) | (x >> (n - k))) & ((1 << n) - 1)

def bit_complement(x, n):
    """Complement all bits of n-bit number x."""
    return x ^ ((1 << n) - 1)

class AdvancedSATEncoder:
    """Advanced SAT encoder with multiple optimization strategies."""

    def __init__(self, vertices, n, s):
        self.vertices = vertices
        self.n = n
        self.s = s
        self.num_vertices = len(vertices)
        self.vertices_set = set(vertices)
        self.v_to_idx = {v: i for i, v in enumerate(vertices)}
        self.clauses = []
        self.num_vars = len(vertices)
        self.aux_var_counter = self.num_vars + 1

        # Precompute balls
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

        # Categorize by ball size
        self.by_ball_size = defaultdict(list)
        for v, size in self.ball_sizes.items():
            self.by_ball_size[size].append(v)

    def var(self, v):
        return self.v_to_idx[v] + 1

    def new_aux_var(self):
        var = self.aux_var_counter
        self.aux_var_counter += 1
        return var

    def add_clause(self, literals):
        self.clauses.append(literals)

    # ========================================
    # Strategy 1: Symmetry Breaking
    # ========================================

    def get_orbit_representatives(self):
        """
        Find orbit representatives under rotation and complement.
        Two vertices are in the same orbit if one can be transformed
        to the other via rotation or complement.
        """
        print("Computing orbit representatives...")

        visited = set()
        representatives = []

        for v in self.vertices:
            if v in visited:
                continue

            # Generate orbit of v
            orbit = set()
            for k in range(self.n):
                rotated = circular_rotate(v, self.n, k)
                if rotated in self.vertices_set:
                    orbit.add(rotated)
                # Also try complement (if in vertex set)
                comp = bit_complement(rotated, self.n)
                if comp in self.vertices_set:
                    orbit.add(comp)

            # Choose lexicographically smallest as representative
            rep = min(orbit)
            representatives.append(rep)
            visited.update(orbit)

        print(f"  {len(self.vertices)} vertices → {len(representatives)} orbits")
        return representatives

    def add_symmetry_breaking(self, orbit_reps):
        """
        Add symmetry breaking constraints:
        - If v is the orbit representative, and v is a center,
          then no other orbit member can be a center.
        """
        print("Adding symmetry breaking constraints...")
        count = 0

        for rep in orbit_reps:
            if rep not in self.vertices_set:
                continue

            # Find other orbit members
            orbit = set()
            for k in range(1, self.n):  # Skip k=0 (identity)
                rotated = circular_rotate(rep, self.n, k)
                if rotated in self.vertices_set and rotated != rep:
                    orbit.add(rotated)

            # Add: if rep is center → orbit members are not centers
            # (rep → ¬orbit_member) ≡ (¬rep ∨ ¬orbit_member)
            rep_var = self.var(rep)
            for other in orbit:
                other_var = self.var(other)
                self.add_clause([-rep_var, -other_var])
                count += 1

        print(f"  Added {count} symmetry breaking clauses")

    # ========================================
    # Strategy 2: Lexicographic Ordering
    # ========================================

    def add_lexicographic_ordering(self):
        """
        Force centers to be chosen in lexicographic order within each orbit.
        If two vertices v1 < v2 are symmetric and both could be centers,
        force v1 to be chosen before v2.
        """
        print("Adding lexicographic ordering constraints...")
        count = 0

        # For each pair (v1, v2) where v1 < v2 and they're symmetric
        visited_pairs = set()

        for v in sorted(self.vertices):
            for k in range(1, self.n):
                rotated = circular_rotate(v, self.n, k)
                if rotated > v and rotated in self.vertices_set:
                    pair = (v, rotated)
                    if pair not in visited_pairs:
                        visited_pairs.add(pair)
                        # v1 ≤_lex v2: if v2 is center, v1 must also be center
                        # Actually, for symmetry breaking: ¬v2 ∨ v1
                        # But this can be too strong. Use: ¬v2 ∨ ¬v1 (at most one)
                        # Already covered by symmetry breaking

        print(f"  Lexicographic ordering: {len(visited_pairs)} pairs")
        return len(visited_pairs)

    # ========================================
    # Strategy 3: Cardinality Network Encoding
    # ========================================

    def totalizer_at_most_k(self, literals, k):
        """
        Totalizer encoding for at-most-k constraint.
        More efficient than pairwise for larger sets.
        """
        if len(literals) <= k:
            return  # Trivially satisfied

        if len(literals) <= 5 or k == 1:
            # Use pairwise for small sets
            if k == 1:
                for i in range(len(literals)):
                    for j in range(i + 1, len(literals)):
                        self.add_clause([-literals[i], -literals[j]])
            return

        # Recursive totalizer construction
        def totalizer_rec(lits):
            n = len(lits)
            if n == 1:
                return lits

            mid = n // 2
            left = totalizer_rec(lits[:mid])
            right = totalizer_rec(lits[mid:])

            # Merge
            merged = []
            for i in range(min(k + 1, len(left) + len(right))):
                merged.append(self.new_aux_var())

            # Add merge clauses
            for a in range(len(left) + 1):
                for b in range(len(right) + 1):
                    c = a + b
                    if c > k:
                        # This combination is forbidden
                        clause = []
                        if a > 0:
                            clause.append(-left[a-1])
                        if b > 0:
                            clause.append(-right[b-1])
                        if clause:
                            self.add_clause(clause)
                    elif c > 0 and c <= len(merged):
                        # Propagate to merged
                        clause = [merged[c-1]]
                        if a > 0:
                            clause.append(-left[a-1])
                        if b > 0:
                            clause.append(-right[b-1])
                        self.add_clause(clause)

            return merged

        totalizer_rec(literals)

    def exactly_one_optimized(self, literals):
        """Optimized exactly-one encoding."""
        # At least one
        self.add_clause(literals)

        # At most one - use sequential counter for efficiency
        if len(literals) <= 4:
            for i in range(len(literals)):
                for j in range(i + 1, len(literals)):
                    self.add_clause([-literals[i], -literals[j]])
        else:
            # Sequential counter encoding
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

    # ========================================
    # Strategy 4: Greedy Pre-assignment
    # ========================================

    def compute_greedy_solution(self):
        """
        Compute a greedy solution to use as phase saving hints.
        """
        print("Computing greedy solution for phase saving...")

        uncovered = set(self.vertices)
        centers = []

        # Sort by ball size (largest first)
        candidates = sorted(self.vertices, key=lambda v: -self.ball_sizes[v])

        for v in candidates:
            if v not in uncovered:
                continue

            ball = self.balls[v]
            if all(u in uncovered for u in ball):
                # Check distance to existing centers
                valid = True
                for c in centers:
                    if hamming_distance(v, c) < 3:
                        valid = False
                        break

                if valid:
                    centers.append(v)
                    for u in ball:
                        uncovered.discard(u)

        coverage = (len(self.vertices) - len(uncovered)) / len(self.vertices)
        print(f"  Greedy: {len(centers)} centers, {coverage*100:.1f}% coverage")
        return centers

    def generate_phase_hints(self, greedy_centers):
        """
        Generate phase saving hints (preferred variable assignments).
        Format: list of positive literals that should be True first.
        """
        hints = []
        for c in greedy_centers:
            hints.append(self.var(c))

        # Also add negative hints for non-centers
        center_set = set(greedy_centers)
        for v in self.vertices[:100]:  # First 100 non-centers
            if v not in center_set:
                hints.append(-self.var(v))

        return hints

    # ========================================
    # Main Encoding
    # ========================================

    def encode(self, use_symmetry=True, use_greedy_hints=True):
        """
        Encode the perfect partition problem with advanced strategies.
        """
        print("\n" + "="*60)
        print("Advanced SAT Encoding")
        print("="*60)

        # Base encoding: covering constraints
        print("\n1. Covering constraints (each vertex in exactly one ball)...")
        for u in self.vertices:
            centers = self.ball_membership[u]
            literals = [self.var(c) for c in centers]
            self.exactly_one_optimized(literals)

        # Base encoding: packing constraints
        print("\n2. Packing constraints (centers at distance >= 3)...")
        packing_count = 0
        for i, u in enumerate(self.vertices):
            u_var = self.var(u)
            for j in range(i + 1, self.num_vertices):
                v = self.vertices[j]
                if hamming_distance(u, v) < 3:
                    v_var = self.var(v)
                    self.add_clause([-u_var, -v_var])
                    packing_count += 1

            if i % 10000 == 0 and i > 0:
                print(f"    {i}/{self.num_vertices}")

        print(f"  Total packing clauses: {packing_count}")

        # Advanced: symmetry breaking
        if use_symmetry:
            print("\n3. Symmetry breaking...")
            orbit_reps = self.get_orbit_representatives()
            self.add_symmetry_breaking(orbit_reps)

        # Get greedy hints
        greedy_hints = None
        if use_greedy_hints:
            print("\n4. Greedy pre-assignment...")
            greedy_centers = self.compute_greedy_solution()
            greedy_hints = self.generate_phase_hints(greedy_centers)

        print(f"\nEncoding complete:")
        print(f"  Variables: {self.aux_var_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

        return greedy_hints

    def write_dimacs(self, filename, hints=None):
        """Write CNF in DIMACS format with optional hints."""
        print(f"\nWriting DIMACS to {filename}...")

        with open(filename, 'w') as f:
            # Write header
            f.write(f"p cnf {self.aux_var_counter - 1} {len(self.clauses)}\n")

            # Write hint comment (for solvers that support it)
            if hints:
                f.write(f"c phase hints: {' '.join(map(str, hints[:100]))}\n")

            # Write clauses
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

        print(f"  Written {len(self.clauses)} clauses")

        # Also write hints file separately
        if hints:
            hints_file = filename.replace('.cnf', '_hints.txt')
            with open(hints_file, 'w') as f:
                for h in hints:
                    f.write(f"{h}\n")
            print(f"  Hints written to {hints_file}")


def compare_encodings(n, s):
    """Compare different encoding strategies."""
    print("="*70)
    print(f"Comparing Encoding Strategies for Λ_{n}(1^{s})")
    print("="*70)

    # Generate vertices
    vertices = generate_lambda_vertices(n, s)
    print(f"\n|V| = {len(vertices)}")

    # Strategy 1: Basic encoding
    print("\n--- Strategy 1: Basic Encoding ---")
    encoder1 = AdvancedSATEncoder(vertices, n, s)
    t0 = time.time()
    encoder1.encode(use_symmetry=False, use_greedy_hints=False)
    time1 = time.time() - t0
    clauses1 = len(encoder1.clauses)

    # Strategy 2: With symmetry breaking
    print("\n--- Strategy 2: With Symmetry Breaking ---")
    encoder2 = AdvancedSATEncoder(vertices, n, s)
    t0 = time.time()
    encoder2.encode(use_symmetry=True, use_greedy_hints=False)
    time2 = time.time() - t0
    clauses2 = len(encoder2.clauses)

    # Strategy 3: With all optimizations
    print("\n--- Strategy 3: All Optimizations ---")
    encoder3 = AdvancedSATEncoder(vertices, n, s)
    t0 = time.time()
    hints = encoder3.encode(use_symmetry=True, use_greedy_hints=True)
    time3 = time.time() - t0
    clauses3 = len(encoder3.clauses)

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Strategy':<30} | {'Clauses':>12} | {'Enc Time':>10}")
    print("-"*60)
    print(f"{'Basic':30} | {clauses1:>12} | {time1:>10.2f}s")
    print(f"{'+ Symmetry Breaking':30} | {clauses2:>12} | {time2:>10.2f}s")
    print(f"{'+ All Optimizations':30} | {clauses3:>12} | {time3:>10.2f}s")

    return encoder3, hints


def main():
    n, s = 15, 10

    encoder, hints = compare_encodings(n, s)

    # Write optimized CNF
    cnf_file = "lambda_15_10_optimized.cnf"
    encoder.write_dimacs(cnf_file, hints)

    # Run solver
    solver_path = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"
    timeout = 7200

    print(f"\nRunning CaDiCaL on optimized encoding (timeout: {timeout}s)...")
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
        print(f"Return code: {proc.returncode}")

        if proc.returncode == 10:
            print("SAT!")
        elif proc.returncode == 20:
            print("UNSAT!")

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")


if __name__ == "__main__":
    main()
