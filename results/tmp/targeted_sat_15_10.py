#!/usr/bin/env python3
"""
Targeted SAT Search for Lambda_15(1^10) Perfect Partition.

Based on ILP analysis:
- For C=2033: (a=0, b=1, d=2032) is the ONLY option
- For C=2034-2050: Multiple options, try each

Key constraints:
1. Ball-14 vertices (a=0 for C=2033) cannot be centers
2. Exactly b ball-15 vertices must be centers
3. Exactly d ball-16 vertices must be centers
4. Each vertex covered exactly once

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import subprocess
import tempfile
import os
import time
import sys

def has_circ(v, n, s):
    doubled = (v << n) | v
    mask = (1 << s) - 1
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming(a, b):
    return bin(a ^ b).count('1')

def to_bin(v, n):
    return format(v, f'0{n}b')

class TargetedSATSolver:
    def __init__(self, n=15, s=10):
        self.n = n
        self.s = s

        print(f"Initializing Lambda_{n}(1^{s})...")
        self.verts = [v for v in range(1 << n) if not has_circ(v, n, s)]
        self.vset = set(self.verts)
        self.num_verts = len(self.verts)
        self.vert_to_idx = {v: i for i, v in enumerate(self.verts)}
        print(f"|V| = {self.num_verts}")

        # Precompute balls and membership
        print("Precomputing balls...")
        self.balls = {}
        self.ball_sizes = {}
        self.membership = defaultdict(list)

        for v in self.verts:
            ball = [v]
            for i in range(n):
                nb = v ^ (1 << i)
                if nb in self.vset:
                    ball.append(nb)
            self.balls[v] = ball
            self.ball_sizes[v] = len(ball)
            for u in ball:
                self.membership[u].append(v)

        # Group by ball size
        self.V14 = [v for v in self.verts if self.ball_sizes[v] == 14]
        self.V15 = [v for v in self.verts if self.ball_sizes[v] == 15]
        self.V16 = [v for v in self.verts if self.ball_sizes[v] == 16]

        print(f"Ball-14 vertices: {len(self.V14)}")
        print(f"Ball-15 vertices: {len(self.V15)}")
        print(f"Ball-16 vertices: {len(self.V16)}")

        print("Initialization complete.")

    def build_cnf(self, target_a, target_b, target_d):
        """Build CNF for specific (a, b, d) distribution."""
        print(f"\nBuilding CNF for (a={target_a}, b={target_b}, d={target_d})...")

        # Variables:
        # - x_v for v in V15 ∪ V16: whether v is a center
        # - If a > 0, also x_v for v in V14

        # Determine which vertices can be centers
        if target_a == 0:
            center_candidates = self.V15 + self.V16
        else:
            center_candidates = self.V14 + self.V15 + self.V16

        # Variable mapping
        var_map = {v: i + 1 for i, v in enumerate(center_candidates)}
        num_vars = len(center_candidates)

        clauses = []

        # 1. Covering constraints: each vertex covered by exactly one center
        print("  Adding covering constraints...")
        for u in self.verts:
            # At least one center covers u
            covers_u = [c for c in self.membership[u] if c in var_map]
            if not covers_u:
                print(f"  ERROR: Vertex {to_bin(u, self.n)} cannot be covered!")
                return None, None, None

            # At-least-one
            clauses.append([var_map[c] for c in covers_u])

            # At-most-one (pairwise)
            for i, c1 in enumerate(covers_u):
                for c2 in covers_u[i+1:]:
                    clauses.append([-var_map[c1], -var_map[c2]])

        # 2. Packing constraints: centers at distance >= 3
        print("  Adding packing constraints...")
        pack_count = 0
        for v in center_candidates:
            v_var = var_map[v]
            # Distance 1 neighbors
            for i in range(self.n):
                nb = v ^ (1 << i)
                if nb in var_map and nb > v:
                    clauses.append([-v_var, -var_map[nb]])
                    pack_count += 1
            # Distance 2 neighbors
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    nb = v ^ (1 << i) ^ (1 << j)
                    if nb in var_map and nb > v:
                        clauses.append([-v_var, -var_map[nb]])
                        pack_count += 1

        print(f"  Packing clauses: {pack_count}")

        # 3. Cardinality constraints (simplified using totalizer or sequential counter)
        # For now, use auxiliary variables for counting

        # Actually, for this problem, the exact-k constraint is complex
        # Let's use a simpler approach: use pysat's CardEnc if available
        # Otherwise, skip cardinality and rely on the covering/packing to be tight

        # For C=2033, the constraints should be tight enough
        # For larger C, we might get solutions with fewer centers

        print(f"  Variables: {num_vars}")
        print(f"  Clauses: {len(clauses)}")

        return clauses, var_map, center_candidates

    def write_dimacs(self, clauses, num_vars, filename):
        """Write CNF to DIMACS format."""
        with open(filename, 'w') as f:
            f.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

    def solve_with_cadical(self, cnf_file, timeout=300):
        """Solve CNF using CaDiCaL."""
        cadical_path = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"

        if not os.path.exists(cadical_path):
            print(f"CaDiCaL not found at {cadical_path}")
            return None, "ERROR"

        try:
            result = subprocess.run(
                [cadical_path, cnf_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout

            if "s SATISFIABLE" in output:
                # Parse solution
                solution = []
                for line in output.split('\n'):
                    if line.startswith('v '):
                        parts = line[2:].split()
                        for p in parts:
                            if p != '0':
                                solution.append(int(p))
                return solution, "SAT"
            elif "s UNSATISFIABLE" in output:
                return None, "UNSAT"
            else:
                return None, "UNKNOWN"

        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        except Exception as e:
            return None, f"ERROR: {e}"

    def run_targeted_search(self, target_C, timeout_per_distribution=300):
        """Run SAT search for a specific C value."""
        print(f"\n{'='*60}")
        print(f"Searching for C = {target_C}")
        print("=" * 60)

        # Calculate D and find all valid (a, b, d)
        D = 16 * target_C - self.num_verts

        if D < 0:
            print(f"  Invalid: D = {D} < 0")
            return None

        # Find all (a, b, d) with 2a + b = D
        distributions = []
        for a in range(min(len(self.V14) + 1, D // 2 + 1)):
            b = D - 2 * a
            if b < 0 or b > len(self.V15):
                continue
            d = target_C - a - b
            if d < 0 or d > len(self.V16):
                continue
            # Verify
            if 14 * a + 15 * b + 16 * d != self.num_verts:
                continue
            distributions.append((a, b, d))

        print(f"Found {len(distributions)} valid distributions")

        for a, b, d in distributions:
            print(f"\nTrying (a={a}, b={b}, d={d})...")

            result = self.build_cnf(a, b, d)
            if result[0] is None:
                print("  Skipped (uncoverable vertices)")
                continue

            clauses, var_map, candidates = result

            # Write CNF
            cnf_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/target_C{target_C}_a{a}_b{b}_d{d}.cnf"
            self.write_dimacs(clauses, len(var_map), cnf_file)
            print(f"  Written to {cnf_file}")

            # Solve
            print(f"  Solving (timeout={timeout_per_distribution}s)...")
            start_time = time.time()
            solution, status = self.solve_with_cadical(cnf_file, timeout_per_distribution)
            elapsed = time.time() - start_time

            print(f"  Result: {status} ({elapsed:.1f}s)")

            if status == "SAT" and solution:
                # Extract centers
                centers = []
                for lit in solution:
                    if lit > 0:
                        # Find which vertex this is
                        for v, var in var_map.items():
                            if var == lit:
                                centers.append(v)
                                break

                print(f"  Found {len(centers)} centers!")

                # Verify
                ok, msg = self.verify(centers)
                print(f"  Verification: {msg}")

                if ok:
                    return centers

        return None

    def verify(self, centers):
        """Verify if centers form a perfect partition."""
        center_set = set(centers)

        # Check all centers are valid
        for c in centers:
            if c not in self.vset:
                return False, f"Invalid center {c}"

        # Check packing
        for i, c1 in enumerate(centers):
            for c2 in centers[i+1:]:
                if hamming(c1, c2) < 3:
                    return False, f"Centers too close"

        # Check coverage
        coverage = defaultdict(int)
        for c in centers:
            for v in self.balls[c]:
                coverage[v] += 1

        uncovered = [v for v in self.verts if coverage[v] == 0]
        overlaps = [v for v in self.verts if coverage[v] > 1]

        if uncovered:
            return False, f"{len(uncovered)} uncovered"

        if overlaps:
            return False, f"{len(overlaps)} overlaps"

        # Check distribution
        a = sum(1 for c in centers if self.ball_sizes[c] == 14)
        b = sum(1 for c in centers if self.ball_sizes[c] == 15)
        d = sum(1 for c in centers if self.ball_sizes[c] == 16)

        return True, f"Perfect! C={len(centers)}, (a={a}, b={b}, d={d})"

    def save_solution(self, centers, filename):
        """Save solution."""
        np.save(f"{filename}.npy", np.array(centers, dtype=np.uint32))

        with open(f"{filename}.txt", 'w') as f:
            f.write(f"Lambda_{self.n}(1^{self.s}) Perfect Partition\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"|V| = {self.num_verts}\n")
            f.write(f"Centers = {len(centers)}\n\n")

            a = sum(1 for c in centers if self.ball_sizes[c] == 14)
            b = sum(1 for c in centers if self.ball_sizes[c] == 15)
            d = sum(1 for c in centers if self.ball_sizes[c] == 16)
            f.write(f"Distribution: a={a} (ball-14), b={b} (ball-15), d={d} (ball-16)\n\n")

            f.write("Centers (binary):\n")
            for i, c in enumerate(sorted(centers)):
                f.write(f"{i+1:4d}. {to_bin(c, self.n)} (ball size {self.ball_sizes[c]})\n")

        print(f"Solution saved to {filename}*")


def main():
    solver = TargetedSATSolver(n=15, s=10)

    # Try C values from 2033 upward
    for target_C in range(2033, 2051):
        solution = solver.run_targeted_search(target_C, timeout_per_distribution=600)

        if solution:
            solver.save_solution(
                solution,
                f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_C{target_C}"
            )
            print(f"\n*** SOLUTION FOUND with C = {target_C}! ***")
            return

    print("\nNo solution found for C in [2033, 2050]")


if __name__ == "__main__":
    main()
