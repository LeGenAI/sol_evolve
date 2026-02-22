#!/usr/bin/env python3
"""
Greedy + SAT Hybrid for Lambda_15(1^10) Perfect Partition.

Strategy:
1. Use greedy algorithm to select ~80% of centers
2. Use SAT to complete the remaining coverage

This bypasses the s=11 repair problem which proved UNSAT.

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import time
import subprocess
import random

# ============================================================================
# Core Functions
# ============================================================================

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

# ============================================================================
# Greedy Center Selection
# ============================================================================

def greedy_center_selection(vertices, n, target_coverage=0.95, max_centers=2100):
    """
    Greedy algorithm to select centers covering maximum vertices.

    Priority: larger ball size first, then random tiebreaker
    """
    print("=" * 60)
    print(f"Greedy Center Selection (target: {target_coverage*100:.0f}%)")
    print("=" * 60)

    vertices_set = set(vertices)

    # Precompute balls and sizes
    print("\nPrecomputing balls...")
    balls = {}
    ball_sizes = {}
    for v in vertices:
        ball = get_ball(v, n, vertices_set)
        balls[v] = ball
        ball_sizes[v] = len(ball)

    # Sort candidates by ball size (descending)
    candidates = sorted(vertices, key=lambda v: -ball_sizes[v])

    print(f"Ball size distribution:")
    size_dist = Counter(ball_sizes.values())
    for s in sorted(size_dist.keys(), reverse=True):
        print(f"  Size {s}: {size_dist[s]} vertices")

    # Greedy selection
    print("\nRunning greedy selection...")
    selected = []
    covered = set()
    blocked = set()  # Vertices that can't be centers (too close to selected)

    target_count = int(len(vertices) * target_coverage)

    for i, c in enumerate(candidates):
        if len(covered) >= target_count or len(selected) >= max_centers:
            break

        if c in blocked:
            continue

        ball = balls[c]

        # Check if this would cause overlap
        if any(v in covered for v in ball):
            continue

        # Check distance from existing centers
        too_close = False
        for existing in selected:
            if hamming_distance(c, existing) < 3:
                too_close = True
                break

        if too_close:
            continue

        # Add this center
        selected.append(c)
        covered.update(ball)

        # Block vertices too close to this center
        for v in vertices:
            if hamming_distance(c, v) < 3:
                blocked.add(v)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(candidates)}, "
                  f"selected {len(selected)}, covered {len(covered)}/{len(vertices)}")

    print(f"\nGreedy result:")
    print(f"  Centers selected: {len(selected)}")
    print(f"  Vertices covered: {len(covered)}/{len(vertices)}")
    print(f"  Coverage: {len(covered)/len(vertices)*100:.1f}%")

    uncovered = vertices_set - covered
    print(f"  Uncovered: {len(uncovered)}")

    return selected, covered, uncovered

# ============================================================================
# SAT Completion
# ============================================================================

class CompletionEncoder:
    """SAT encoder to complete coverage after greedy."""

    def __init__(self, greedy_centers, uncovered, vertices_set, n):
        self.greedy_centers = greedy_centers
        self.greedy_set = set(greedy_centers)
        self.uncovered = uncovered
        self.vertices_set = vertices_set
        self.n = n
        self.clauses = []

        # Compute covered by greedy
        self.greedy_covered = set()
        for c in greedy_centers:
            ball = get_ball(c, n, vertices_set)
            self.greedy_covered.update(ball)

        # Find candidates: can cover uncovered, valid, not already selected
        print("\nFinding completion candidates...")
        self.candidates = []
        self.candidate_balls = {}
        self.ball_membership = defaultdict(list)

        for v in self.vertices_set:
            if v in self.greedy_set:
                continue
            ball = get_ball(v, n, vertices_set)

            # Check if ball overlaps with greedy coverage
            if any(u in self.greedy_covered for u in ball):
                continue

            # Check distance from greedy centers
            too_close = False
            for gc in greedy_centers:
                if hamming_distance(v, gc) < 3:
                    too_close = True
                    break
            if too_close:
                continue

            # Check if covers any uncovered vertex
            covers = [u for u in ball if u in uncovered]
            if covers:
                self.candidates.append(v)
                self.candidate_balls[v] = ball
                for u in covers:
                    self.ball_membership[u].append(v)

        print(f"Completion candidates: {len(self.candidates)}")

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
        print("Building SAT Encoding for Completion")
        print("=" * 60)

        # Check coverability
        uncoverable = []
        for v in self.uncovered:
            if not self.ball_membership[v]:
                uncoverable.append(v)

        if uncoverable:
            print(f"\nWARNING: {len(uncoverable)} vertices cannot be covered!")
            return False

        # Covering constraints
        print("\nEncoding covering constraints...")
        for v in self.uncovered:
            covering = self.ball_membership[v]
            lits = [self.var_map[c] for c in covering]
            self.add_clause(lits)  # At least one
            self.at_most_one_sequential(lits)  # At most one

        # Packing constraints between candidates
        print("Encoding packing constraints...")
        pack_count = 0
        for i, c1 in enumerate(self.candidates):
            v1 = self.var_map[c1]
            for c2 in self.candidates[i+1:]:
                if hamming_distance(c1, c2) < 3:
                    v2 = self.var_map[c2]
                    self.add_clause([-v1, -v2])
                    pack_count += 1
        print(f"  Packing clauses: {pack_count}")

        print(f"\nTotal encoding:")
        print(f"  Variables: {self.aux_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

        return True

    def write_dimacs(self, filename):
        with open(filename, 'w') as f:
            f.write(f"c Lambda_15(1^10) SAT Completion\n")
            f.write(f"c Greedy centers: {len(self.greedy_centers)}\n")
            f.write(f"c Candidates: {len(self.candidates)}\n")
            f.write(f"c Uncovered: {len(self.uncovered)}\n")
            f.write(f"p cnf {self.aux_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        print(f"Written to {filename}")

    def parse_solution(self, output):
        new_centers = []
        for line in output.split('\n'):
            if line.startswith('v '):
                parts = line.split()[1:]
                for p in parts:
                    if p == '0':
                        continue
                    var = int(p)
                    if var > 0 and var <= self.num_vars:
                        new_centers.append(self.candidates[var - 1])
        return new_centers

# ============================================================================
# Verification
# ============================================================================

def verify_solution(centers, n=15, s=10):
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    print(f"\nSolution: {len(centers)} centers")
    print(f"Target: {len(vertices)} vertices")

    # Check validity
    invalid = [c for c in centers if has_circular_consecutive_ones(c, n, s)]
    if invalid:
        print(f"\nERROR: {len(invalid)} invalid centers!")
        return False
    print("All centers valid: OK")

    # Check distances
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

    print(f"Covered: {len(covered)}/{len(vertices)}")
    print(f"Uncovered: {len(uncovered)}")
    print(f"Overlaps: {overlaps}")
    print(f"Ball sizes: {dict(Counter(ball_sizes))}")

    if len(uncovered) == 0 and overlaps == 0 and min_dist >= 3:
        print("\nPERFECT PARTITION VERIFIED!")
        return True
    else:
        print("\nNOT a perfect partition")
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

        f.write("DISCOVERY SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Date: 2025-11-27\n")
        f.write(f"Author: Jae-Hyun Baek (with Claude Code)\n")
        f.write(f"Method: Greedy + SAT Hybrid\n")
        f.write(f"Status: VERIFIED\n\n")

        f.write("BASIC STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"n = {n}, s = {s}\n")
        f.write(f"Vertex count |V| = {len(vertices)}\n")
        f.write(f"Center count = {len(centers)}\n")

        ball_sizes = []
        for c in centers:
            ball = get_ball(c, n, vertices_set)
            ball_sizes.append(len(ball))
        f.write(f"Ball sizes: {dict(Counter(ball_sizes))}\n\n")

        f.write("CENTERS (binary representation)\n")
        f.write("-" * 40 + "\n")
        for i, c in enumerate(sorted(centers)):
            ball = get_ball(c, n, vertices_set)
            f.write(f"{i+1:4d}. {to_binary(c, n)}  (ball size: {len(ball)})\n")

    print("\nSolution saved!")

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Lambda_15(1^10) Perfect Partition Search")
    print("Strategy: Greedy + SAT Hybrid")
    print("=" * 70)

    n, s = 15, 10

    # Generate vertices
    print("\nGenerating Lambda_15(1^10) vertices...")
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)
    print(f"|V| = {len(vertices)}")

    # Try different coverage targets
    for target in [0.98, 0.95, 0.90, 0.85, 0.80]:
        print(f"\n{'='*70}")
        print(f"Trying greedy with target coverage = {target*100:.0f}%")
        print("=" * 70)

        greedy_centers, covered, uncovered = greedy_center_selection(
            vertices, n, target_coverage=target
        )

        if not uncovered:
            # Perfect! No SAT needed
            print("\nGreedy achieved 100% coverage!")
            if verify_solution(greedy_centers, n, s):
                save_solution(greedy_centers, n, s)
                return
            continue

        # Try SAT completion
        encoder = CompletionEncoder(greedy_centers, uncovered, vertices_set, n)

        if not encoder.encode():
            print(f"\nCompletion encoding failed for target {target}")
            continue

        cnf_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/greedy_completion_{int(target*100)}.cnf"
        encoder.write_dimacs(cnf_file)

        solver = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"
        timeout = 7200  # 2 hours

        print(f"\nRunning CaDiCaL (timeout: {timeout}s)...")
        t0 = time.time()

        try:
            proc = subprocess.run(
                [solver, cnf_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            solve_time = time.time() - t0
            print(f"Solve time: {solve_time:.1f}s")

            if proc.returncode == 10:
                print("\nSAT! Parsing solution...")
                new_centers = encoder.parse_solution(proc.stdout)

                final_centers = greedy_centers + new_centers
                print(f"Final: {len(final_centers)} centers")
                print(f"  Greedy: {len(greedy_centers)}")
                print(f"  SAT: {len(new_centers)}")

                if verify_solution(final_centers, n, s):
                    save_solution(final_centers, n, s)
                    return

            elif proc.returncode == 20:
                print(f"\nUNSAT for target {target}")
                continue

        except subprocess.TimeoutExpired:
            print(f"\nTimeout for target {target}")
            continue

    print("\n" + "=" * 70)
    print("All attempts failed. Consider full SAT search.")
    print("=" * 70)

if __name__ == "__main__":
    main()
