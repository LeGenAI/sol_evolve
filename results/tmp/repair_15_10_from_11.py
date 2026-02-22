#!/usr/bin/env python3
"""
Repair Lambda_15(1^10) Perfect Partition from Lambda_15(1^11) Solution.

Strategy:
1. Load s=11 solution (2047 centers)
2. Identify centers violating s=10 constraint (have 10+ consecutive circular 1s)
3. Find uncovered vertices after removing bad centers
4. Use SAT to find replacement centers

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import time
import subprocess
import os

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
    """Compute Hamming distance between two integers."""
    return bin(a ^ b).count('1')

def generate_lambda_vertices(n, s):
    """Generate all vertices of Lambda_n(1^s)."""
    return [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]

def get_ball(center, n, vertices_set):
    """Get closed ball of radius 1 around center in the graph."""
    ball = [center]
    for i in range(n):
        neighbor = center ^ (1 << i)
        if neighbor in vertices_set:
            ball.append(neighbor)
    return ball

def to_binary(v, n):
    """Convert integer to n-bit binary string."""
    return format(v, f'0{n}b')

# ============================================================================
# Step 1: Load and Analyze s=11 Solution
# ============================================================================

def load_s11_solution():
    """Load Lambda_15(1^11) solution."""
    # Try loading from numpy file first
    npy_path = '/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_11_centers.npy'
    if os.path.exists(npy_path):
        centers = np.load(npy_path)
        return list(centers)

    # Fallback: parse from text file
    txt_path = '/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_11_centers.txt'
    centers = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Parse lines like "   1. 000000000000000  (ball size: 16)"
            if '.' in line and '(' in line:
                parts = line.split('.')
                if len(parts) >= 2:
                    binary_part = parts[1].strip().split()[0]
                    if len(binary_part) == 15 and all(c in '01' for c in binary_part):
                        centers.append(int(binary_part, 2))
    return centers

def analyze_s11_solution(s11_centers, n=15, s_new=10):
    """Analyze which s=11 centers violate s=10 constraint."""
    print("=" * 60)
    print("Step 1: Analyzing s=11 Solution for s=10 Compatibility")
    print("=" * 60)

    good_centers = []
    bad_centers = []

    for c in s11_centers:
        if has_circular_consecutive_ones(c, n, s_new):
            bad_centers.append(c)
        else:
            good_centers.append(c)

    print(f"\nTotal s=11 centers: {len(s11_centers)}")
    print(f"Good centers (valid for s=10): {len(good_centers)}")
    print(f"Bad centers (violate s=10): {len(bad_centers)}")

    # Analyze bad centers
    if bad_centers:
        print(f"\nBad centers analysis:")
        weight_dist = Counter(bin(c).count('1') for c in bad_centers)
        for w in sorted(weight_dist.keys()):
            print(f"  Weight {w}: {weight_dist[w]} centers")

        print(f"\nSample bad centers:")
        for c in bad_centers[:10]:
            print(f"  {to_binary(c, n)} (weight {bin(c).count('1')})")

    return good_centers, bad_centers

# ============================================================================
# Step 2: Analyze Repair Requirements
# ============================================================================

def analyze_repair_requirements(good_centers, bad_centers, n=15, s_new=10):
    """Find uncovered vertices and potential repair candidates."""
    print("\n" + "=" * 60)
    print("Step 2: Analyzing Repair Requirements")
    print("=" * 60)

    # Generate Lambda_15(1^10) vertices
    print("\nGenerating Lambda_15(1^10) vertices...")
    vertices_s10 = generate_lambda_vertices(n, s_new)
    vertices_s10_set = set(vertices_s10)
    print(f"|V(Lambda_15(1^10))| = {len(vertices_s10)}")

    # Compute coverage by good centers
    print("\nComputing coverage by good centers...")
    covered_by_good = set()
    good_set = set(good_centers)

    for c in good_centers:
        ball = get_ball(c, n, vertices_s10_set)
        covered_by_good.update(ball)

    # Find uncovered vertices
    uncovered = vertices_s10_set - covered_by_good
    print(f"Vertices covered by good centers: {len(covered_by_good)}")
    print(f"Uncovered vertices: {len(uncovered)}")

    # Analyze what bad centers were covering
    print("\nAnalyzing what bad centers covered...")
    bad_covered = set()
    for c in bad_centers:
        ball = get_ball(c, n, vertices_s10_set)
        bad_covered.update(ball)

    # Critical: vertices covered ONLY by bad centers (not by any good center)
    critical_uncovered = bad_covered - covered_by_good
    print(f"Vertices covered by bad centers: {len(bad_covered)}")
    print(f"Critical uncovered (only covered by bad): {len(critical_uncovered)}")

    # Note: uncovered and critical_uncovered might differ slightly
    # because some bad center balls might include vertices outside Lambda_15(1^10)
    # or vertices already covered by good centers
    # The key metric is 'uncovered' which is what we need to cover
    if critical_uncovered != uncovered:
        print(f"Note: Slight mismatch - using uncovered ({len(uncovered)}) as target")

    # Find repair candidates: vertices in s10 that can cover uncovered
    print("\nFinding repair candidates...")
    repair_candidates = []

    for v in vertices_s10:
        if v in good_set:
            continue  # Already a good center
        ball = get_ball(v, n, vertices_s10_set)
        covers_uncovered = any(u in uncovered for u in ball)
        if covers_uncovered:
            repair_candidates.append(v)

    print(f"Repair candidates: {len(repair_candidates)}")

    # Filter by ball size
    candidate_balls = {}
    for v in repair_candidates:
        ball = get_ball(v, n, vertices_s10_set)
        candidate_balls[v] = ball

    ball_sizes = Counter(len(b) for b in candidate_balls.values())
    print(f"Candidate ball size distribution:")
    for size in sorted(ball_sizes.keys()):
        print(f"  Size {size}: {ball_sizes[size]}")

    return vertices_s10, vertices_s10_set, uncovered, repair_candidates, candidate_balls

# ============================================================================
# Step 3: Build and Run Local SAT Repair
# ============================================================================

class RepairSATEncoder:
    """SAT encoder for repair problem."""

    def __init__(self, good_centers, repair_candidates, uncovered, vertices_s10_set, n):
        self.good_centers = good_centers
        self.good_set = set(good_centers)
        self.candidates = repair_candidates
        self.uncovered = uncovered
        self.vertices_set = vertices_s10_set
        self.n = n
        self.clauses = []

        # Variable mapping: candidate -> SAT variable
        self.var_map = {c: i + 1 for i, c in enumerate(self.candidates)}
        self.num_vars = len(self.candidates)
        self.aux_counter = self.num_vars + 1

        # Precompute balls for candidates
        self.candidate_balls = {}
        self.ball_membership = defaultdict(list)  # vertex -> candidates covering it

        for c in self.candidates:
            ball = get_ball(c, self.n, self.vertices_set)
            self.candidate_balls[c] = ball
            for v in ball:
                self.ball_membership[v].append(c)

    def new_aux(self):
        var = self.aux_counter
        self.aux_counter += 1
        return var

    def add_clause(self, lits):
        self.clauses.append(lits)

    def at_most_one_sequential(self, lits):
        """Sequential counter encoding for at-most-one."""
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
        """Build SAT encoding for repair problem."""
        print("\n" + "=" * 60)
        print("Step 3: Building SAT Encoding for Repair")
        print("=" * 60)

        # Constraint 1: Cover all uncovered vertices exactly once
        print("\nEncoding covering constraints for uncovered vertices...")
        uncoverable = 0

        for v in self.uncovered:
            # Find candidates that can cover v
            covering_candidates = [c for c in self.ball_membership[v]
                                   if c in self.var_map]

            if not covering_candidates:
                uncoverable += 1
                print(f"  WARNING: {to_binary(v, self.n)} cannot be covered!")
                continue

            lits = [self.var_map[c] for c in covering_candidates]
            self.add_clause(lits)  # At least one
            self.at_most_one_sequential(lits)  # At most one

        if uncoverable > 0:
            print(f"\nERROR: {uncoverable} vertices cannot be covered - UNSAT!")
            return False

        # Constraint 2: Packing - new centers at distance >= 3 from each other
        print("\nEncoding packing constraints (new centers)...")
        pack_count = 0

        for i, c1 in enumerate(self.candidates):
            v1 = self.var_map[c1]
            for c2 in self.candidates[i+1:]:
                if hamming_distance(c1, c2) < 3:
                    v2 = self.var_map[c2]
                    self.add_clause([-v1, -v2])
                    pack_count += 1

        print(f"  Packing clauses (between candidates): {pack_count}")

        # Constraint 3: Packing - new centers at distance >= 3 from good centers
        print("\nEncoding packing constraints (new vs good centers)...")
        good_pack_count = 0

        for c1 in self.candidates:
            v1 = self.var_map[c1]
            for c2 in self.good_centers:
                if hamming_distance(c1, c2) < 3:
                    self.add_clause([-v1])  # Cannot be a center
                    good_pack_count += 1
                    break  # Only need one violation to block

        print(f"  Blocked candidates (too close to good): {good_pack_count}")

        # Constraint 4: No double coverage of already covered vertices
        print("\nEncoding no-double-coverage constraints...")
        double_count = 0

        covered_by_good = set()
        for c in self.good_centers:
            ball = get_ball(c, self.n, self.vertices_set)
            covered_by_good.update(ball)

        for v in covered_by_good:
            # v is already covered by a good center
            # No candidate should also cover v
            covering_candidates = [c for c in self.ball_membership[v]
                                   if c in self.var_map]
            if covering_candidates:
                # At most zero candidates can cover v (i.e., none)
                for c in covering_candidates:
                    self.add_clause([-self.var_map[c]])
                    double_count += 1

        print(f"  Blocked by double-coverage: {double_count}")

        print(f"\nTotal encoding:")
        print(f"  Variables: {self.aux_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

        return True

    def write_dimacs(self, filename):
        """Write CNF to DIMACS format."""
        with open(filename, 'w') as f:
            f.write(f"c Lambda_15(1^10) Repair SAT Encoding\n")
            f.write(f"c Candidates: {len(self.candidates)}\n")
            f.write(f"c Uncovered: {len(self.uncovered)}\n")
            f.write(f"p cnf {self.aux_counter - 1} {len(self.clauses)}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        print(f"Written to {filename}")

    def parse_solution(self, output):
        """Parse SAT solver output to get new centers."""
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

def run_repair_sat(encoder, timeout=14400):
    """Run SAT solver on repair encoding."""
    print("\n" + "=" * 60)
    print("Running SAT Solver for Repair")
    print("=" * 60)

    cnf_file = "/Users/baegjaehyeon/CodeEvolve/results/tmp/repair_15_10.cnf"
    encoder.write_dimacs(cnf_file)

    solver_path = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"

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
            print("\nSAT! Parsing solution...")
            new_centers = encoder.parse_solution(proc.stdout)
            return new_centers
        elif proc.returncode == 20:
            print("\nUNSAT - Repair not possible with current constraints")
            return None
        else:
            print(f"\nUnknown result (code {proc.returncode})")
            return None

    except subprocess.TimeoutExpired:
        print(f"\nTimeout after {timeout}s")
        return None

# ============================================================================
# Step 4: Verification
# ============================================================================

def verify_solution(centers, n=15, s=10):
    """Verify the complete solution."""
    print("\n" + "=" * 60)
    print("Step 4: Verification")
    print("=" * 60)

    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    print(f"\nSolution: {len(centers)} centers")
    print(f"Target: {len(vertices)} vertices to cover")

    # Check all centers are valid (no s consecutive 1s)
    invalid_centers = [c for c in centers if has_circular_consecutive_ones(c, n, s)]
    if invalid_centers:
        print(f"\nERROR: {len(invalid_centers)} centers violate s={s} constraint!")
        return False
    print(f"All centers valid for s={s}: OK")

    # Check pairwise distances
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

    print(f"\nCoverage:")
    print(f"  Covered: {len(covered)}/{len(vertices)}")
    print(f"  Uncovered: {len(uncovered)}")
    print(f"  Overlaps: {overlaps}")

    print(f"\nBall sizes: {dict(Counter(ball_sizes))}")

    if len(uncovered) == 0 and overlaps == 0 and min_dist >= 3:
        print("\n" + "=" * 60)
        print("PERFECT PARTITION VERIFIED!")
        print("=" * 60)
        return True
    else:
        print("\nNOT a perfect partition")
        return False

def save_solution(centers, n=15, s=10):
    """Save solution to files."""
    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    # Save numpy
    np.save('/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_centers.npy',
            np.array(centers))

    # Save text
    with open('/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_centers.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Perfect Partition of Lambda_{n}(1^{s})\n")
        f.write("=" * 60 + "\n\n")

        f.write("DISCOVERY SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Date: 2025-11-27\n")
        f.write(f"Author: Jae-Hyun Baek (with Claude Code)\n")
        f.write(f"Method: s=11 Solution Repair\n")
        f.write(f"Status: VERIFIED\n\n")

        f.write("BASIC STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"n = {n}, s = {s}\n")
        f.write(f"Vertex count |V| = {len(vertices)}\n")
        f.write(f"Center count = {len(centers)}\n")

        # Compute stats
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

    print("\nSolution saved to:")
    print("  lambda_15_10_centers.npy")
    print("  lambda_15_10_centers.txt")

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Lambda_15(1^10) Perfect Partition Search")
    print("Strategy: Repair from Lambda_15(1^11) Solution")
    print("=" * 70)

    n, s_old, s_new = 15, 11, 10

    # Step 1: Load and analyze s=11 solution
    print("\nLoading s=11 solution...")
    s11_centers = load_s11_solution()
    print(f"Loaded {len(s11_centers)} centers")

    good_centers, bad_centers = analyze_s11_solution(s11_centers, n, s_new)

    if not bad_centers:
        print("\nNo bad centers! s=11 solution is already valid for s=10.")
        if verify_solution(s11_centers, n, s_new):
            save_solution(s11_centers, n, s_new)
        return

    # Step 2: Analyze repair requirements
    vertices_s10, vertices_s10_set, uncovered, repair_candidates, _ = \
        analyze_repair_requirements(good_centers, bad_centers, n, s_new)

    if not uncovered:
        print("\nNo uncovered vertices! Good centers already cover everything.")
        if verify_solution(good_centers, n, s_new):
            save_solution(good_centers, n, s_new)
        return

    # Step 3: Build and run SAT repair
    encoder = RepairSATEncoder(good_centers, repair_candidates, uncovered,
                               vertices_s10_set, n)

    if not encoder.encode():
        print("\nEncoding failed - problem is UNSAT by construction")
        return

    new_centers = run_repair_sat(encoder, timeout=14400)  # 4 hours

    if new_centers is None:
        print("\nRepair failed. Consider:")
        print("  1. Expanding repair radius")
        print("  2. Trying greedy + SAT hybrid")
        print("  3. Full SAT search with longer timeout")
        return

    # Combine good centers with new centers
    final_centers = good_centers + new_centers
    print(f"\nFinal solution: {len(final_centers)} centers")
    print(f"  Good centers: {len(good_centers)}")
    print(f"  New centers: {len(new_centers)}")

    # Step 4: Verify and save
    if verify_solution(final_centers, n, s_new):
        save_solution(final_centers, n, s_new)
    else:
        print("\nVerification failed!")

if __name__ == "__main__":
    main()
