#!/usr/bin/env python3
"""
Expanded Repair for Lambda_15(1^10) Perfect Partition.

The local repair failed because:
- All repair candidates are either too close to good centers, OR
- Would double-cover already covered vertices

Solution: Expand the "repair zone" - allow some good centers near bad centers
to be dropped and replaced along with the bad centers.

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
# Load s=11 Solution
# ============================================================================

def load_s11_solution():
    """Load Lambda_15(1^11) solution."""
    npy_path = '/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_11_centers.npy'
    if os.path.exists(npy_path):
        centers = np.load(npy_path)
        return list(centers)

    txt_path = '/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_11_centers.txt'
    centers = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '.' in line and '(' in line:
                parts = line.split('.')
                if len(parts) >= 2:
                    binary_part = parts[1].strip().split()[0]
                    if len(binary_part) == 15 and all(c in '01' for c in binary_part):
                        centers.append(int(binary_part, 2))
    return centers

# ============================================================================
# Expanded Repair Strategy
# ============================================================================

def find_repair_zone(s11_centers, n=15, s_new=10, radius=2):
    """
    Find the "repair zone" - centers that need to be replaced.

    This includes:
    1. Bad centers (violate s=10 constraint)
    2. Good centers within 'radius' hops of bad centers in the dependency graph

    The dependency graph has an edge between centers c1, c2 if:
    - Their balls overlap (share at least one vertex)
    - They are at distance < 3 (packing conflict)
    """
    print("=" * 60)
    print(f"Finding Repair Zone (radius={radius})")
    print("=" * 60)

    vertices_s10 = generate_lambda_vertices(n, s_new)
    vertices_s10_set = set(vertices_s10)

    # Classify centers
    good_centers = []
    bad_centers = []

    for c in s11_centers:
        if has_circular_consecutive_ones(c, n, s_new):
            bad_centers.append(c)
        else:
            good_centers.append(c)

    print(f"Total centers: {len(s11_centers)}")
    print(f"Bad centers: {len(bad_centers)}")
    print(f"Good centers: {len(good_centers)}")

    # Build dependency graph (ball overlap or packing conflict)
    print("\nBuilding dependency graph...")
    s11_set = set(s11_centers)

    # Compute balls
    center_balls = {}
    for c in s11_centers:
        center_balls[c] = set(get_ball(c, n, vertices_s10_set))

    # Build adjacency: centers are adjacent if balls overlap
    adjacency = defaultdict(set)
    for i, c1 in enumerate(s11_centers):
        for c2 in s11_centers[i+1:]:
            # Check ball overlap
            if center_balls[c1] & center_balls[c2]:
                adjacency[c1].add(c2)
                adjacency[c2].add(c1)
            # Also add if at distance < 3 (would be packing conflict)
            elif hamming_distance(c1, c2) < 3:
                adjacency[c1].add(c2)
                adjacency[c2].add(c1)

    # BFS from bad centers to find repair zone
    repair_zone = set(bad_centers)
    frontier = set(bad_centers)

    for r in range(radius):
        new_frontier = set()
        for c in frontier:
            for neighbor in adjacency[c]:
                if neighbor not in repair_zone:
                    new_frontier.add(neighbor)
        repair_zone.update(new_frontier)
        frontier = new_frontier
        print(f"  Radius {r+1}: added {len(new_frontier)} centers (total: {len(repair_zone)})")

    # Fixed centers are good centers NOT in repair zone
    fixed_centers = [c for c in good_centers if c not in repair_zone]
    dropped_centers = [c for c in s11_centers if c in repair_zone]

    print(f"\nRepair zone: {len(repair_zone)} centers")
    print(f"  Bad: {len(bad_centers)}")
    print(f"  Good dropped: {len(repair_zone) - len(bad_centers)}")
    print(f"Fixed centers: {len(fixed_centers)}")

    return fixed_centers, dropped_centers, vertices_s10, vertices_s10_set

# ============================================================================
# SAT Encoding for Expanded Repair
# ============================================================================

class ExpandedRepairEncoder:
    """SAT encoder for expanded repair problem."""

    def __init__(self, fixed_centers, vertices_s10, vertices_s10_set, n, s):
        self.fixed_centers = fixed_centers
        self.fixed_set = set(fixed_centers)
        self.vertices = vertices_s10
        self.vertices_set = vertices_s10_set
        self.n = n
        self.s = s
        self.clauses = []

        # Compute coverage by fixed centers
        self.fixed_covered = set()
        for c in fixed_centers:
            ball = get_ball(c, n, vertices_s10_set)
            self.fixed_covered.update(ball)

        # Uncovered vertices that need new centers
        self.uncovered = self.vertices_set - self.fixed_covered
        print(f"\nUncovered vertices: {len(self.uncovered)}")

        # Find candidate centers for uncovered vertices
        # Candidates must:
        # 1. Be in Lambda_15(1^10) (no 10+ consecutive 1s)
        # 2. Not be a fixed center
        # 3. Cover at least one uncovered vertex
        # 4. Have ball entirely in Lambda_15(1^10)

        print("Finding repair candidates...")
        self.candidates = []
        self.candidate_balls = {}

        for v in vertices_s10:
            if v in self.fixed_set:
                continue
            ball = get_ball(v, n, vertices_s10_set)
            covers_uncovered = any(u in self.uncovered for u in ball)
            if covers_uncovered:
                self.candidates.append(v)
                self.candidate_balls[v] = ball

        print(f"Repair candidates: {len(self.candidates)}")

        # Build ball membership for uncovered vertices
        self.ball_membership = defaultdict(list)
        for c in self.candidates:
            for v in self.candidate_balls[c]:
                if v in self.uncovered:
                    self.ball_membership[v].append(c)

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
        """Build SAT encoding."""
        print("\n" + "=" * 60)
        print("Building SAT Encoding")
        print("=" * 60)

        # Check coverability first
        uncoverable = []
        for v in self.uncovered:
            if not self.ball_membership[v]:
                uncoverable.append(v)

        if uncoverable:
            print(f"\nERROR: {len(uncoverable)} vertices cannot be covered!")
            print("Sample uncoverable:")
            for v in uncoverable[:5]:
                print(f"  {to_binary(v, self.n)}")
            return False

        # Constraint 1: Cover each uncovered vertex exactly once
        print("\nEncoding covering constraints...")
        for v in self.uncovered:
            covering = self.ball_membership[v]
            lits = [self.var_map[c] for c in covering]
            self.add_clause(lits)  # At least one
            self.at_most_one_sequential(lits)  # At most one

        # Constraint 2: Packing between new centers
        print("Encoding packing constraints (new-new)...")
        pack_count = 0
        for i, c1 in enumerate(self.candidates):
            v1 = self.var_map[c1]
            for c2 in self.candidates[i+1:]:
                if hamming_distance(c1, c2) < 3:
                    v2 = self.var_map[c2]
                    self.add_clause([-v1, -v2])
                    pack_count += 1
        print(f"  Packing clauses: {pack_count}")

        # Constraint 3: Packing between new and fixed centers
        print("Encoding packing constraints (new-fixed)...")
        blocked = 0
        for c1 in self.candidates:
            v1 = self.var_map[c1]
            for c2 in self.fixed_centers:
                if hamming_distance(c1, c2) < 3:
                    self.add_clause([-v1])
                    blocked += 1
                    break
        print(f"  Blocked candidates: {blocked}")

        # Constraint 4: No overlap with fixed center balls
        print("Encoding no-overlap constraints...")
        overlap_blocked = 0
        for c in self.candidates:
            v = self.var_map[c]
            ball = set(self.candidate_balls[c])
            # Check if any vertex in ball is already covered by fixed
            if ball & self.fixed_covered:
                # This candidate would cause overlap
                self.add_clause([-v])
                overlap_blocked += 1
        print(f"  Overlap-blocked candidates: {overlap_blocked}")

        print(f"\nTotal encoding:")
        print(f"  Variables: {self.aux_counter - 1}")
        print(f"  Clauses: {len(self.clauses)}")

        return True

    def write_dimacs(self, filename):
        with open(filename, 'w') as f:
            f.write(f"c Lambda_15(1^10) Expanded Repair SAT Encoding\n")
            f.write(f"c Fixed centers: {len(self.fixed_centers)}\n")
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
    """Verify the complete solution."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    vertices = generate_lambda_vertices(n, s)
    vertices_set = set(vertices)

    print(f"\nSolution: {len(centers)} centers")
    print(f"Target: {len(vertices)} vertices to cover")

    # Check all centers are valid
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
    """Save solution to files."""
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
        f.write(f"Method: Expanded s=11 Solution Repair\n")
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
    print("Strategy: Expanded Repair from Lambda_15(1^11) Solution")
    print("=" * 70)

    n, s_new = 15, 10

    # Load s=11 solution
    print("\nLoading s=11 solution...")
    s11_centers = load_s11_solution()
    print(f"Loaded {len(s11_centers)} centers")

    # Try progressively larger repair radii
    for radius in [1, 2, 3, 4, 5]:
        print(f"\n{'='*70}")
        print(f"Trying repair with radius = {radius}")
        print("=" * 70)

        fixed_centers, dropped_centers, vertices_s10, vertices_s10_set = \
            find_repair_zone(s11_centers, n, s_new, radius=radius)

        encoder = ExpandedRepairEncoder(fixed_centers, vertices_s10,
                                         vertices_s10_set, n, s_new)

        if not encoder.encode():
            print(f"\nEncoding failed for radius {radius}")
            continue

        # Run SAT
        cnf_file = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/repair_15_10_r{radius}.cnf"
        encoder.write_dimacs(cnf_file)

        solver = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"
        timeout = 3600  # 1 hour per attempt

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

                final_centers = fixed_centers + new_centers
                print(f"Final: {len(final_centers)} centers")
                print(f"  Fixed: {len(fixed_centers)}")
                print(f"  New: {len(new_centers)}")

                if verify_solution(final_centers, n, s_new):
                    save_solution(final_centers, n, s_new)
                    return  # Success!

            elif proc.returncode == 20:
                print(f"\nUNSAT for radius {radius}")
                continue

        except subprocess.TimeoutExpired:
            print(f"\nTimeout for radius {radius}")
            continue

    print("\n" + "=" * 70)
    print("All repair attempts failed.")
    print("Consider full SAT search or different strategy.")
    print("=" * 70)

if __name__ == "__main__":
    main()
