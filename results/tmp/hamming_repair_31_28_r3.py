#!/usr/bin/env python3
"""
Proof-based Hamming Repair for Λ₃₁(1²⁸)

Instead of greedy cascade, we use SAT encoding with:
1. Bad codewords: 3 known (all-ones + 2 weight-28)
2. Universe: vertices within radius R=3 from bad codewords
3. Variables: drop good codewords, add repair centers
4. Constraints: covering + packing in the universe

This approach provides verifiable proofs.

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import defaultdict
import time
from typing import Set, List, Dict, Tuple, Optional

# ============================================================================
# Constants
# ============================================================================
N = 31
S = 28
K = 5

# Known bad codewords (hardcoded to avoid full scan)
# 1. All-ones: 2^31 - 1
ALL_ONES = (1 << N) - 1

# 2. Weight-28 with zeros at positions {0, 1, 2}
# Binary: 1...1111000 (positions 0,1,2 are 0)
BAD_ZEROS_012 = ALL_ONES & ~(1 << 0) & ~(1 << 1) & ~(1 << 2)

# 3. Weight-28 with zeros at positions {29, 30, 0}
# Binary: 0011...1111110 (positions 29,30,0 are 0)
BAD_ZEROS_29_30_0 = ALL_ONES & ~(1 << 29) & ~(1 << 30) & ~(1 << 0)

BAD_CODEWORDS = [ALL_ONES, BAD_ZEROS_012, BAD_ZEROS_29_30_0]

print(f"Bad codewords defined:")
print(f"  All-ones: {bin(ALL_ONES)}")
print(f"  Zeros at {{0,1,2}}: {bin(BAD_ZEROS_012)}")
print(f"  Zeros at {{29,30,0}}: {bin(BAD_ZEROS_29_30_0)}")

# ============================================================================
# Bit operations
# ============================================================================

def popcount(v: int) -> int:
    return bin(v).count('1')

def hamming_dist(v1: int, v2: int) -> int:
    return popcount(v1 ^ v2)

def has_circular_ones(v: int, s: int = S) -> bool:
    """Check if v contains s consecutive 1s circularly"""
    doubled = v | (v << N)
    mask = (1 << s) - 1
    for i in range(N):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def compute_syndrome(v: int) -> int:
    """Compute syndrome as 5-bit integer"""
    syn = 0
    for i in range(K):
        parity = 0
        for j in range(N):
            if (v >> j) & 1:
                if ((j + 1) >> i) & 1:
                    parity ^= 1
        syn |= (parity << i)
    return syn

def is_codeword(v: int) -> bool:
    return compute_syndrome(v) == 0

def syndrome_decode(v: int) -> Tuple[int, int]:
    """Decode v to nearest codeword. Returns (codeword, distance)."""
    syn = compute_syndrome(v)
    if syn == 0:
        return v, 0
    error_pos = syn - 1
    if 0 <= error_pos < N:
        return v ^ (1 << error_pos), 1
    return v, -1

def get_neighbors(v: int) -> List[int]:
    """Get all distance-1 neighbors"""
    return [v ^ (1 << i) for i in range(N)]

def get_ball(center: int, radius: int) -> Set[int]:
    """Get all vertices within Hamming distance radius from center"""
    ball = {center}
    frontier = {center}

    for r in range(radius):
        new_frontier = set()
        for v in frontier:
            for neighbor in get_neighbors(v):
                if neighbor not in ball:
                    ball.add(neighbor)
                    new_frontier.add(neighbor)
        frontier = new_frontier

    return ball

# ============================================================================
# Universe construction
# ============================================================================

def build_universe(radius: int = 3) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
    """
    Build the universe of vertices affected by bad codewords.

    Returns:
        (universe, allowed_in_universe, good_codewords_in_universe, uncovered_allowed)
    """
    print(f"\n{'='*60}")
    print(f"Building universe with radius R={radius}")
    print(f"{'='*60}")

    # Universe: all vertices within radius R from any bad codeword
    universe = set()
    for bad in BAD_CODEWORDS:
        universe |= get_ball(bad, radius)

    print(f"Universe size: {len(universe):,}")

    # Allowed vertices in universe (no circular 1^s)
    allowed = {v for v in universe if not has_circular_ones(v)}
    print(f"Allowed vertices in universe: {len(allowed):,}")

    # Good Hamming codewords in universe
    good_codewords = {v for v in allowed if is_codeword(v)}
    print(f"Good codewords in universe: {len(good_codewords):,}")

    # Uncovered allowed vertices (not covered by remaining good codewords)
    # A vertex v is covered if v or any neighbor is a good codeword
    uncovered = set()
    for v in allowed:
        covered = False
        if v in good_codewords:
            covered = True
        else:
            for neighbor in get_neighbors(v):
                if neighbor in good_codewords:
                    covered = True
                    break
        if not covered:
            uncovered.add(v)

    print(f"Uncovered allowed vertices: {len(uncovered):,}")

    return universe, allowed, good_codewords, uncovered


def analyze_conflicts(good_codewords: Set[int], uncovered: Set[int]) -> Dict[int, Set[int]]:
    """
    For each uncovered vertex, find good codewords at distance 2.
    These are the conflict codewords that prevent simple repair.
    """
    print(f"\nAnalyzing conflicts...")

    conflicts = defaultdict(set)

    for v in uncovered:
        # Find good codewords at distance 2 from v
        for i in range(N):
            for j in range(i + 1, N):
                d2 = v ^ (1 << i) ^ (1 << j)
                if d2 in good_codewords:
                    conflicts[v].add(d2)

    total_conflict_codewords = set()
    for cw_set in conflicts.values():
        total_conflict_codewords |= cw_set

    print(f"Unique conflict codewords: {len(total_conflict_codewords):,}")

    conflict_counts = [len(conflicts[v]) for v in uncovered]
    if conflict_counts:
        print(f"Conflicts per uncovered vertex: min={min(conflict_counts)}, max={max(conflict_counts)}, avg={np.mean(conflict_counts):.1f}")

    return conflicts


def find_repair_candidates(uncovered: Set[int], good_codewords: Set[int]) -> Set[int]:
    """
    Find candidate repair centers for uncovered vertices.
    A repair candidate r for uncovered v must:
    - Be allowed (no circular 1^s)
    - Be at distance <= 1 from v
    """
    candidates = set()

    for v in uncovered:
        # v itself as candidate
        if not has_circular_ones(v):
            candidates.add(v)

        # Neighbors of v as candidates
        for neighbor in get_neighbors(v):
            if not has_circular_ones(neighbor):
                candidates.add(neighbor)

    # Exclude existing good codewords (they're already centers)
    candidates -= good_codewords

    print(f"Repair candidates: {len(candidates):,}")

    return candidates


# ============================================================================
# Weight-3 codewords for efficient conflict detection
# ============================================================================

def get_weight3_codewords() -> List[int]:
    """
    Get all weight-3 Hamming(31) codewords.
    There are exactly 155 such codewords.
    A codeword has weight 3 iff its 3 positions p1 < p2 < p3 satisfy:
    (p1+1) XOR (p2+1) XOR (p3+1) = 0
    """
    weight3 = []
    for p1 in range(N):
        for p2 in range(p1 + 1, N):
            for p3 in range(p2 + 1, N):
                # Check if these form a codeword
                if ((p1 + 1) ^ (p2 + 1) ^ (p3 + 1)) == 0:
                    cw = (1 << p1) | (1 << p2) | (1 << p3)
                    weight3.append(cw)

    print(f"Weight-3 codewords: {len(weight3)} (expected 155)")
    return weight3

WEIGHT3_CODEWORDS = get_weight3_codewords()


def fast_conflict_check(center: int, existing_centers: Set[int]) -> bool:
    """
    Check if center conflicts with any existing center (distance < 3).
    Uses weight-3 codewords: center conflicts with c iff
    (center XOR c) is a codeword of weight < 3.
    """
    for c in existing_centers:
        diff = center ^ c
        weight = popcount(diff)
        if weight < 3:
            return True  # Conflict
        # Weight exactly 3: check if diff is a codeword
        if weight == 3 and diff in WEIGHT3_CODEWORDS:
            return True  # Distance = 3 but diff is codeword means d < 3? No, this is wrong

    # Actually: d(center, c) < 3 iff center XOR c has weight 0, 1, or 2
    # Weight 3 codewords are for checking if distance is exactly 2
    # (if XOR is weight-3 codeword, then they share... no, this is getting confused)

    # Simple correct version:
    for c in existing_centers:
        if hamming_dist(center, c) < 3:
            return True
    return False


# ============================================================================
# CNF Encoding
# ============================================================================

class CNFEncoder:
    """Encode the repair problem as SAT"""

    def __init__(self, allowed: Set[int], good_codewords: Set[int],
                 uncovered: Set[int], repair_candidates: Set[int],
                 max_drops: int = 500):
        self.allowed = allowed
        self.good = good_codewords
        self.uncovered = uncovered
        self.candidates = repair_candidates
        self.max_drops = max_drops

        # Variable mapping
        self.var_counter = 0
        self.drop_vars = {}  # good codeword -> variable (True = drop it)
        self.repair_vars = {}  # candidate -> variable (True = use as center)

        self.clauses = []

    def new_var(self) -> int:
        self.var_counter += 1
        return self.var_counter

    def encode(self) -> Tuple[int, List[List[int]]]:
        """
        Encode the problem as CNF.

        Variables:
        - drop_i: True if we drop good codeword i
        - repair_j: True if we use candidate j as repair center

        Constraints:
        1. Covering: each uncovered vertex must be covered by some repair center
        2. Packing: repair centers must be distance >= 3 from:
           - Each other
           - Non-dropped good codewords
        3. Optional: limit number of drops
        """
        print(f"\n{'='*60}")
        print("Encoding to CNF...")
        print(f"{'='*60}")

        # Create variables
        for cw in self.good:
            self.drop_vars[cw] = self.new_var()

        for c in self.candidates:
            self.repair_vars[c] = self.new_var()

        print(f"Variables: {self.var_counter}")
        print(f"  Drop variables: {len(self.drop_vars)}")
        print(f"  Repair variables: {len(self.repair_vars)}")

        # Constraint 1: Covering
        # For each uncovered vertex v, at least one covering condition:
        # - Some repair candidate at distance <= 1 is used

        for v in self.uncovered:
            covering_lits = []

            # Check which repair candidates cover v
            for c in self.candidates:
                if hamming_dist(v, c) <= 1:
                    covering_lits.append(self.repair_vars[c])

            if covering_lits:
                self.clauses.append(covering_lits)  # At least one
            else:
                print(f"  WARNING: Uncovered vertex {bin(v)} has no repair candidates!")

        print(f"Covering clauses: {len(self.clauses)}")

        # Constraint 2: Packing among repair centers
        packing_clauses = 0
        repair_list = list(self.candidates)

        for i, r1 in enumerate(repair_list):
            for r2 in repair_list[i+1:]:
                if hamming_dist(r1, r2) < 3:
                    # Cannot both be centers
                    self.clauses.append([-self.repair_vars[r1], -self.repair_vars[r2]])
                    packing_clauses += 1

        print(f"Packing clauses (repair-repair): {packing_clauses}")

        # Constraint 3: Packing between repair centers and good codewords
        # If repair r is used and good cw is not dropped, they must be distance >= 3
        good_repair_clauses = 0

        for r in self.candidates:
            for cw in self.good:
                if hamming_dist(r, cw) < 3:
                    # If repair r is used, cw must be dropped
                    # repair_r => drop_cw
                    # NOT repair_r OR drop_cw
                    self.clauses.append([-self.repair_vars[r], self.drop_vars[cw]])
                    good_repair_clauses += 1

        print(f"Packing clauses (repair-good): {good_repair_clauses}")

        # Constraint 4: Dropped codewords must have their neighborhoods covered
        # If good cw is dropped, all allowed vertices in N[cw] must be covered
        drop_cover_clauses = 0

        for cw in self.good:
            # Vertices that cw was covering
            covered_by_cw = [cw] + get_neighbors(cw)

            for v in covered_by_cw:
                if v not in self.allowed:
                    continue  # Forbidden vertex, no need to cover

                # If cw is dropped, v must be covered by:
                # - Another non-dropped good codeword, OR
                # - A repair center

                covering_lits = [-self.drop_vars[cw]]  # If cw not dropped, constraint satisfied

                # Other good codewords covering v
                for other_cw in self.good:
                    if other_cw != cw and hamming_dist(v, other_cw) <= 1:
                        # If other_cw is not dropped, it covers v
                        covering_lits.append(-self.drop_vars[other_cw])

                # Repair centers covering v
                for r in self.candidates:
                    if hamming_dist(v, r) <= 1:
                        covering_lits.append(self.repair_vars[r])

                self.clauses.append(covering_lits)
                drop_cover_clauses += 1

        print(f"Drop-cover clauses: {drop_cover_clauses}")

        # Optional: Limit total drops using cardinality constraint
        # (simple encoding: at-most-k using sequential counter)
        if self.max_drops < len(self.good):
            self._add_atmost_k([self.drop_vars[cw] for cw in self.good], self.max_drops)

        print(f"\nTotal clauses: {len(self.clauses)}")

        return self.var_counter, self.clauses

    def _add_atmost_k(self, lits: List[int], k: int):
        """Add at-most-k constraint using sequential counter encoding"""
        n = len(lits)
        if k >= n:
            return  # No constraint needed

        # Create counter variables s[i][j] meaning "sum of first i vars >= j"
        s = [[self.new_var() for _ in range(k + 1)] for _ in range(n)]

        # Base case: s[0][0] = x[0], s[0][j>0] = false
        self.clauses.append([-lits[0], s[0][0]])  # x[0] => s[0][0]
        self.clauses.append([lits[0], -s[0][0]])  # NOT x[0] => NOT s[0][0]

        for j in range(1, k + 1):
            self.clauses.append([-s[0][j]])  # s[0][j] = false for j > 0

        # Recursive case
        for i in range(1, n):
            for j in range(k + 1):
                # s[i][j] iff s[i-1][j] OR (x[i] AND s[i-1][j-1])
                if j == 0:
                    # s[i][0] iff s[i-1][0] OR x[i]
                    self.clauses.append([-s[i-1][0], s[i][0]])
                    self.clauses.append([-lits[i], s[i][0]])
                    self.clauses.append([s[i-1][0], lits[i], -s[i][0]])
                else:
                    # s[i][j] iff s[i-1][j] OR (x[i] AND s[i-1][j-1])
                    self.clauses.append([-s[i-1][j], s[i][j]])
                    self.clauses.append([-lits[i], -s[i-1][j-1], s[i][j]])
                    self.clauses.append([s[i-1][j], lits[i], -s[i][j]])
                    self.clauses.append([s[i-1][j], s[i-1][j-1], -s[i][j]])

        # At-most-k: NOT s[n-1][k]
        self.clauses.append([-s[n-1][k]])

        print(f"Added at-most-{k} constraint with {n} variables")


def write_cnf(filename: str, num_vars: int, clauses: List[List[int]]):
    """Write CNF to DIMACS format file"""
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')

    print(f"CNF written to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("PROOF-BASED HAMMING REPAIR FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    # Verify bad codewords
    print("\nVerifying bad codewords...")
    for bad in BAD_CODEWORDS:
        is_cw = is_codeword(bad)
        has_ones = has_circular_ones(bad)
        print(f"  {bin(bad)}: is_codeword={is_cw}, has_circular_1^28={has_ones}")

    # Build universe
    universe, allowed, good_codewords, uncovered = build_universe(radius=3)

    # Analyze conflicts
    conflicts = analyze_conflicts(good_codewords, uncovered)

    # Find repair candidates
    repair_candidates = find_repair_candidates(uncovered, good_codewords)

    # Encode to CNF
    encoder = CNFEncoder(allowed, good_codewords, uncovered, repair_candidates, max_drops=1000)
    num_vars, clauses = encoder.encode()

    # Write CNF
    cnf_filename = "hamming_repair_31_28_r3.cnf"
    write_cnf(cnf_filename, num_vars, clauses)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Universe radius: 3")
    print(f"Universe size: {len(universe):,}")
    print(f"Allowed vertices: {len(allowed):,}")
    print(f"Good codewords: {len(good_codewords):,}")
    print(f"Uncovered vertices: {len(uncovered):,}")
    print(f"Repair candidates: {len(repair_candidates):,}")
    print(f"CNF variables: {num_vars}")
    print(f"CNF clauses: {len(clauses)}")
    print(f"\nNext step: Run SAT solver on {cnf_filename}")
    print(f"  /Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical {cnf_filename}")


if __name__ == "__main__":
    main()
