#!/usr/bin/env python3
"""
Complete SAT Encoding for Hamming Repair in Λ₃₁(1²⁸)

Key improvements over slim version:
1. Conditional covering: droppable codewords provide coverage when NOT dropped
2. Full coverage model: fixed_good + (non-dropped droppable) + repairs = complete cover
3. Clear universe boundary and verification

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import defaultdict
import time
from typing import Set, List, Dict, Tuple, FrozenSet
from itertools import combinations

# ============================================================================
# Constants
# ============================================================================
N = 31
S = 28
K = 5

ALL_ONES = (1 << N) - 1
BAD_ZEROS_012 = ALL_ONES & ~7
BAD_ZEROS_29_30_0 = ALL_ONES & ~((1 << 29) | (1 << 30) | 1)
BAD_CODEWORDS = frozenset([ALL_ONES, BAD_ZEROS_012, BAD_ZEROS_29_30_0])

# ============================================================================
# Bit operations
# ============================================================================

def popcount(v: int) -> int:
    return bin(v).count('1')

def hamming_dist(v1: int, v2: int) -> int:
    return popcount(v1 ^ v2)

def has_circular_ones(v: int, s: int = S) -> bool:
    doubled = v | (v << N)
    mask = (1 << s) - 1
    for i in range(N):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def compute_syndrome(v: int) -> int:
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

def get_neighbors(v: int) -> List[int]:
    return [v ^ (1 << i) for i in range(N)]

def get_ball(center: int, radius: int) -> Set[int]:
    if radius == 0:
        return {center}
    ball = {center}
    frontier = [center]
    for _ in range(radius):
        next_frontier = []
        for v in frontier:
            for i in range(N):
                neighbor = v ^ (1 << i)
                if neighbor not in ball:
                    ball.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return ball

# ============================================================================
# Complete model with conditional covering
# ============================================================================

def build_complete_model(drop_radius: int = 2, max_drops: int = 20, cover_radius: int = 3):
    """
    Build complete SAT model with conditional covering.

    Coverage equation for each allowed vertex v:
        v is covered iff:
        - Some fixed_good c covers v (dist <= 1), OR
        - Some droppable d covers v AND d is NOT dropped, OR
        - Some repair r covers v AND r is selected

    Args:
        drop_radius: Droppable scope (dist from bad)
        max_drops: Maximum drops allowed
        cover_radius: Universe radius for coverage verification
    """
    print("=" * 60)
    print("BUILDING COMPLETE MODEL WITH CONDITIONAL COVERING")
    print("=" * 60)
    print(f"Parameters: drop_radius={drop_radius}, max_drops={max_drops}, cover_radius={cover_radius}")

    # Step 1: Define universe (vertices that must be covered)
    universe = set()
    for bad in BAD_CODEWORDS:
        universe |= get_ball(bad, cover_radius)

    allowed_universe = {v for v in universe if not has_circular_ones(v)}
    print(f"Universe size: {len(universe)}")
    print(f"Allowed vertices in universe: {len(allowed_universe)}")

    # Step 2: Identify droppable good codewords (within drop_radius from bad)
    droppable = set()
    for bad in BAD_CODEWORDS:
        ball = get_ball(bad, drop_radius)
        for v in ball:
            if is_codeword(v) and not has_circular_ones(v) and v not in BAD_CODEWORDS:
                droppable.add(v)

    print(f"Droppable good codewords: {len(droppable)}")

    # Step 3: Identify fixed good codewords (can cover universe but not droppable)
    # These are good codewords within (cover_radius + 1) that are not droppable
    extended_universe = set()
    for bad in BAD_CODEWORDS:
        extended_universe |= get_ball(bad, cover_radius + 1)

    fixed_good = set()
    for v in extended_universe:
        if is_codeword(v) and not has_circular_ones(v) and v not in BAD_CODEWORDS and v not in droppable:
            fixed_good.add(v)

    print(f"Fixed good codewords: {len(fixed_good)}")

    # Step 4: Build coverage map for each allowed vertex
    # For each v: who can cover it?
    # - fixed_cover[v] = fixed good codewords covering v (always active)
    # - droppable_cover[v] = droppable codewords covering v (active if not dropped)
    # - potential_repair_cover[v] = potential repair positions covering v

    fixed_cover = defaultdict(set)
    droppable_cover = defaultdict(set)

    for v in allowed_universe:
        # Check self and neighbors
        check_positions = [v] + get_neighbors(v)
        for c in check_positions:
            if c in fixed_good:
                fixed_cover[v].add(c)
            elif c in droppable:
                droppable_cover[v].add(c)

    # Vertices covered by fixed good (always covered, no constraint needed)
    always_covered = {v for v in allowed_universe if fixed_cover[v]}
    print(f"Always covered by fixed good: {len(always_covered)}")

    # Vertices needing conditional coverage
    needs_coverage = allowed_universe - always_covered
    print(f"Vertices needing conditional coverage: {len(needs_coverage)}")

    # Step 5: Identify repair candidates
    # A repair candidate must:
    # - Be allowed
    # - Cover at least one vertex in needs_coverage
    # - Have dist >= 3 from all fixed_good

    repair_candidates = set()
    candidate_covers = defaultdict(set)  # repair -> vertices it covers

    for v in needs_coverage:
        # v itself as candidate
        if not has_circular_ones(v):
            ok = True
            for fg in fixed_good:
                if hamming_dist(v, fg) < 3:
                    ok = False
                    break
            if ok:
                repair_candidates.add(v)
                candidate_covers[v].add(v)

        # Neighbors of v as candidates
        for neighbor in get_neighbors(v):
            if has_circular_ones(neighbor):
                continue
            ok = True
            for fg in fixed_good:
                if hamming_dist(neighbor, fg) < 3:
                    ok = False
                    break
            if ok:
                repair_candidates.add(neighbor)
                candidate_covers[neighbor].add(v)

    print(f"Repair candidates: {len(repair_candidates)}")

    # Step 6: Build conflict structures
    # repair_conflicts: pairs of repairs that can't both be selected
    repair_conflicts = []
    repair_list = list(repair_candidates)
    for i, r1 in enumerate(repair_list):
        for r2 in repair_list[i+1:]:
            if hamming_dist(r1, r2) < 3:
                repair_conflicts.append((r1, r2))

    print(f"Repair conflict pairs: {len(repair_conflicts)}")

    # repair_drop_must: if repair r is used, droppable d must be dropped
    repair_drop_must = defaultdict(set)
    for r in repair_candidates:
        for d in droppable:
            if hamming_dist(r, d) < 3:
                repair_drop_must[r].add(d)

    # Step 7: Coverage inverse map
    # covers_vertex[v] = {repair candidates covering v}
    covers_vertex = defaultdict(set)
    for r, covered in candidate_covers.items():
        for v in covered:
            covers_vertex[v].add(r)

    # Check coverage feasibility
    uncoverable = set()
    for v in needs_coverage:
        # v is covered by: droppable_cover[v] (if kept) OR covers_vertex[v] (repairs)
        if not droppable_cover[v] and not covers_vertex[v]:
            uncoverable.add(v)

    if uncoverable:
        print(f"WARNING: {len(uncoverable)} vertices cannot be covered!")
        for v in list(uncoverable)[:5]:
            print(f"  {bin(v)}")

    return {
        'allowed_universe': allowed_universe,
        'needs_coverage': needs_coverage,
        'droppable': droppable,
        'fixed_good': fixed_good,
        'repair_candidates': repair_candidates,
        'fixed_cover': fixed_cover,
        'droppable_cover': droppable_cover,
        'covers_vertex': covers_vertex,
        'repair_conflicts': repair_conflicts,
        'repair_drop_must': repair_drop_must,
        'candidate_covers': candidate_covers,
        'max_drops': max_drops,
    }


def encode_complete_cnf(model: dict) -> Tuple[int, List[List[int]], dict]:
    """
    Encode complete model with conditional covering.

    For each vertex v in needs_coverage:
        (OR over d in droppable_cover[v]: NOT drop_d) OR (OR over r in covers_vertex[v]: repair_r)

    This means: v is covered if some droppable covering it is kept, OR some repair is selected.
    """
    print("\n" + "=" * 60)
    print("ENCODING COMPLETE CNF")
    print("=" * 60)

    droppable = model['droppable']
    repair_candidates = model['repair_candidates']
    needs_coverage = model['needs_coverage']
    droppable_cover = model['droppable_cover']
    covers_vertex = model['covers_vertex']
    repair_conflicts = model['repair_conflicts']
    repair_drop_must = model['repair_drop_must']
    max_drops = model['max_drops']

    # Variable assignment
    var_counter = 0
    drop_vars = {}  # d -> var (True = drop d)
    repair_vars = {}  # r -> var (True = select r)

    for d in droppable:
        var_counter += 1
        drop_vars[d] = var_counter

    for r in repair_candidates:
        var_counter += 1
        repair_vars[r] = var_counter

    print(f"Variables: {var_counter}")
    print(f"  Drop vars: {len(drop_vars)}")
    print(f"  Repair vars: {len(repair_vars)}")

    clauses = []

    # Constraint 1: Conditional covering for each vertex needing coverage
    # For vertex v: (OR_d NOT drop_d) OR (OR_r repair_r)
    # where d ranges over droppable_cover[v], r ranges over covers_vertex[v]
    cover_clauses = 0

    for v in needs_coverage:
        clause = []

        # NOT dropped droppables that cover v
        for d in droppable_cover[v]:
            clause.append(-drop_vars[d])  # NOT drop_d means d is kept and covers v

        # Selected repairs that cover v
        for r in covers_vertex[v]:
            clause.append(repair_vars[r])

        if clause:
            clauses.append(clause)
            cover_clauses += 1
        else:
            print(f"  ERROR: Vertex {bin(v)} has no covering options!")

    print(f"Covering clauses: {cover_clauses}")

    # Constraint 2: Repair packing (no two repairs at dist < 3)
    packing_clauses = 0
    for r1, r2 in repair_conflicts:
        clauses.append([-repair_vars[r1], -repair_vars[r2]])
        packing_clauses += 1

    print(f"Repair packing clauses: {packing_clauses}")

    # Constraint 3: If repair r is selected, conflicting droppables must be dropped
    # repair_r => drop_d for each d in repair_drop_must[r]
    impl_clauses = 0
    for r, must_drops in repair_drop_must.items():
        for d in must_drops:
            clauses.append([-repair_vars[r], drop_vars[d]])
            impl_clauses += 1

    print(f"Repair-drop implication clauses: {impl_clauses}")

    # Constraint 4: At-most-k drops
    if max_drops < len(droppable):
        drop_var_list = list(drop_vars.values())

        if max_drops <= 5:
            # Small k: use pairwise
            for combo in combinations(drop_var_list, max_drops + 1):
                clauses.append([-v for v in combo])
            print(f"At-most-{max_drops} clauses (pairwise): {len(list(combinations(drop_var_list, max_drops + 1)))}")
        else:
            # Larger k: use sequential counter
            atmost_clauses, var_counter = encode_sequential_counter(drop_var_list, max_drops, var_counter)
            clauses.extend(atmost_clauses)
            print(f"At-most-{max_drops} clauses (seq counter): {len(atmost_clauses)}")

    print(f"\nTotal clauses: {len(clauses)}")
    print(f"Final variables: {var_counter}")

    return var_counter, clauses, {'drop_vars': drop_vars, 'repair_vars': repair_vars}


def encode_sequential_counter(lits: List[int], k: int, start_var: int) -> Tuple[List[List[int]], int]:
    """Sequential counter encoding for at-most-k"""
    n = len(lits)
    if k >= n:
        return [], start_var

    clauses = []
    var = start_var

    # s[i][j] = "sum of lits[0..i] >= j+1"
    s = {}
    for i in range(n):
        for j in range(min(i + 1, k + 1)):
            var += 1
            s[(i, j)] = var

    # Base: s[0][0] <=> lits[0]
    clauses.append([-lits[0], s[(0, 0)]])
    clauses.append([lits[0], -s[(0, 0)]])

    # Recursion
    for i in range(1, n):
        for j in range(min(i + 1, k + 1)):
            if j == 0:
                # s[i][0] <=> s[i-1][0] OR lits[i]
                clauses.append([-s[(i-1, 0)], s[(i, 0)]])
                clauses.append([-lits[i], s[(i, 0)]])
                clauses.append([s[(i-1, 0)], lits[i], -s[(i, 0)]])
            else:
                # s[i][j] <=> s[i-1][j] OR (lits[i] AND s[i-1][j-1])
                if (i-1, j) in s:
                    clauses.append([-s[(i-1, j)], s[(i, j)]])
                clauses.append([-lits[i], -s[(i-1, j-1)], s[(i, j)]])
                # Negative
                neg = [s[(i-1, j)] if (i-1, j) in s else None, lits[i], s[(i-1, j-1)], -s[(i, j)]]
                neg = [x for x in neg if x is not None]
                clauses.append(neg)

    # At-most-k: NOT s[n-1][k-1] would mean sum >= k, but we want sum <= k
    # Actually s[n-1][k] means sum >= k+1, so we forbid it
    if (n-1, k) in s:
        clauses.append([-s[(n-1, k)]])

    return clauses, var


def write_cnf(filename: str, num_vars: int, clauses: List[List[int]]):
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')
    print(f"Written to {filename}: {num_vars} vars, {len(clauses)} clauses")


def parse_solution(filename: str, var_info: dict) -> Tuple[Set[int], Set[int]]:
    """Parse SAT solution and return (dropped, selected_repairs)"""
    drop_vars = var_info['drop_vars']
    repair_vars = var_info['repair_vars']

    # Invert mappings
    var_to_drop = {v: k for k, v in drop_vars.items()}
    var_to_repair = {v: k for k, v in repair_vars.items()}

    dropped = set()
    repairs = set()

    with open(filename) as f:
        for line in f:
            if line.startswith('v '):
                for token in line[2:].split():
                    lit = int(token)
                    if lit == 0:
                        break
                    if lit > 0:
                        if lit in var_to_drop:
                            dropped.add(var_to_drop[lit])
                        if lit in var_to_repair:
                            repairs.add(var_to_repair[lit])

    return dropped, repairs


def verify_solution(model: dict, dropped: Set[int], repairs: Set[int]) -> bool:
    """Verify that solution provides valid covering and packing"""
    print("\n" + "=" * 60)
    print("VERIFYING SOLUTION")
    print("=" * 60)

    fixed_good = model['fixed_good']
    droppable = model['droppable']
    allowed_universe = model['allowed_universe']

    # Active centers = fixed_good + (droppable - dropped) + repairs
    active_centers = fixed_good | (droppable - dropped) | repairs

    print(f"Active centers: {len(active_centers)}")
    print(f"  Fixed good: {len(fixed_good)}")
    print(f"  Kept droppable: {len(droppable - dropped)}")
    print(f"  Repairs: {len(repairs)}")

    # Check covering
    uncovered = []
    for v in allowed_universe:
        covered = False
        if v in active_centers:
            covered = True
        else:
            for neighbor in get_neighbors(v):
                if neighbor in active_centers:
                    covered = True
                    break
        if not covered:
            uncovered.append(v)

    if uncovered:
        print(f"FAIL: {len(uncovered)} vertices uncovered!")
        return False
    print("Covering: OK")

    # Check packing
    conflicts = []
    center_list = list(active_centers)
    for i, c1 in enumerate(center_list):
        for c2 in center_list[i+1:]:
            if hamming_dist(c1, c2) < 3:
                conflicts.append((c1, c2))

    if conflicts:
        print(f"FAIL: {len(conflicts)} packing violations!")
        return False
    print("Packing: OK")

    print("VERIFICATION PASSED")
    return True


def main():
    print("=" * 70)
    print("COMPLETE SAT MODEL FOR Λ₃₁(1²⁸) REPAIR")
    print("=" * 70)

    # Try different parameter combinations
    configs = [
        (2, 10, 2),   # Tight: drop_radius=2, max_drops=10, cover_radius=2
        (2, 20, 3),   # Medium: drop_radius=2, max_drops=20, cover_radius=3
        (3, 50, 3),   # Wider: drop_radius=3, max_drops=50, cover_radius=3
    ]

    for drop_r, max_d, cover_r in configs:
        print(f"\n\n{'#' * 70}")
        print(f"# CONFIG: drop_radius={drop_r}, max_drops={max_d}, cover_radius={cover_r}")
        print(f"{'#' * 70}")

        model = build_complete_model(drop_radius=drop_r, max_drops=max_d, cover_radius=cover_r)

        # Skip if model is too large
        total_vars = len(model['droppable']) + len(model['repair_candidates'])
        if total_vars > 50000:
            print(f"Skipping: too many variables ({total_vars})")
            continue

        num_vars, clauses, var_info = encode_complete_cnf(model)

        if len(clauses) > 5_000_000:
            print(f"Skipping: too many clauses ({len(clauses)})")
            continue

        filename = f"repair_31_28_complete_r{drop_r}_d{max_d}_c{cover_r}.cnf"
        write_cnf(filename, num_vars, clauses)

        print(f"\nTo solve: /Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical {filename}")


if __name__ == "__main__":
    main()
