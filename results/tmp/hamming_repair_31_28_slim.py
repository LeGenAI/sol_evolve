#!/usr/bin/env python3
"""
Slim SAT Encoding for Hamming Repair in Λ₃₁(1²⁸)

Key optimizations:
1. Drop scope: only good codewords at dist <= 2 from bad (drastically reduced)
2. Repair candidates: only allowed vertices covering uncovered AND dist >= 3 from fixed good
3. Adjacency-list based clause generation (no nested loops over full sets)
4. Pre-computed cover maps and conflict graphs
5. Strict cardinality bound on drops

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import defaultdict
import time
from typing import Set, List, Dict, Tuple, FrozenSet

# ============================================================================
# Constants
# ============================================================================
N = 31
S = 28
K = 5

# Bad codewords (hardcoded)
ALL_ONES = (1 << N) - 1
BAD_ZEROS_012 = ALL_ONES & ~7  # Clear bits 0,1,2
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

def get_dist2_neighbors(v: int) -> List[int]:
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(v ^ (1 << i) ^ (1 << j))
    return result

def get_ball(center: int, radius: int) -> Set[int]:
    """Get vertices within radius using BFS (more efficient)"""
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
# Scoped universe construction
# ============================================================================

def build_scoped_model(drop_radius: int = 2, max_drops: int = 50):
    """
    Build a slim SAT model with tightly scoped variables.

    Args:
        drop_radius: Only good codewords within this distance from bad can be dropped
        max_drops: Maximum number of drops allowed

    Returns:
        Model information for CNF encoding
    """
    print("=" * 60)
    print("BUILDING SCOPED MODEL")
    print("=" * 60)
    print(f"Drop radius: {drop_radius}")
    print(f"Max drops: {max_drops}")

    # Step 1: Find droppable good codewords
    # These are good (allowed) Hamming codewords within drop_radius from any bad
    droppable = set()

    for bad in BAD_CODEWORDS:
        ball = get_ball(bad, drop_radius)
        for v in ball:
            if is_codeword(v) and not has_circular_ones(v):
                droppable.add(v)

    # Remove bad codewords themselves (they're already removed)
    droppable -= BAD_CODEWORDS

    print(f"Droppable good codewords: {len(droppable)}")

    # Step 2: Find uncovered allowed vertices after removing bad codewords
    # Universe for checking: vertices within (drop_radius + 1) from bad
    check_universe = set()
    for bad in BAD_CODEWORDS:
        check_universe |= get_ball(bad, drop_radius + 1)

    # Allowed vertices in check universe
    allowed_vertices = {v for v in check_universe if not has_circular_ones(v)}
    print(f"Allowed vertices in check universe: {len(allowed_vertices)}")

    # Fixed good codewords: all good codewords NOT in droppable set
    # (For this scoped model, we assume all others are fixed)
    # We need to know which good codewords cover each vertex

    # Find all good codewords relevant to our universe
    # A codeword c covers v if dist(c, v) <= 1
    # So relevant codewords are within dist (drop_radius + 2) from bad

    relevant_good = set()
    for bad in BAD_CODEWORDS:
        extended_ball = get_ball(bad, drop_radius + 2)
        for v in extended_ball:
            if is_codeword(v) and not has_circular_ones(v):
                relevant_good.add(v)

    relevant_good -= BAD_CODEWORDS
    fixed_good = relevant_good - droppable

    print(f"Relevant good codewords: {len(relevant_good)}")
    print(f"Fixed good codewords: {len(fixed_good)}")

    # Step 3: Build coverage map
    # cover_map[v] = set of good codewords covering v
    cover_map = defaultdict(set)

    for v in allowed_vertices:
        # Check if v is a good codeword
        if v in relevant_good:
            cover_map[v].add(v)

        # Check neighbors
        for neighbor in get_neighbors(v):
            if neighbor in relevant_good:
                cover_map[v].add(neighbor)

    # Step 4: Find initially uncovered vertices (not covered by any fixed good)
    uncovered = set()
    for v in allowed_vertices:
        covering_fixed = cover_map[v] & fixed_good
        if not covering_fixed:
            # Also check if any droppable covers it
            covering_any = cover_map[v]
            if not covering_any:
                uncovered.add(v)  # Truly uncovered
            else:
                # Covered only by droppable codewords
                uncovered.add(v)

    # More precisely: uncovered means not covered by fixed_good
    uncovered = {v for v in allowed_vertices if not (cover_map[v] & fixed_good)}

    print(f"Vertices not covered by fixed good: {len(uncovered)}")

    # Step 5: Find repair candidates
    # A repair candidate r for uncovered v must:
    # - Be allowed (no circular 1^s)
    # - Cover some uncovered vertex (dist <= 1)
    # - Have dist >= 3 to all fixed good codewords

    repair_candidates = set()

    for v in uncovered:
        # v itself
        if not has_circular_ones(v):
            # Check distance to fixed good
            conflict = False
            for fg in fixed_good:
                if hamming_dist(v, fg) < 3:
                    conflict = True
                    break
            if not conflict:
                repair_candidates.add(v)

        # Neighbors of v
        for neighbor in get_neighbors(v):
            if has_circular_ones(neighbor):
                continue

            conflict = False
            for fg in fixed_good:
                if hamming_dist(neighbor, fg) < 3:
                    conflict = True
                    break

            if not conflict:
                repair_candidates.add(neighbor)

    print(f"Repair candidates (dist >= 3 from fixed): {len(repair_candidates)}")

    # If no repair candidates, problem is infeasible
    if uncovered and not repair_candidates:
        print("WARNING: No repair candidates available!")
        print("Need to expand droppable scope or relax constraints")

    # Step 6: Build conflict graph among repair candidates
    # repair_conflicts[r] = set of other repair candidates at dist < 3
    repair_conflicts = defaultdict(set)

    repair_list = list(repair_candidates)
    for i, r1 in enumerate(repair_list):
        for r2 in repair_list[i+1:]:
            if hamming_dist(r1, r2) < 3:
                repair_conflicts[r1].add(r2)
                repair_conflicts[r2].add(r1)

    conflict_edges = sum(len(s) for s in repair_conflicts.values()) // 2
    print(f"Repair conflict edges: {conflict_edges}")

    # Step 7: Build droppable conflict map
    # If we use repair r, which droppable codewords must we NOT drop?
    # Actually opposite: if repair r is used and droppable d is NOT dropped, they must have dist >= 3
    # So: repair_drop_conflicts[r] = droppable codewords at dist < 3 from r
    repair_drop_conflicts = defaultdict(set)

    for r in repair_candidates:
        for d in droppable:
            if hamming_dist(r, d) < 3:
                repair_drop_conflicts[r].add(d)

    # Step 8: Coverage by repair candidates
    # candidate_covers[r] = set of uncovered vertices covered by r
    candidate_covers = defaultdict(set)

    for r in repair_candidates:
        for v in uncovered:
            if hamming_dist(r, v) <= 1:
                candidate_covers[r].add(v)

    # Inverse: which candidates cover each uncovered vertex?
    covers_vertex = defaultdict(set)
    for r, covered in candidate_covers.items():
        for v in covered:
            covers_vertex[v].add(r)

    # Check: every uncovered vertex must have at least one candidate
    uncovered_without_candidate = {v for v in uncovered if not covers_vertex[v]}
    if uncovered_without_candidate:
        print(f"WARNING: {len(uncovered_without_candidate)} uncovered vertices have no candidates!")

    # Summary
    print("\n" + "-" * 60)
    print("MODEL SUMMARY")
    print("-" * 60)
    print(f"Droppable variables: {len(droppable)}")
    print(f"Repair variables: {len(repair_candidates)}")
    print(f"Covering constraints: {len(uncovered)}")
    print(f"Repair packing constraints: {conflict_edges}")
    print(f"Max drops: {max_drops}")

    return {
        'droppable': droppable,
        'fixed_good': fixed_good,
        'uncovered': uncovered,
        'repair_candidates': repair_candidates,
        'repair_conflicts': repair_conflicts,
        'repair_drop_conflicts': repair_drop_conflicts,
        'covers_vertex': covers_vertex,
        'candidate_covers': candidate_covers,
        'cover_map': cover_map,
        'allowed_vertices': allowed_vertices,
        'max_drops': max_drops,
    }


def encode_cnf(model: dict) -> Tuple[int, List[List[int]], dict]:
    """
    Encode the model as CNF with minimal clause count.

    Returns:
        (num_vars, clauses, var_info)
    """
    print("\n" + "=" * 60)
    print("ENCODING TO CNF")
    print("=" * 60)

    droppable = model['droppable']
    repair_candidates = model['repair_candidates']
    repair_conflicts = model['repair_conflicts']
    repair_drop_conflicts = model['repair_drop_conflicts']
    covers_vertex = model['covers_vertex']
    cover_map = model['cover_map']
    uncovered = model['uncovered']
    max_drops = model['max_drops']

    # Variable assignment
    var_counter = 0
    drop_vars = {}  # droppable codeword -> var
    repair_vars = {}  # repair candidate -> var

    for d in droppable:
        var_counter += 1
        drop_vars[d] = var_counter

    for r in repair_candidates:
        var_counter += 1
        repair_vars[r] = var_counter

    print(f"Total variables: {var_counter}")
    print(f"  Drop vars: {len(drop_vars)}")
    print(f"  Repair vars: {len(repair_vars)}")

    clauses = []

    # Constraint 1: Each uncovered vertex must be covered by some repair candidate
    cover_clauses = 0
    for v in uncovered:
        candidates = covers_vertex[v]
        if candidates:
            clause = [repair_vars[c] for c in candidates]
            clauses.append(clause)
            cover_clauses += 1

    print(f"Covering clauses: {cover_clauses}")

    # Constraint 2: Packing among repair candidates
    packing_clauses = 0
    seen_pairs = set()

    for r1, conflicts in repair_conflicts.items():
        for r2 in conflicts:
            pair = (min(r1, r2), max(r1, r2))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                clauses.append([-repair_vars[r1], -repair_vars[r2]])
                packing_clauses += 1

    print(f"Repair packing clauses: {packing_clauses}")

    # Constraint 3: If repair r is used, conflicting droppables must be dropped
    # repair_r => drop_d for each d in repair_drop_conflicts[r]
    # Equivalent: NOT repair_r OR drop_d
    repair_drop_clauses = 0

    for r, conflicts in repair_drop_conflicts.items():
        for d in conflicts:
            clauses.append([-repair_vars[r], drop_vars[d]])
            repair_drop_clauses += 1

    print(f"Repair-drop implication clauses: {repair_drop_clauses}")

    # Constraint 4: At-most-k drops
    # Use totalizer encoding for efficiency
    if max_drops < len(droppable):
        drop_var_list = [drop_vars[d] for d in droppable]
        atmost_clauses, var_counter = encode_atmost_totalizer(drop_var_list, max_drops, var_counter)
        clauses.extend(atmost_clauses)
        print(f"At-most-{max_drops} clauses: {len(atmost_clauses)}")

    print(f"\nTotal clauses: {len(clauses)}")

    var_info = {
        'drop_vars': drop_vars,
        'repair_vars': repair_vars,
    }

    return var_counter, clauses, var_info


def encode_atmost_totalizer(lits: List[int], k: int, start_var: int) -> Tuple[List[List[int]], int]:
    """
    Encode at-most-k constraint using totalizer encoding.
    More efficient than sequential counter for larger k.
    """
    n = len(lits)
    if k >= n:
        return [], start_var

    clauses = []
    var_counter = start_var

    # Simple pairwise encoding for small k
    if k <= 3:
        # At-most-k means no (k+1) variables can all be true
        from itertools import combinations
        for combo in combinations(lits, k + 1):
            clauses.append([-lit for lit in combo])
        return clauses, var_counter

    # For larger k, use sequential counter
    # s[i][j] = "sum of first i vars is at least j"
    s = {}

    for i in range(n):
        for j in range(min(i + 2, k + 2)):
            var_counter += 1
            s[(i, j)] = var_counter

    # Base case
    # s[0][0] <=> x[0]
    clauses.append([-lits[0], s[(0, 0)]])
    clauses.append([lits[0], -s[(0, 0)]])

    # s[0][1] = false
    if (0, 1) in s:
        clauses.append([-s[(0, 1)]])

    # Recursive
    for i in range(1, n):
        for j in range(min(i + 2, k + 2)):
            # s[i][j] <=> s[i-1][j] OR (x[i] AND s[i-1][j-1])
            if j == 0:
                # s[i][0] <=> s[i-1][0] OR x[i]
                clauses.append([-s[(i-1, 0)], s[(i, 0)]])
                clauses.append([-lits[i], s[(i, 0)]])
                clauses.append([s[(i-1, 0)], lits[i], -s[(i, 0)]])
            else:
                # s[i][j] <=> s[i-1][j] OR (x[i] AND s[i-1][j-1])
                if (i-1, j) in s:
                    clauses.append([-s[(i-1, j)], s[(i, j)]])
                if (i-1, j-1) in s:
                    clauses.append([-lits[i], -s[(i-1, j-1)], s[(i, j)]])

                # Negative implication
                neg_clause = []
                if (i-1, j) in s:
                    neg_clause.append(s[(i-1, j)])
                neg_clause.append(lits[i])
                if (i-1, j-1) in s:
                    neg_clause.append(s[(i-1, j-1)])
                neg_clause.append(-s[(i, j)])
                if len(neg_clause) > 1:
                    clauses.append(neg_clause)

    # At-most-k: NOT s[n-1][k]
    if (n-1, k) in s:
        clauses.append([-s[(n-1, k)]])

    return clauses, var_counter


def write_cnf(filename: str, num_vars: int, clauses: List[List[int]]):
    """Write CNF in DIMACS format"""
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')

    print(f"CNF written to {filename}")
    print(f"  Variables: {num_vars}")
    print(f"  Clauses: {len(clauses)}")


def main():
    print("=" * 70)
    print("SLIM SAT ENCODING FOR Λ₃₁(1²⁸) REPAIR")
    print("=" * 70)

    # Verify bad codewords
    print("\nBad codewords:")
    for bad in BAD_CODEWORDS:
        print(f"  {bin(bad)}: weight={popcount(bad)}, syndrome={compute_syndrome(bad)}, has_1^28={has_circular_ones(bad)}")

    # Build scoped model with different parameters
    for drop_radius in [2, 3]:
        for max_drops in [10, 50, 100]:
            print(f"\n\n{'#' * 70}")
            print(f"# DROP_RADIUS={drop_radius}, MAX_DROPS={max_drops}")
            print(f"{'#' * 70}")

            model = build_scoped_model(drop_radius=drop_radius, max_drops=max_drops)

            # Only encode if reasonable size
            if len(model['droppable']) + len(model['repair_candidates']) > 100000:
                print("Model too large, skipping CNF encoding")
                continue

            num_vars, clauses, var_info = encode_cnf(model)

            if len(clauses) > 10_000_000:
                print("Too many clauses, skipping file write")
                continue

            filename = f"repair_31_28_r{drop_radius}_d{max_drops}.cnf"
            write_cnf(filename, num_vars, clauses)


if __name__ == "__main__":
    main()
