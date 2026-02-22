#!/usr/bin/env python3
"""
Extended SAT Model for Λ₃₁(1²⁸) - testing larger drop limits
"""

import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import Set, List, Dict, Tuple

N = 31
S = 28
K = 5

ALL_ONES = (1 << N) - 1
BAD_ZEROS_012 = ALL_ONES & ~7
BAD_ZEROS_29_30_0 = ALL_ONES & ~((1 << 29) | (1 << 30) | 1)
BAD_CODEWORDS = frozenset([ALL_ONES, BAD_ZEROS_012, BAD_ZEROS_29_30_0])

def popcount(v): return bin(v).count('1')
def hamming_dist(v1, v2): return popcount(v1 ^ v2)

def has_circular_ones(v, s=S):
    doubled = v | (v << N)
    mask = (1 << s) - 1
    for i in range(N):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def compute_syndrome(v):
    syn = 0
    for i in range(K):
        parity = 0
        for j in range(N):
            if (v >> j) & 1:
                if ((j + 1) >> i) & 1:
                    parity ^= 1
        syn |= (parity << i)
    return syn

def is_codeword(v): return compute_syndrome(v) == 0
def get_neighbors(v): return [v ^ (1 << i) for i in range(N)]

def get_ball(center, radius):
    if radius == 0: return {center}
    ball = {center}
    frontier = [center]
    for _ in range(radius):
        nf = []
        for v in frontier:
            for i in range(N):
                n = v ^ (1 << i)
                if n not in ball:
                    ball.add(n)
                    nf.append(n)
        frontier = nf
    return ball


def build_model(drop_radius, max_drops, cover_radius):
    """Build model with given parameters"""
    # Universe
    universe = set()
    for bad in BAD_CODEWORDS:
        universe |= get_ball(bad, cover_radius)
    allowed = {v for v in universe if not has_circular_ones(v)}

    # Droppable
    droppable = set()
    for bad in BAD_CODEWORDS:
        ball = get_ball(bad, drop_radius)
        for v in ball:
            if is_codeword(v) and not has_circular_ones(v) and v not in BAD_CODEWORDS:
                droppable.add(v)

    # Fixed good (extended universe)
    ext_univ = set()
    for bad in BAD_CODEWORDS:
        ext_univ |= get_ball(bad, cover_radius + 1)

    fixed_good = set()
    for v in ext_univ:
        if is_codeword(v) and not has_circular_ones(v) and v not in BAD_CODEWORDS and v not in droppable:
            fixed_good.add(v)

    # Coverage maps
    fixed_cover = defaultdict(set)
    droppable_cover = defaultdict(set)
    for v in allowed:
        for c in [v] + get_neighbors(v):
            if c in fixed_good:
                fixed_cover[v].add(c)
            elif c in droppable:
                droppable_cover[v].add(c)

    always_covered = {v for v in allowed if fixed_cover[v]}
    needs_coverage = allowed - always_covered

    # Repair candidates
    repair_candidates = set()
    candidate_covers = defaultdict(set)

    for v in needs_coverage:
        for cand in [v] + get_neighbors(v):
            if has_circular_ones(cand):
                continue
            ok = True
            for fg in fixed_good:
                if hamming_dist(cand, fg) < 3:
                    ok = False
                    break
            if ok:
                repair_candidates.add(cand)
                candidate_covers[cand].add(v)

    # Conflicts
    repair_conflicts = []
    repair_list = list(repair_candidates)
    for i, r1 in enumerate(repair_list):
        for r2 in repair_list[i+1:]:
            if hamming_dist(r1, r2) < 3:
                repair_conflicts.append((r1, r2))

    repair_drop_must = defaultdict(set)
    for r in repair_candidates:
        for d in droppable:
            if hamming_dist(r, d) < 3:
                repair_drop_must[r].add(d)

    covers_vertex = defaultdict(set)
    for r, covered in candidate_covers.items():
        for v in covered:
            covers_vertex[v].add(r)

    return {
        'allowed': allowed,
        'needs_coverage': needs_coverage,
        'droppable': droppable,
        'fixed_good': fixed_good,
        'repair_candidates': repair_candidates,
        'droppable_cover': droppable_cover,
        'covers_vertex': covers_vertex,
        'repair_conflicts': repair_conflicts,
        'repair_drop_must': repair_drop_must,
        'max_drops': max_drops,
    }


def encode_cnf(model):
    droppable = model['droppable']
    repair_candidates = model['repair_candidates']
    needs_coverage = model['needs_coverage']
    droppable_cover = model['droppable_cover']
    covers_vertex = model['covers_vertex']
    repair_conflicts = model['repair_conflicts']
    repair_drop_must = model['repair_drop_must']
    max_drops = model['max_drops']

    var = 0
    drop_vars = {}
    repair_vars = {}

    for d in droppable:
        var += 1
        drop_vars[d] = var
    for r in repair_candidates:
        var += 1
        repair_vars[r] = var

    clauses = []

    # Covering
    for v in needs_coverage:
        clause = []
        for d in droppable_cover[v]:
            clause.append(-drop_vars[d])
        for r in covers_vertex[v]:
            clause.append(repair_vars[r])
        if clause:
            clauses.append(clause)

    # Repair packing
    for r1, r2 in repair_conflicts:
        clauses.append([-repair_vars[r1], -repair_vars[r2]])

    # Repair-drop implications
    for r, must_drops in repair_drop_must.items():
        for d in must_drops:
            clauses.append([-repair_vars[r], drop_vars[d]])

    # At-most-k (skip for large max_drops)
    if max_drops < len(droppable):
        if max_drops <= 100:
            # Sequential counter would be huge, use relaxed constraint
            # Or skip for now to see if unlimited drops work
            pass

    return var, clauses, {'drop_vars': drop_vars, 'repair_vars': repair_vars}


def write_cnf(filename, num_vars, clauses):
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for c in clauses:
            f.write(' '.join(map(str, c)) + ' 0\n')
    print(f"Written {filename}: {num_vars} vars, {len(clauses)} clauses")


def main():
    print("Testing extended parameters...")

    # Test without max_drops constraint first
    configs = [
        (3, 500, 3),    # Large max_drops
        (3, 1000, 3),   # Even larger
        (4, 500, 4),    # Larger radii
    ]

    for drop_r, max_d, cover_r in configs:
        print(f"\n{'='*60}")
        print(f"drop_radius={drop_r}, max_drops={max_d}, cover_radius={cover_r}")
        print(f"{'='*60}")

        model = build_model(drop_r, max_d, cover_r)

        print(f"Droppable: {len(model['droppable'])}")
        print(f"Fixed good: {len(model['fixed_good'])}")
        print(f"Needs coverage: {len(model['needs_coverage'])}")
        print(f"Repair candidates: {len(model['repair_candidates'])}")
        print(f"Repair conflicts: {len(model['repair_conflicts'])}")

        # Check uncoverable
        uncoverable = [v for v in model['needs_coverage']
                       if not model['droppable_cover'][v] and not model['covers_vertex'][v]]
        if uncoverable:
            print(f"UNCOVERABLE: {len(uncoverable)}")
            continue

        num_vars, clauses, var_info = encode_cnf(model)
        print(f"CNF: {num_vars} vars, {len(clauses)} clauses")

        filename = f"repair_31_28_ext_r{drop_r}_d{max_d}_c{cover_r}.cnf"
        write_cnf(filename, num_vars, clauses)


if __name__ == "__main__":
    main()
