#!/usr/bin/env python3
"""
Restricted SAT search for Λ_31(1^28) with limited drops of nearby Hamming centers.

Idea:
  - Universe U: vertices within Hamming distance <= R of any bad Hamming codeword
    (bad = codewords containing circular 1^28 runs).
  - For each u in U, decode its unique nearest Hamming codeword cw(u) via syndrome decoding.
  - Collect the finite set C of codewords that appear as cw(u) for u in U.
    Split C into bad (B) and good (G).
  - Allow dropping at most k_drop of the droppable good codewords (those within R_drop of a bad).
  - Add repair centers from allowed vertices in U.
  - Constraints (over U):
      * Coverage: each allowed vertex in U must be covered by exactly one selected center
        (kept good or chosen repair, bad are always dropped).
      * Packing: selected centers pairwise distance >= 3.
      * Drop limit: at most k_drop droppable goods.
  This is a local, certificate-style search; it does not enumerate the full Hamming code.
"""

from collections import defaultdict
import itertools
import subprocess
import sys
from pathlib import Path

N = 31
S = 28  # forbid circular 1^28
DEFAULT_R = 2   # radius for local universe
DEFAULT_R_DROP = 2  # droppable goods: within this distance of a bad codeword
DEFAULT_K_DROP = 2  # allow dropping up to k droppable good centers
DEFAULT_SOLVER = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------
def has_circular_run(v: int, n: int, L: int) -> bool:
    mask = (1 << L) - 1
    doubled = (v << n) | v
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def neighbors(v: int, n: int):
    for i in range(n):
        yield v ^ (1 << i)


# ------------------------------------------------------------
# Hamming(31) syndrome decoding (m=5 parity bits)
# ------------------------------------------------------------
m = 5
cols = []
for i in range(1, N + 1):
    col = [(i >> j) & 1 for j in range(m)]
    cols.append(col[::-1])
import numpy as np
H = np.array(cols, dtype=int).T  # shape (5,31)


def syndrome(v: int):
    bits = np.array([(v >> i) & 1 for i in range(N)], dtype=int)
    syn = (H @ bits) % 2
    return tuple(int(x) for x in syn)


def nearest_codeword(v: int):
    syn = syndrome(v)
    if max(syn) == 0:
        return v, 0
    syn_rev = syn[::-1]
    err_idx = int("".join(str(x) for x in syn_rev), 2)
    return v ^ (1 << (err_idx - 1)), 1


# ------------------------------------------------------------
# Build bad set and local universe
# ------------------------------------------------------------
def build_bad_set():
    def from_zero_positions(zs):
        v = (1 << N) - 1
        for z in zs:
            v &= ~(1 << z)
        return v

    return {
        from_zero_positions([]),            # all ones
        from_zero_positions([0, 1, 2]),
        from_zero_positions([0, 29, 30]),
    }


def build_universe(bad, radius):
    U = set()
    frontier = set(bad)
    for _ in range(radius):
        new_frontier = set()
        for v in frontier:
            new_frontier.update(neighbors(v, N))
        U.update(frontier)
        frontier = new_frontier
    U.update(frontier)
    allowed = {v for v in U if not has_circular_run(v, N, S)}
    return allowed


# ------------------------------------------------------------
# CNF builder
# ------------------------------------------------------------
class CNF:
    def __init__(self):
        self.clauses = []
        self.var_counter = 0
        self.varmap = {}

    def new_var(self, key):
        self.var_counter += 1
        self.varmap[key] = self.var_counter
        return self.var_counter

    def var(self, key):
        return self.varmap[key]

    def add(self, lits):
        self.clauses.append(lits)

    def at_least_one(self, lits):
        self.add(lits)

    def at_most_one_pairwise(self, lits):
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                self.add([-lits[i], -lits[j]])

    def exactly_one(self, lits):
        if not lits:
            self.add([1, -1])  # impossible
            return
        self.at_least_one(lits)
        self.at_most_one_pairwise(lits)

    def write_dimacs(self, path):
        with open(path, "w") as f:
            f.write(f"p cnf {self.var_counter} {len(self.clauses)}\n")
            for c in self.clauses:
                f.write(" ".join(map(str, c)) + " 0\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Restricted Λ_31(1^28) repair search on Hamming backbone")
    parser.add_argument("--radius", type=int, default=DEFAULT_R, help="Universe radius R from bad centers")
    parser.add_argument("--drop-radius", type=int, default=DEFAULT_R_DROP, help="Droppable goods are within this distance of a bad center")
    parser.add_argument("--k-drop", type=int, default=DEFAULT_K_DROP, help="Maximum number of droppable goods that may be dropped")
    parser.add_argument("--solver", type=str, default=DEFAULT_SOLVER, help="Path to SAT solver (Cadical)")
    parser.add_argument("--dimacs", type=str, default="lambda_31_28_restricted.cnf", help="Output CNF path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    R = args.radius
    R_drop = args.drop_radius
    k_drop = args.k_drop

    bad = build_bad_set()
    allowed_universe = build_universe(bad, R)
    print(f"R={R}, R_drop={R_drop}, k_drop={k_drop}")
    print(f"Local allowed universe size: {len(allowed_universe)}")

    # Decode nearest codeword for each allowed vertex
    cw_of = {}
    for v in allowed_universe:
        cw, _ = nearest_codeword(v)
        cw_of[v] = cw

    C = set(cw_of.values())
    B = {c for c in C if c in bad}
    G = sorted(c for c in C if c not in bad)
    print(f"Codewords touching universe: total {len(C)}, good {len(G)}, bad {len(B)}")

    # Split good centers into droppable vs fixed (distance to bad <= R_drop)
    droppable = []
    fixed_goods = []
    for g in G:
        if min(hamming_distance(g, b) for b in bad) <= R_drop:
            droppable.append(g)
        else:
            fixed_goods.append(g)
    print(f"Droppable goods: {len(droppable)}, fixed goods: {len(fixed_goods)}")

    # Determine covered vertices by fixed goods
    covered_by_fixed = set()
    for v in allowed_universe:
        cw = cw_of[v]
        if cw in B:
            continue
        if cw in fixed_goods:
            covered_by_fixed.add(v)
    partially_uncovered = allowed_universe - covered_by_fixed

    # Candidate repairs: allowed vertices within distance 1 of partially uncovered vertices,
    # filtered to respect packing vs fixed goods (distance >=3)
    candidates = set()
    for u in partially_uncovered:
        candidates.add(u)
        candidates.update(neighbors(u, N))
    candidates = {v for v in candidates if not has_circular_run(v, N, S)}

    def ok_vs_fixed(v):
        return all(hamming_distance(v, g) >= 3 for g in fixed_goods)
    candidates = {v for v in candidates if ok_vs_fixed(v)}

    # Drop subset of droppable: variables x_g (1=keep)
    cnf = CNF()
    x_vars = {}
    for g in droppable:
        x_vars[g] = cnf.new_var(("keep", g))

    y_vars = {}
    for r in candidates:
        y_vars[r] = cnf.new_var(("repair", r))

    # Drop limit: at most k_drop dropped => at least len(droppable)-k_drop kept
    if k_drop < len(droppable):
        from itertools import combinations
        for subset in combinations(droppable, k_drop + 1):
            cnf.add([x_vars[g] for g in subset])

    # Coverage constraints for each vertex in allowed_universe
    for v in allowed_universe:
        cw = cw_of[v]
        fixed_cover = []
        if cw in bad:
            pass  # removed
        elif cw in droppable:
            fixed_cover = [x_vars[cw]]
        elif cw in fixed_goods:
            fixed_cover = [1]  # implicit keep
        else:
            fixed_cover = [1]  # implicit keep (codeword outside local split)

        dyn_cover = []
        for r in candidates:
            if hamming_distance(r, v) <= 1:
                dyn_cover.append(y_vars[r])

        if fixed_cover == [1]:
            for lit in dyn_cover:
                cnf.add([-lit])
        else:
            cover_list = fixed_cover + dyn_cover
            cnf.exactly_one(cover_list)

    # Packing constraints
    # repairs vs repairs
    cand_list = list(candidates)
    for i, a in enumerate(cand_list):
        va = y_vars[a]
        for b in cand_list[i + 1:]:
            if hamming_distance(a, b) < 3:
                cnf.add([-va, -y_vars[b]])
    # repairs vs kept droppable goods
    for r in candidates:
        vr = y_vars[r]
        for g in droppable:
            if hamming_distance(r, g) < 3:
                cnf.add([-vr, -x_vars[g]])

    dimacs = Path(args.dimacs)
    cnf.write_dimacs(dimacs)
    print(f"Wrote CNF with {cnf.var_counter} vars, {len(cnf.clauses)} clauses")

    # Solve
    solver = args.solver
    if not Path(solver).exists():
        print(f"Solver not found at {solver}")
        return
    proc = subprocess.run(
        [solver, str(dimacs)],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (10, 20):
        print("Solver error:", proc.returncode)
        print(proc.stdout)
        print(proc.stderr)
        return
    sat = proc.returncode == 10
    print("SAT status:", "SAT" if sat else "UNSAT")
    if not sat:
        return
    # Parse model
    vals = set()
    for line in proc.stdout.splitlines():
        if not line.startswith("v"):
            continue
        for tok in line.split()[1:]:
            if tok == "0":
                continue
            vals.add(int(tok))
    chosen_repairs = [r for r, vid in y_vars.items() if vid in vals]
    kept_goods = [g for g, vid in x_vars.items() if vid in vals]
    dropped_goods = [g for g in droppable if g not in kept_goods]
    print(f"Repairs chosen: {len(chosen_repairs)}; kept droppable goods: {len(kept_goods)} / {len(droppable)}; dropped: {len(dropped_goods)}")
    if args.verbose:
        def bits(v): return format(v, f"0{N}b")
        print("Dropped goods (bin):", [bits(g) for g in dropped_goods])
        print("Repairs (bin):", [bits(r) for r in chosen_repairs])


if __name__ == "__main__":
    main()
