#!/usr/bin/env python3
"""Partial MaxSAT seed for Lambda_15(1^10).

Hard clauses: packing (distance < 3 forbidden), centers must be allowed vertices.
Soft clauses: each vertex should be covered by some center (closed ball).

Goal: find a near-perfect cover quickly as a seed for LNS/repair.
Saves centers and a coverage report to results/tmp.
"""

import sys
import time
from collections import defaultdict

import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver


def has_circ_ones(v: int, n: int, s: int) -> bool:
    """Return True if v has s consecutive ones on a ring of length n."""
    doubled = (v << n) | v
    mask = (1 << s) - 1
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def generate_vertices(n: int, s: int):
    verts = [v for v in range(1 << n) if not has_circ_ones(v, n, s)]
    return verts, set(verts)


def build_balls(vertices, n, vset):
    balls = {}
    membership = defaultdict(list)
    for v in vertices:
        b = [v]
        for i in range(n):
            nb = v ^ (1 << i)
            if nb in vset:
                b.append(nb)
        balls[v] = b
        for u in b:
            membership[u].append(v)
    return balls, membership


def main():
    n, s = 15, 10
    out_prefix = "/Users/baegjaehyeon/CodeEvolve/results/tmp/maxsat_seed_15_10"

    print(f"Lambda_{n}(1^{s}) partial MaxSAT seed")
    t0 = time.time()
    verts, vset = generate_vertices(n, s)
    print(f"|V| = {len(verts)} (generated in {time.time() - t0:.2f}s)")

    print("Building balls and membership...")
    t1 = time.time()
    balls, membership = build_balls(verts, n, vset)
    print(f"Done in {time.time() - t1:.2f}s")

    # Variable map
    var_map = {v: i + 1 for i, v in enumerate(verts)}
    num_vars = len(verts)

    wcnf = WCNF()

    # Soft clauses: coverage
    for u in verts:
        lits = [var_map[c] for c in membership[u]]
        # All vertices have at least self in membership, so lits non-empty
        wcnf.append(lits, weight=1)

    # Hard clauses: packing distance < 3 forbidden
    print("Adding packing clauses...")
    pack_added = 0
    for v in verts:
        v_var = var_map[v]
        # distance 1
        for i in range(n):
            nb = v ^ (1 << i)
            if nb in vset and nb > v:
                wcnf.append([-v_var, -var_map[nb]])
                pack_added += 1
        # distance 2
        for i in range(n):
            for j in range(i + 1, n):
                nb = v ^ (1 << i) ^ (1 << j)
                if nb in vset and nb > v:
                    wcnf.append([-v_var, -var_map[nb]])
                    pack_added += 1
    print(f"Packing clauses: {pack_added}")

    print("Solving (RC2)...")
    t2 = time.time()
    solver = RC2(wcnf, solver="g4")
    model = solver.compute()
    dt = time.time() - t2
    cost = solver.cost
    print(f"RC2 finished in {dt:.1f}s, cost={cost} (uncovered vertices)")

    centers = [verts[i] for i in range(len(verts)) if model[i] > 0]
    print(f"Centers selected: {len(centers)}")

    # Coverage check
    covered = set()
    for c in centers:
        for u in balls[c]:
            covered.add(u)
    uncovered = [v for v in verts if v not in covered]
    print(f"Covered vertices: {len(covered)} / {len(verts)}")
    print(f"Remaining uncovered: {len(uncovered)}")

    np.save(f"{out_prefix}_centers.npy", np.array(centers, dtype=np.uint32))
    np.save(f"{out_prefix}_uncovered.npy", np.array(uncovered, dtype=np.uint32))
    with open(f"{out_prefix}_summary.txt", "w") as f:
        f.write(f"n={n}, s={s}\n")
        f.write(f"|V|={len(verts)}\n")
        f.write(f"centers={len(centers)}\n")
        f.write(f"uncovered={len(uncovered)} (RC2 cost={cost})\n")
        f.write(f"time={dt:.1f}s (solver), total={time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()