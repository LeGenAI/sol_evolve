#!/usr/bin/env python3
"""
Cardinality-constrained SAT encoder for Λ₁₅(1¹⁰).

Modes
- Exact (legacy positional args):
    a b C
    -> exactly a centers from V14, b from V15, d=C-a-b from V16
- Band (coarse search):
    --band C_min C_max --a-range A_min A_max --b-range B_min B_max
    -> total centers in [C_min, C_max], type14 centers in [A_min, A_max],
       type15 centers in [B_min, B_max]. Type16 count left implicit.

Base constraints (both modes):
  - Covering: each vertex is covered by exactly one chosen center within distance 1
  - Packing: chosen centers must be pairwise distance >= 3

Usage example (minimum center case):
  python3 lambda_15_10_card.py 0 1 2033
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np
from pysat.card import CardEnc, EncType


def has_circ(v: int, n: int, s: int) -> bool:
    doubled = (v << n) | v
    mask = (1 << s) - 1
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ball(c: int, n: int, vset):
    b = [c]
    for i in range(n):
        nb = c ^ (1 << i)
        if nb in vset:
            b.append(nb)
    return b


class Encoder:
    def __init__(self, n: int, s: int):
        self.n = n
        self.s = s
        self.clauses = []
        self.aux_var = 1  # will set after vars

    def add_clause(self, lits):
        self.clauses.append(lits)

    def new_var(self):
        v = self.aux_var
        self.aux_var += 1
        return v

    def amo_seq(self, lits):
        """At-most-one via sequential counter."""
        if len(lits) <= 4:
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self.add_clause([-lits[i], -lits[j]])
            return
        n = len(lits)
        s = [self.new_var() for _ in range(n - 1)]
        self.add_clause([-lits[0], s[0]])
        self.add_clause([lits[0], -s[0]])
        for i in range(1, n - 1):
            self.add_clause([-s[i], s[i - 1], lits[i]])
            self.add_clause([-s[i - 1], s[i]])
            self.add_clause([-lits[i], s[i]])
            self.add_clause([-s[i - 1], -lits[i]])
        self.add_clause([-s[-1], -lits[-1]])

    def exactly_one(self, lits):
        self.add_clause(lits)  # at least one
        self.amo_seq(lits)     # at most one

    def add_cardinality_equals(self, lits, k):
        """Add exactly-k constraint using PySAT's sequential counter."""
        if k < 0 or k > len(lits):
            raise ValueError("k out of range for cardinality")
        # top_id must be current max var id
        top_id = self.aux_var - 1
        cnf = CardEnc.equals(lits, bound=k, encoding=EncType.seqcounter, top_id=top_id)
        # update aux_var to reflect new variables introduced
        self.aux_var = cnf.nv + 1
        for c in cnf.clauses:
            self.add_clause(c)

    def add_cardinality_atmost(self, lits, k):
        if k >= len(lits):
            return
        if k < 0:
            raise ValueError("k out of range for atmost")
        top_id = self.aux_var - 1
        cnf = CardEnc.atmost(lits, bound=k, encoding=EncType.seqcounter, top_id=top_id)
        self.aux_var = cnf.nv + 1
        for c in cnf.clauses:
            self.add_clause(c)

    def add_cardinality_atleast(self, lits, k):
        if k <= 0:
            return
        if k > len(lits):
            raise ValueError("k out of range for atleast")
        # at-least-k is equivalent to at-most-(n-k) on negated literals
        self.add_cardinality_atmost([-l for l in lits], len(lits) - k)

    def add_cardinality_between(self, lits, lo, hi):
        if lo > hi:
            raise ValueError("invalid bounds lo>hi")
        self.add_cardinality_atleast(lits, lo)
        self.add_cardinality_atmost(lits, hi)


def main():
    parser = argparse.ArgumentParser(
        description="Cardinality SAT encoder for Λ_15(1^10) with exact or banded counts."
    )
    sub = parser.add_subparsers(dest="mode", required=False)

    p_exact = sub.add_parser("exact", help="Exact a,b,C (legacy positional)")
    p_exact.add_argument("a", type=int, nargs=1)
    p_exact.add_argument("b", type=int, nargs=1)
    p_exact.add_argument("C", type=int, nargs=1)

    p_band = sub.add_parser("band", help="Banded counts for total/type14/type15")
    p_band.add_argument("--band", nargs=2, type=int, required=True, metavar=("C_MIN", "C_MAX"))
    p_band.add_argument("--a-range", nargs=2, type=int, required=True, metavar=("A_MIN", "A_MAX"))
    p_band.add_argument("--b-range", nargs=2, type=int, required=True, metavar=("B_MIN", "B_MAX"))

    args = parser.parse_args()

    n, s = 15, 10

    if args.mode == "band":
        C_min, C_max = args.band
        A_min, A_max = args.a_range
        B_min, B_max = args.b_range
        if not (0 <= C_min <= C_max):
            print("Invalid C range", file=sys.stderr); sys.exit(1)
        if not (0 <= A_min <= A_max):
            print("Invalid A (type14) range", file=sys.stderr); sys.exit(1)
        if not (0 <= B_min <= B_max):
            print("Invalid B (type15) range", file=sys.stderr); sys.exit(1)
        mode_desc = f"band: C in [{C_min},{C_max}], a in [{A_min},{A_max}], b in [{B_min},{B_max}]"
    else:
        # default to exact mode if subcommand omitted
        if args.mode not in (None, "exact"):
            parser.error("Unknown mode")
        if len(sys.argv) < 4:
            parser.error("Usage (exact mode): python3 lambda_15_10_card.py exact a b C")
        a = int(args.a[0]) if args.mode == "exact" else int(sys.argv[1])
        b = int(args.b[0]) if args.mode == "exact" else int(sys.argv[2])
        C = int(args.C[0]) if args.mode == "exact" else int(sys.argv[3])
        mode_desc = f"exact: a={a}, b={b}, C={C}"

    print(f"Λ_{n}(1^{s}) cardinality SAT ({mode_desc})")

    # Generate vertices
    t0 = time.time()
    verts = [v for v in range(1 << n) if not has_circ(v, n, s)]
    vset = set(verts)
    print(f"|V| = {len(verts)} (gen {time.time() - t0:.2f}s)")

    # Balls, membership, and type buckets
    print("Precomputing balls and membership...")
    t1 = time.time()
    balls = {}
    membership = defaultdict(list)
    type14, type15, type16 = [], [], []
    for v in verts:
        bset = ball(v, n, vset)
        balls[v] = bset
        size = len(bset)
        if size == 14:
            type14.append(v)
        elif size == 15:
            type15.append(v)
        elif size == 16:
            type16.append(v)
        else:
            raise ValueError(f"Unexpected ball size {size} at vertex {v}")
        for u in bset:
            membership[u].append(v)
    print(f"Done ({time.time() - t1:.2f}s)")
    print(f"Type counts: 14->{len(type14)}, 15->{len(type15)}, 16->{len(type16)}")

    # Check feasibility of counts
    if args.mode == "band":
        if A_min > len(type14) or B_min > len(type15):
            print("Lower bounds exceed available vertices", file=sys.stderr); sys.exit(1)
        if A_max > len(type14): A_max = len(type14)
        if B_max > len(type15): B_max = len(type15)
    else:
        if not (0 <= a <= len(type14) and 0 <= b <= len(type15)):
            print("Requested a/b exceed available vertices", file=sys.stderr)
            sys.exit(1)
        d = C - a - b
        if d < 0 or d > len(type16):
            print(f"Invalid d derived from C: d={d}", file=sys.stderr)
            sys.exit(1)

    enc = Encoder(n, s)

    # Variable map
    var_map = {v: i + 1 for i, v in enumerate(verts)}
    enc.aux_var = len(verts) + 1

    def var(v):
        return var_map[v]

    # Covering constraints
    print("Encoding covering (exactly one center covers each vertex)...")
    for u in verts:
        lits = [var(c) for c in membership[u]]
        enc.exactly_one(lits)

    # Packing constraints (distance < 3 forbidden)
    print("Encoding packing constraints...")
    pack_clauses = 0
    for idx, v in enumerate(verts):
        vv = var(v)
        # distance 1 neighbors
        for i in range(n):
            nb = v ^ (1 << i)
            if nb in vset and nb > v:
                enc.add_clause([-vv, -var(nb)])
                pack_clauses += 1
        # distance 2 neighbors
        for i in range(n):
            for j in range(i + 1, n):
                nb = v ^ (1 << i) ^ (1 << j)
                if nb in vset and nb > v:
                    enc.add_clause([-vv, -var(nb)])
                    pack_clauses += 1
        if (idx + 1) % 8000 == 0:
            print(f"  processed {idx + 1}/{len(verts)} vertices")
    print(f"Packing clauses added: {pack_clauses}")

    # Cardinality constraints per type
    if args.mode == "band":
        print("Adding banded cardinality constraints...")
        enc.add_cardinality_between([var(v) for v in type14], A_min, A_max)
        enc.add_cardinality_between([var(v) for v in type15], B_min, B_max)
        enc.add_cardinality_between(list(var_map.values()), C_min, C_max)
    else:
        print("Adding cardinality constraints (type14, type15)...")
        enc.add_cardinality_equals([var(v) for v in type14], a)
        enc.add_cardinality_equals([var(v) for v in type15], b)
        # We do not encode type16 count explicitly; with exact cover + packing,
        # the total number of centers is forced to the unique feasible value.

    max_var = 0
    for c in enc.clauses:
        for lit in c:
            if abs(lit) > max_var:
                max_var = abs(lit)

    print(f"Vars total (header): {max_var}")
    print(f"Clauses total: {len(enc.clauses)}")

    if args.mode == "band":
        cnf_path = (
            f"/Users/baegjaehyeon/CodeEvolve/results/tmp/"
            f"lambda_15_10_band_C{C_min}-{C_max}_a{A_min}-{A_max}_b{B_min}-{B_max}.cnf"
        )
    else:
        cnf_path = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_a{a}_b{b}_C{C}.cnf"
    print(f"Writing CNF to {cnf_path} ...")
    with open(cnf_path, "w") as f:
        f.write(f"p cnf {max_var} {len(enc.clauses)}\n")
        for c in enc.clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
    print("Done.")


if __name__ == "__main__":
    main()
