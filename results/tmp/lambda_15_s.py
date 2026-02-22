#!/usr/bin/env python3
"""Lambda_15(1^s) - Quick SAT test."""

import numpy as np
from collections import defaultdict, Counter
import subprocess
import time
import sys

def has_circ(v, n, s):
    doubled = (v << n) | v
    mask = (1 << s) - 1
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming(a, b):
    return bin(a ^ b).count('1')

def ball(c, n, vset):
    b = [c]
    for i in range(n):
        nb = c ^ (1 << i)
        if nb in vset:
            b.append(nb)
    return b

s = int(sys.argv[1])
n = 15

print(f"Lambda_{n}(1^{s})")

verts = [v for v in range(1 << n) if not has_circ(v, n, s)]
vset = set(verts)
print(f"|V| = {len(verts)}")

# Ball sizes
balls = {v: ball(v, n, vset) for v in verts}
sizes = Counter(len(b) for b in balls.values())
print(f"Ball sizes: {dict(sizes)}")

# Expected centers
expected = sum(1.0 / len(balls[v]) for v in verts)
print(f"Expected centers: {expected:.1f}")

# Ball membership
membership = defaultdict(list)
for v in verts:
    for u in balls[v]:
        membership[u].append(v)

# SAT Encoding
var_map = {c: i + 1 for i, c in enumerate(verts)}
aux = [len(verts) + 1]
clauses = []

def new_aux():
    r = aux[0]
    aux[0] += 1
    return r

def amo(lits):
    if len(lits) <= 1:
        return
    if len(lits) <= 4:
        for i in range(len(lits)):
            for j in range(i+1, len(lits)):
                clauses.append([-lits[i], -lits[j]])
        return
    n_l = len(lits)
    sv = [new_aux() for _ in range(n_l - 1)]
    clauses.append([-lits[0], sv[0]])
    for i in range(1, n_l - 1):
        clauses.append([-sv[i-1], sv[i]])
        clauses.append([-lits[i], sv[i]])
    clauses.append([-sv[-1], -lits[-1]])
    for i in range(n_l - 1):
        clauses.append([-lits[i+1], -sv[i]])

print("Encoding...")
for v in verts:
    cov = membership[v]
    lits = [var_map[c] for c in cov]
    clauses.append(lits)  # ALO
    amo(lits)  # AMO

# Packing
pack = 0
for i, c1 in enumerate(verts):
    for c2 in verts[i+1:]:
        if hamming(c1, c2) < 3:
            clauses.append([-var_map[c1], -var_map[c2]])
            pack += 1

print(f"Packing clauses: {pack}")

# Symmetry
if 0 in var_map:
    clauses.append([var_map[0]])

print(f"Vars: {aux[0]-1}, Clauses: {len(clauses)}")

# Write CNF
cnf = f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s}.cnf"
with open(cnf, 'w') as f:
    f.write(f"p cnf {aux[0]-1} {len(clauses)}\n")
    for c in clauses:
        f.write(" ".join(map(str, c)) + " 0\n")

# Solve
solver = "/Users/baegjaehyeon/CodeEvolve/sat_solvers/cadical/build/cadical"
print(f"\nRunning CaDiCaL...")
t0 = time.time()
try:
    proc = subprocess.run([solver, cnf], capture_output=True, text=True, timeout=600)
    dt = time.time() - t0

    if proc.returncode == 10:
        print(f"SAT! ({dt:.1f}s)")
        centers = []
        for line in proc.stdout.split('\n'):
            if line.startswith('v '):
                for p in line.split()[1:]:
                    if p == '0': continue
                    v = int(p)
                    if v > 0 and v <= len(verts):
                        centers.append(verts[v-1])
        print(f"Centers: {len(centers)}")
        np.save(f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_{s}_centers.npy", np.array(centers))
    elif proc.returncode == 20:
        print(f"UNSAT ({dt:.1f}s)")
    else:
        print(f"Unknown ({proc.returncode})")
except subprocess.TimeoutExpired:
    print("TIMEOUT (600s)")
