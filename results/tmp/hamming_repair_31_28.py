#!/usr/bin/env python3
"""
Restricted repair search for Λ_31(1^28) on top of the Hamming(31) perfect code.

Assumption: keep all Hamming codewords except the three “bad” ones that contain a
circular 1^28 run. Try to add repair centers that:
  - are valid (no circular 1^28),
  - have distance >= 3 from every *kept* Hamming codeword,
  - have pairwise distance >= 3,
  - cover every vertex that becomes uncovered when the bad centers are removed.

We work implicitly:
  * Hamming(31) has 2^26 codewords (too large to enumerate).
  * Each vertex decodes to a unique nearest codeword; other codewords within
    distance < 3 differ from it by a weight-3 codeword.
  * There are only 155 weight-3 codewords (triples of parity-check columns summing to 0).

Result: with all good codewords kept, no repair center exists (same conclusion as
the earlier combinatorial check). This script codifies that reasoning cleanly.
"""
from functools import lru_cache
import itertools

N = 31
S = 28  # forbid circular run of length 28

# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Parity-check machinery for Hamming(31)
# ----------------------------------------------------------------------

m = 5
cols = []
for i in range(1, N + 1):
    col = [(i >> j) & 1 for j in range(m)]
    cols.append(col[::-1])  # reversed bits
import numpy as np
H = np.array(cols, dtype=int).T  # shape (5,31)

@lru_cache(None)
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
    cw = v ^ (1 << (err_idx - 1))
    return cw, 1

# Enumerate all weight-3 codewords (155 of them)
weight3_masks = []
col_vectors = [H[:, i] for i in range(N)]
for comb in itertools.combinations(range(N), 3):
    if (col_vectors[comb[0]] ^ col_vectors[comb[1]] ^ col_vectors[comb[2]]).sum() == 0:
        mask = 0
        for pos in comb:
            mask |= (1 << pos)
        weight3_masks.append(mask)

def min_distance_to_good_codeword(v: int, bad_set: set):
    """
    Return the minimum distance from v to any Hamming codeword not in bad_set.
    Uses: all codewords = nearest cw0 plus any cw0 xor w where w is a codeword.
    Min distance to other codewords is min_{w} weight(e xor w) where e=v xor cw0.
    """
    cw0, d0 = nearest_codeword(v)
    e_mask = v ^ cw0  # weight 0 or 1
    best = 100
    # First, cw0 itself if good
    if cw0 not in bad_set:
        best = min(best, d0)
    # Other codewords reachable via weight-3 codewords
    for w_mask in weight3_masks:
        cw = cw0 ^ w_mask
        if cw in bad_set:
            continue
        d = (e_mask ^ w_mask).bit_count()
        if d < best:
            best = d
            if best == 1:
                break
    return best

# ----------------------------------------------------------------------
# Local universe around bad centers
# ----------------------------------------------------------------------

def main():
    # Identify the three bad codewords explicitly by pattern
    def from_zero_positions(zs):
        v = (1 << N) - 1
        for z in zs:
            v &= ~(1 << z)
        return v

    bad = {
        from_zero_positions([]),               # all ones
        from_zero_positions([0, 1, 2]),        # zeros at 0,1,2
        from_zero_positions([0, 29, 30]),      # zeros at 0,29,30
    }
    print("Bad centers (zero positions):", [[i for i in range(N) if ((b >> i) & 1) == 0] for b in bad])

    # Universe = radius-2 ball around bad centers
    universe = set()
    for b in bad:
        universe.add(b)
        for nb in neighbors(b, N):
            universe.add(nb)
            for n2 in neighbors(nb, N):
                universe.add(n2)

    allowed_universe = {v for v in universe if not has_circular_run(v, N, S)}
    print("Local allowed universe size:", len(allowed_universe))

    # Uncovered vertices after removing bad centers: those whose nearest codeword is bad
    uncovered = set()
    for v in allowed_universe:
        cw, d = nearest_codeword(v)
        if cw in bad:
            uncovered.add(v)
    print("Uncovered vertices (allowed, local):", len(uncovered))

    # Candidate repairs: allowed vertices within distance 1 of uncovered vertices
    cand = set()
    for u in uncovered:
        cand.add(u)
        cand.update(neighbors(u, N))
    cand = {v for v in cand if not has_circular_run(v, N, S)}
    print("Raw candidates:", len(cand))

    # Filter by packing vs good codewords (distance >=3 from every good codeword)
    packed = []
    for c in cand:
        mind = min_distance_to_good_codeword(c, bad)
        if mind >= 3:
            packed.append(c)
    print("Candidates distance>=3 from all good codewords:", len(packed))
    if not packed:
        print("No repair possible without dropping some good centers.")
        return

    # Coverage of uncovered by candidates
    cover = {c: {u for u in uncovered if hamming_distance(c, u) <= 1} for c in packed}
    cover = {c: cov for c, cov in cover.items() if cov}
    print("Candidates that cover uncovered vertices:", len(cover))
    cand_list = list(cover.keys())

    # Pairwise packing among candidates
    conflict = {c: set() for c in cand_list}
    for i, a in enumerate(cand_list):
        for b in cand_list[i + 1:]:
            if hamming_distance(a, b) < 3:
                conflict[a].add(b)
                conflict[b].add(a)

    uncovered_set = set(uncovered)
    best = None

    def search(remaining, chosen, available):
        nonlocal best
        if not remaining:
            best = list(chosen)
            return True
        if best is not None:
            return True
        v = next(iter(remaining))
        options = [c for c in available if v in cover[c]]
        for c in options:
            cov = cover[c]
            new_remaining = remaining - cov
            new_available = available - conflict[c] - {c}
            chosen.append(c)
            if search(new_remaining, chosen, new_available):
                return True
            chosen.pop()
        return False

    search(uncovered_set, [], set(cand_list))
    if best is None:
        print("No repair found with all good centers kept.")
    else:
        print(f"Repair found with {len(best)} centers:", [hex(c) for c in best])


if __name__ == "__main__":
    main()
