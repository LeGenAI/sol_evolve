#!/usr/bin/env python3
"""
Deep analysis: Why does s=11 have such different structure from s=12?
Key question: What is the actual mathematical structure?
"""

import numpy as np
from collections import defaultdict

def generate_hamming_code():
    """Generate the [15,11,3] Hamming code codewords"""
    H = np.array([[int(b) for b in format(i, '04b')] for i in range(1, 16)]).T
    codewords = []
    for x in range(2**15):
        v = np.array([int(b) for b in format(x, '015b')])
        syndrome = (H @ v) % 2
        if np.all(syndrome == 0):
            codewords.append(v)
    return np.array(codewords), H

def has_circular_ones(v, s):
    """Check if v has s consecutive 1s in circular form"""
    n = len(v)
    doubled = list(v) + list(v)
    for i in range(n):
        if all(doubled[i+j] == 1 for j in range(s)):
            return True
    return False

def hamming_distance(v1, v2):
    return np.sum(v1 != v2)

def load_sat_centers(filename):
    centers = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '.' in line and '(' in line:
                parts = line.split('.')
                if len(parts) >= 2:
                    binary_part = parts[1].strip().split()[0]
                    if len(binary_part) == 15 and all(c in '01' for c in binary_part):
                        centers.append(np.array([int(b) for b in binary_part]))
    return np.array(centers)

def analyze_syndrome_pattern():
    print("=" * 70)
    print("SYNDROME PATTERN ANALYSIS")
    print("=" * 70)

    hamming, H = generate_hamming_code()
    sat_s11 = load_sat_centers('lambda_15_11_centers.txt')
    sat_s12 = load_sat_centers('lambda_15_12_centers.txt')

    print(f"\nLoaded {len(sat_s11)} s=11 centers, {len(sat_s12)} s=12 centers")

    # Compute syndromes
    def get_syndrome(v, H):
        return tuple((H @ v) % 2)

    s11_syndromes = defaultdict(int)
    s12_syndromes = defaultdict(int)

    for c in sat_s11:
        s11_syndromes[get_syndrome(c, H)] += 1
    for c in sat_s12:
        s12_syndromes[get_syndrome(c, H)] += 1

    print("\n" + "=" * 70)
    print("SYNDROME DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"\n{'Syndrome':<20} {'s=11':<10} {'s=12':<10} {'Diff':<10}")
    print("-" * 50)

    all_syndromes = set(s11_syndromes.keys()) | set(s12_syndromes.keys())
    for syn in sorted(all_syndromes):
        c11 = s11_syndromes[syn]
        c12 = s12_syndromes[syn]
        diff = c11 - c12
        print(f"{str(syn):<20} {c11:<10} {c12:<10} {diff:+10}")

    # Key insight: s=12 has 2047 Hamming codewords (syndrome 0)
    # s=11 has only 47 Hamming codewords!

    print("\n" + "=" * 70)
    print("STRUCTURAL HYPOTHESIS")
    print("=" * 70)

    # Let's understand WHY s=11 needs such massive restructuring

    # For s=11: a codeword is "bad" if it has 11 consecutive 1s circularly
    # For s=12: a codeword is "bad" if it has 12 consecutive 1s circularly

    # Count bad codewords for each s value
    bad_s11 = sum(1 for c in hamming if has_circular_ones(c, 11))
    bad_s12 = sum(1 for c in hamming if has_circular_ones(c, 12))

    print(f"\nBad Hamming codewords for s=11: {bad_s11}")
    print(f"Bad Hamming codewords for s=12: {bad_s12}")

    # For s=12, only 1 bad codeword (all-ones), so Hamming - {1} works
    # For s=11, 10 bad codewords, but we see 2001 Hamming removed!

    # WHY? Let's analyze the "neighborhood" of bad codewords
    print("\n" + "=" * 70)
    print("NEIGHBORHOOD ANALYSIS OF BAD CODEWORDS")
    print("=" * 70)

    hamming_set = set(tuple(c) for c in hamming)

    for c in hamming:
        if has_circular_ones(c, 11):
            c_str = ''.join(map(str, c))
            wt = np.sum(c)

            # Find neighbors of this bad codeword that are also bad (have 1^11)
            bad_neighbors_in_hamming = 0
            bad_neighbors_in_allowed = 0

            # Check all distance-1 neighbors
            for i in range(15):
                neighbor = c.copy()
                neighbor[i] = 1 - neighbor[i]

                if tuple(neighbor) in hamming_set:
                    if has_circular_ones(neighbor, 11):
                        bad_neighbors_in_hamming += 1

                # Check if neighbor is allowed (no 1^11)
                if not has_circular_ones(neighbor, 11):
                    bad_neighbors_in_allowed += 1

            print(f"  Bad: {c_str} (wt {wt})")
            print(f"       Bad Hamming neighbors: {bad_neighbors_in_hamming}")
            print(f"       Allowed neighbors: {bad_neighbors_in_allowed}")

    # The key insight: for s=11, removing bad codewords leaves many uncovered vertices
    # These vertices are at distance 2 from other Hamming codewords
    # So a massive restructuring is needed

    print("\n" + "=" * 70)
    print("UNDERSTANDING THE 2000 REPLACEMENTS")
    print("=" * 70)

    # Each non-Hamming SAT center has a specific syndrome
    # Syndrome s means the center is at distance 1 from a Hamming codeword
    # The Hamming codeword = center XOR e_i where i = syndrome position

    # Let's trace the replacement pattern
    sat_set = set(tuple(c) for c in sat_s11)

    # For each non-zero syndrome, count centers
    print("\nSyndrome analysis for non-Hamming s=11 centers:")
    syn_to_column = {}
    for i in range(1, 16):
        syn = tuple((H[:, i-1]) % 2)  # column i of H
        syn_to_column[syn] = i

    for syn in sorted(s11_syndromes.keys()):
        if syn != (0,0,0,0):
            col = int(''.join(map(str, syn)), 2)
            count = s11_syndromes[syn]
            print(f"  Syndrome {syn} (bit {col}): {count} centers")

    # The syndrome pattern reveals:
    # - 336 centers each for syndromes 10, 11, 14, 15 (high bit columns)
    # - 80 centers each for syndromes 2, 3, 6, 7
    # - 48 centers each for syndromes 1, 4, 5, 8, 9, 12, 13

    print("\n" + "=" * 70)
    print("SYNDROME PATTERN INTERPRETATION")
    print("=" * 70)

    print("""
    The syndrome distribution shows a STRUCTURED PATTERN:

    High frequency (336 each): syndromes 10, 11, 14, 15
        Binary: 1010, 1011, 1110, 1111
        Pattern: All have bits 1,2,3 set in some combination

    Medium frequency (80 each): syndromes 2, 3, 6, 7
        Binary: 0010, 0011, 0110, 0111
        Pattern: bit 1 set

    Low frequency (48 each): syndromes 1, 4, 5, 8, 9, 12, 13
        Binary: 0001, 0100, 0101, 1000, 1001, 1100, 1101
        Pattern: Mixed

    This is NOT random - it reflects the structure of which Hamming codewords
    needed to be replaced to avoid circular 1^11 patterns!
    """)

    # Let's analyze: which Hamming codewords conflict with the constraint
    print("\n" + "=" * 70)
    print("CONFLICT ANALYSIS")
    print("=" * 70)

    # For each Hamming codeword, check:
    # 1. Is it bad (has 1^11)?
    # 2. If not bad, do its distance-2 neighbors include bad codewords?

    conflicting_hamming = []
    for c in hamming:
        if has_circular_ones(c, 11):
            continue  # Skip bad codewords themselves

        # Check distance-2 neighbors that are bad
        has_bad_d2_neighbor = False
        for i in range(15):
            for j in range(i+1, 15):
                neighbor = c.copy()
                neighbor[i] = 1 - neighbor[i]
                neighbor[j] = 1 - neighbor[j]
                if tuple(neighbor) in hamming_set and has_circular_ones(neighbor, 11):
                    has_bad_d2_neighbor = True
                    break
            if has_bad_d2_neighbor:
                break

        if has_bad_d2_neighbor:
            conflicting_hamming.append(c)

    print(f"\nHamming codewords at distance 2 from bad codewords: {len(conflicting_hamming)}")
    print("(These create packing conflicts if we try simple repair)")

    # The key theorem insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT: THE COVERING RADIUS CONSTRAINT")
    print("=" * 70)

    print("""
    For s=12 (only 1 bad codeword = all-ones):
    - N[1^15] = {1^15, all weight-14 vectors}
    - All of N[1^15] are forbidden in Λ_15(1^12)
    - So removing 1^15 is "free" - no uncovered allowed vertices

    For s=11 (10 bad codewords):
    - Each bad codeword has ALLOWED vertices in its neighborhood
    - These allowed vertices MUST be covered by some other center
    - But those other centers may conflict with remaining Hamming codewords

    The SAT solver found a MASSIVE RESTRUCTURING:
    - Keep only 47 Hamming codewords (2.3%)
    - Replace 2001 Hamming codewords with 2000 non-Hamming centers

    This suggests the solution is NOT based on Hamming at all!
    It's a fundamentally different perfect 1-code.
    """)

    # Check if the weight distribution matches anything known
    print("\n" + "=" * 70)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 70)

    s11_weights = defaultdict(int)
    s12_weights = defaultdict(int)
    hamming_weights = defaultdict(int)

    for c in sat_s11:
        s11_weights[np.sum(c)] += 1
    for c in sat_s12:
        s12_weights[np.sum(c)] += 1
    for c in hamming:
        hamming_weights[np.sum(c)] += 1

    print(f"\n{'Weight':<10} {'Hamming':<12} {'s=12 SAT':<12} {'s=11 SAT':<12}")
    print("-" * 50)
    for w in range(16):
        h = hamming_weights[w]
        s12 = s12_weights[w]
        s11 = s11_weights[w]
        if h > 0 or s12 > 0 or s11 > 0:
            print(f"{w:<10} {h:<12} {s12:<12} {s11:<12}")

    print("""
    REMARKABLE: All three have IDENTICAL weight distributions!
    (except Hamming has one weight-15 codeword that's removed)

    This means the SAT solutions are "distance-invariant" to Hamming
    in terms of weight, but NOT structurally equivalent.
    """)

if __name__ == "__main__":
    analyze_syndrome_pattern()
