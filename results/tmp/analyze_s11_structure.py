#!/usr/bin/env python3
"""
Deep structural analysis of Λ₁₅(1¹¹) SAT solution
Goal: Identify the mathematical structure underlying the perfect partition
"""

import numpy as np
from collections import defaultdict
from itertools import combinations

# Generate standard Hamming(15) code via parity check matrix
def generate_hamming_code():
    """Generate the [15,11,3] Hamming code codewords"""
    # Parity check matrix H: each column is binary rep of 1..15
    H = np.array([[int(b) for b in format(i, '04b')] for i in range(1, 16)]).T

    codewords = []
    for info_bits in range(2**11):
        # Generate all info bit combinations
        info = np.array([int(b) for b in format(info_bits, '011b')])
        # Systematic encoding: info bits + parity bits
        # But Hamming is not systematic in standard form, so enumerate
        pass

    # Alternative: enumerate all 2^15 and check syndrome = 0
    codewords = []
    for x in range(2**15):
        v = np.array([int(b) for b in format(x, '015b')])
        syndrome = (H @ v) % 2
        if np.all(syndrome == 0):
            codewords.append(v)

    return np.array(codewords)

def has_circular_ones(v, s):
    """Check if v has s consecutive 1s in circular form"""
    n = len(v)
    doubled = list(v) + list(v)
    for i in range(n):
        if all(doubled[i+j] == 1 for j in range(s)):
            return True
    return False

def hamming_distance(v1, v2):
    """Compute Hamming distance between two vectors"""
    return np.sum(v1 != v2)

def compute_syndrome(v, H):
    """Compute syndrome of v under parity check matrix H"""
    return tuple((H @ v) % 2)

def analyze_structure():
    print("=" * 70)
    print("Deep Structural Analysis of Λ₁₅(1¹¹) SAT Solution")
    print("=" * 70)

    # Load SAT solution
    sat_centers = []
    with open('lambda_15_11_centers.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and '.' in line and '(' in line:
                # Parse lines like "   1. 000000000000000  (weight= 0, ball=16)"
                parts = line.split('.')
                if len(parts) >= 2:
                    binary_part = parts[1].strip().split()[0]
                    if len(binary_part) == 15 and all(c in '01' for c in binary_part):
                        sat_centers.append(np.array([int(b) for b in binary_part]))

    sat_centers = np.array(sat_centers)
    print(f"\nLoaded {len(sat_centers)} SAT centers")

    # Generate Hamming code
    print("\nGenerating Hamming(15) code...")
    hamming = generate_hamming_code()
    print(f"Generated {len(hamming)} Hamming codewords")

    # Parity check matrix
    H = np.array([[int(b) for b in format(i, '04b')] for i in range(1, 16)]).T

    # Analyze syndrome distribution of SAT centers
    print("\n" + "=" * 70)
    print("SYNDROME ANALYSIS")
    print("=" * 70)

    syndrome_counts = defaultdict(int)
    syndrome_to_centers = defaultdict(list)

    for i, c in enumerate(sat_centers):
        s = compute_syndrome(c, H)
        syndrome_counts[s] += 1
        syndrome_to_centers[s].append(i)

    print(f"\nUnique syndromes: {len(syndrome_counts)}")
    print(f"\nSyndrome distribution (sorted by count):")
    sorted_syndromes = sorted(syndrome_counts.items(), key=lambda x: -x[1])
    for syn, count in sorted_syndromes[:20]:
        print(f"  {syn}: {count} centers")

    # Check which syndromes
    print(f"\nSyndrome (0,0,0,0) count: {syndrome_counts[(0,0,0,0)]}")

    # Analyze coset structure
    print("\n" + "=" * 70)
    print("COSET STRUCTURE ANALYSIS")
    print("=" * 70)

    # For each syndrome s, check if centers form a coset of Hamming
    # A coset H + v has the property that all elements have the same syndrome

    hamming_set = set(tuple(c) for c in hamming)
    sat_set = set(tuple(c) for c in sat_centers)

    # Count how many SAT centers are in each Hamming coset
    coset_membership = defaultdict(list)
    for c in sat_centers:
        syn = compute_syndrome(c, H)
        coset_membership[syn].append(tuple(c))

    # Each Hamming coset has 2048 elements
    print(f"\nCoset coverage analysis:")
    total_in_cosets = 0
    for syn in sorted(coset_membership.keys()):
        count = len(coset_membership[syn])
        if count > 1:
            print(f"  Syndrome {syn}: {count} centers")
        total_in_cosets += count

    # Check if SAT solution is a union of coset representatives
    print("\n" + "=" * 70)
    print("NON-LINEAR CODE STRUCTURE ANALYSIS")
    print("=" * 70)

    # Check: is SAT solution a perfect 1-code?
    # A perfect 1-code in Q_15 has exactly 2^15 / 16 = 2048 codewords
    print(f"\nSAT solution size: {len(sat_centers)}")
    print(f"Expected for perfect 1-code: 2048")
    print(f"Difference: {2048 - len(sat_centers)}")

    # Check minimum distance
    print("\nChecking minimum distance...")
    min_dist = float('inf')
    min_pair = None
    for i in range(min(len(sat_centers), 200)):  # Sample first 200
        for j in range(i+1, min(len(sat_centers), 200)):
            d = hamming_distance(sat_centers[i], sat_centers[j])
            if d < min_dist:
                min_dist = d
                min_pair = (i, j)
    print(f"Minimum distance (sampled): {min_dist}")

    # Check: is SAT solution a modification of Hamming?
    print("\n" + "=" * 70)
    print("HAMMING MODIFICATION ANALYSIS")
    print("=" * 70)

    # Find Hamming codewords that are NOT in SAT
    hamming_not_in_sat = [c for c in hamming if tuple(c) not in sat_set]
    print(f"\nHamming codewords NOT in SAT: {len(hamming_not_in_sat)}")

    # Find SAT centers that are NOT in Hamming
    sat_not_in_hamming = [c for c in sat_centers if tuple(c) not in hamming_set]
    print(f"SAT centers NOT in Hamming: {len(sat_not_in_hamming)}")

    # Analyze the "bad" Hamming codewords (those with circular 1^11)
    print("\n" + "=" * 70)
    print("BAD CODEWORD ANALYSIS")
    print("=" * 70)

    bad_hamming = []
    for c in hamming:
        if has_circular_ones(c, 11):
            bad_hamming.append(c)
            wt = np.sum(c)
            # Find position of zeros
            zeros = [i for i in range(15) if c[i] == 0]
            print(f"  Bad: {''.join(map(str, c))} (weight {wt}, zeros at {zeros})")

    print(f"\nTotal bad Hamming codewords: {len(bad_hamming)}")

    # For each bad codeword, check if it's in SAT
    print("\nBad Hamming codewords in SAT solution:")
    for c in bad_hamming:
        in_sat = tuple(c) in sat_set
        print(f"  {''.join(map(str, c))}: {'YES' if in_sat else 'NO'}")

    # Analyze the replacements
    print("\n" + "=" * 70)
    print("REPLACEMENT PATTERN ANALYSIS")
    print("=" * 70)

    # The missing Hamming codewords
    print(f"\nMissing Hamming codewords (not in SAT):")
    for c in hamming_not_in_sat[:20]:
        wt = np.sum(c)
        # Check if bad
        is_bad = has_circular_ones(c, 11)
        # Find zeros
        zeros = [i for i in range(15) if c[i] == 0]
        print(f"  {''.join(map(str, c))} (wt {wt}, zeros at {zeros}, bad={is_bad})")

    if len(hamming_not_in_sat) > 20:
        print(f"  ... and {len(hamming_not_in_sat) - 20} more")

    # Non-Hamming centers in SAT
    print(f"\nNon-Hamming SAT centers (first 20):")
    sat_non_hamming_arr = np.array(sat_not_in_hamming)
    for i, c in enumerate(sat_not_in_hamming[:20]):
        syn = compute_syndrome(c, H)
        wt = np.sum(c)
        print(f"  {''.join(map(str, c))} (wt {wt}, syndrome {syn})")

    if len(sat_not_in_hamming) > 20:
        print(f"  ... and {len(sat_not_in_hamming) - 20} more")

    # Analyze relationship between non-Hamming centers
    print("\n" + "=" * 70)
    print("NON-HAMMING CENTER STRUCTURE")
    print("=" * 70)

    # Group by syndrome
    non_hamming_by_syn = defaultdict(list)
    for c in sat_not_in_hamming:
        syn = compute_syndrome(c, H)
        non_hamming_by_syn[syn].append(c)

    print(f"\nNon-Hamming centers grouped by syndrome:")
    for syn in sorted(non_hamming_by_syn.keys(), key=lambda x: -len(non_hamming_by_syn[x])):
        centers = non_hamming_by_syn[syn]
        # Syndrome position (which bit is flipped to get to Hamming)
        # syndrome s corresponds to column s of H
        syn_int = int(''.join(map(str, syn)), 2)
        print(f"  Syndrome {syn} (col {syn_int}): {len(centers)} centers")
        if len(centers) <= 5:
            for c in centers:
                print(f"    {''.join(map(str, c))}")

    # Check: for each non-Hamming center, find its nearest Hamming codeword
    print("\n" + "=" * 70)
    print("NEAREST HAMMING CODEWORD ANALYSIS")
    print("=" * 70)

    distance_distribution = defaultdict(int)
    for c in sat_not_in_hamming[:100]:  # Sample first 100
        min_dist_to_hamming = min(hamming_distance(c, h) for h in hamming)
        distance_distribution[min_dist_to_hamming] += 1

    print("\nDistance from non-Hamming SAT centers to nearest Hamming codeword:")
    for d in sorted(distance_distribution.keys()):
        print(f"  Distance {d}: {distance_distribution[d]} centers")

    # Key insight: in a perfect 1-code, every vertex is at distance <= 1 from exactly one codeword
    # Non-Hamming centers are at distance 1 from Hamming codewords (by syndrome)
    # So they "replace" a Hamming codeword's role

    print("\n" + "=" * 70)
    print("REPLACEMENT MECHANISM ANALYSIS")
    print("=" * 70)

    # For each missing Hamming codeword h, find which SAT center covers its ball
    print("\nFor each removed Hamming codeword, find its replacement(s):")

    for i, h in enumerate(hamming_not_in_sat[:10]):
        h_tuple = tuple(h)
        # Find which SAT center is at distance <= 1 from h
        covering_sat = []
        for j, c in enumerate(sat_centers):
            if hamming_distance(h, c) <= 1:
                covering_sat.append((j, hamming_distance(h, c), ''.join(map(str, c))))

        h_str = ''.join(map(str, h))
        is_bad = has_circular_ones(h, 11)
        print(f"\n  Removed: {h_str} (bad={is_bad})")
        for idx, dist, cstr in covering_sat:
            in_hamming = tuple(sat_centers[idx]) in hamming_set
            print(f"    Covered by #{idx+1}: {cstr} (dist={dist}, Hamming={in_hamming})")

    # Analyze the algebraic structure
    print("\n" + "=" * 70)
    print("ALGEBRAIC STRUCTURE ANALYSIS")
    print("=" * 70)

    # Check if SAT solution is closed under some operation
    # Test: XOR closure
    print("\nChecking XOR closure (sampling)...")
    xor_in_sat = 0
    xor_not_in_sat = 0

    sample_size = min(100, len(sat_centers))
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            xor_result = sat_centers[i] ^ sat_centers[j]
            if tuple(xor_result) in sat_set:
                xor_in_sat += 1
            else:
                xor_not_in_sat += 1

    total_xor = xor_in_sat + xor_not_in_sat
    print(f"  XOR pairs in SAT: {xor_in_sat} ({100*xor_in_sat/total_xor:.1f}%)")
    print(f"  XOR pairs not in SAT: {xor_not_in_sat} ({100*xor_not_in_sat/total_xor:.1f}%)")

    if xor_not_in_sat > 0:
        print("  => SAT solution is NOT a linear code")

    # Check which dual-distance / covering radius
    print("\n" + "=" * 70)
    print("COVERING PROPERTIES")
    print("=" * 70)

    # Sample vertices and check covering
    print("\nVerifying covering property (sampling allowed vertices)...")

    # Generate some allowed vertices and check covering
    covered_count = 0
    doubly_covered = 0
    sample_vertices = 500

    for _ in range(sample_vertices):
        # Random vertex
        v = np.random.randint(0, 2, 15)
        if has_circular_ones(v, 11):
            continue  # Skip forbidden vertices

        # Count how many centers cover this vertex
        covering = sum(1 for c in sat_centers if hamming_distance(v, c) <= 1)
        if covering > 0:
            covered_count += 1
        if covering > 1:
            doubly_covered += 1

    print(f"  Sample of {sample_vertices} allowed vertices:")
    print(f"  Covered: {covered_count} ({100*covered_count/sample_vertices:.1f}%)")
    print(f"  Doubly covered: {doubly_covered}")

    print("\n" + "=" * 70)
    print("SUMMARY AND CONJECTURES")
    print("=" * 70)

    print(f"""
Key Observations:
1. SAT solution has {len(sat_centers)} centers (vs 2048 for Hamming)
2. {syndrome_counts[(0,0,0,0)]} centers have zero syndrome (are Hamming codewords)
3. {len(sat_not_in_hamming)} centers are non-Hamming
4. {len(hamming_not_in_sat)} Hamming codewords were removed
5. XOR closure: {100*xor_in_sat/total_xor:.1f}% of XOR pairs in SAT

Structural Hypothesis:
The SAT solution appears to be a non-linear perfect 1-code obtained by:
- Starting with Hamming code (2048 codewords)
- Removing 1 + {len(hamming_not_in_sat) - 1} codewords (all-ones + others with circular 1^11)
- Adding {len(sat_not_in_hamming)} replacement centers at strategic positions

This is NOT a simple "Hamming - bad + repairs" structure.
The replacements form a complex non-linear pattern.
""")

if __name__ == "__main__":
    analyze_structure()
