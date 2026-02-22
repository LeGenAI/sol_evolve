#!/usr/bin/env python3
"""
Check inequivalence of FOUR [16,6,6] codes by analyzing their generator matrices.
"""

import numpy as np
from collections import Counter
import sys
import os

# File paths
base_path = "/Users/baegjaehyeon/CodeEvolve/src/kissat_code_search/results/cadical_16_6_6"
files = [
    os.path.join(base_path, "code_16_6_6_solution1_matrix.npy"),
    os.path.join(base_path, "code_16_6_6_solution2_matrix.npy"),
    os.path.join(base_path, "code_16_6_6_solution3_matrix.npy"),
    os.path.join(base_path, "code_16_6_6_solution4_matrix.npy")
]

def generate_all_codewords(G):
    """Generate all codewords from generator matrix G."""
    k = G.shape[0]
    codewords = []
    for i in range(2**k):
        info_bits = np.array([(i >> j) & 1 for j in range(k)], dtype=int)
        codeword = np.dot(info_bits, G) % 2
        codewords.append(tuple(codeword))
    return set(codewords)

def weight_enumerator(codewords):
    """Calculate weight enumerator from set of codewords."""
    weights = [sum(cw) for cw in codewords]
    return dict(sorted(Counter(weights).items()))

def verify_minimum_distance(codewords):
    """Verify the minimum distance of the code."""
    min_dist = float('inf')
    for cw in codewords:
        weight = sum(cw)
        if weight > 0:  # Skip the all-zero codeword
            min_dist = min(min_dist, weight)
    return min_dist

def main():
    print("="*80)
    print("INEQUIVALENCE CHECK FOR FOUR [16,6,6] CODES")
    print("="*80)
    
    # Load matrices
    print("\nLoading generator matrices...")
    matrices = []
    try:
        for f in files:
            if os.path.exists(f):
                matrices.append(np.load(f))
            else:
                print(f"✗ File not found: {f}")
                sys.exit(1)
        print("✓ All 4 matrices loaded successfully")
    except Exception as e:
        print(f"✗ Error loading matrices: {e}")
        sys.exit(1)
    
    # Display matrices
    for idx, G in enumerate(matrices, 1):
        print("\n" + "-"*80)
        print(f"GENERATOR MATRIX {idx} (6×16):")
        print("-"*80)
        for i, row in enumerate(G):
            print(f"Row {i+1}: " + ' '.join(str(int(x)) for x in row))
    
    # Generate codewords
    print("\n" + "="*80)
    print("GENERATING CODEWORDS (2^6 = 64 codewords each)...")
    print("="*80)
    
    codeword_sets = [generate_all_codewords(G) for G in matrices]
    
    for idx, C in enumerate(codeword_sets, 1):
        print(f"✓ Code {idx}: {len(C)} codewords")
    
    # Verify minimum distance
    print("\n" + "="*80)
    print("MINIMUM DISTANCE VERIFICATION:")
    print("="*80)
    
    for idx, C in enumerate(codeword_sets, 1):
        d = verify_minimum_distance(C)
        print(f"Code {idx}: d_min = {d} {'✓' if d >= 6 else '✗'}")
    
    # Weight enumerators
    print("\n" + "="*80)
    print("WEIGHT ENUMERATORS:")
    print("="*80)
    
    weight_enums = [weight_enumerator(C) for C in codeword_sets]
    
    for idx, WE in enumerate(weight_enums, 1):
        print(f"\nCode {idx}: {WE}")
    
    # Codeword set comparison (definitive test)
    print("\n" + "="*80)
    print("CODEWORD SET COMPARISON (Definitive Test):")
    print("="*80)
    
    equiv_matrix = [[False]*4 for _ in range(4)]
    for i in range(4):
        equiv_matrix[i][i] = True
    
    print("\nPairwise equivalence:")
    for i in range(4):
        for j in range(i+1, 4):
            equiv = (codeword_sets[i] == codeword_sets[j])
            equiv_matrix[i][j] = equiv
            equiv_matrix[j][i] = equiv
            symbol = "✓" if equiv else "✗"
            print(f"  Code {i+1} ≡ Code {j+1}: {equiv} {symbol}")
    
    # Equivalence classes
    print("\n" + "="*80)
    print("EQUIVALENCE CLASSES:")
    print("="*80)
    
    # Find equivalence classes
    classes = []
    assigned = [False] * 4
    
    for i in range(4):
        if not assigned[i]:
            eq_class = [i+1]
            assigned[i] = True
            for j in range(i+1, 4):
                if equiv_matrix[i][j]:
                    eq_class.append(j+1)
                    assigned[j] = True
            classes.append(eq_class)
    
    print(f"\nFound {len(classes)} equivalence class(es):")
    for idx, eq_class in enumerate(classes, 1):
        codes_str = ", ".join([f"Code {c}" for c in eq_class])
        print(f"  Class {idx}: [{codes_str}]")
    
    if len(classes) == 4:
        print("\n✅✅✅✅ ALL FOUR CODES ARE MUTUALLY INEQUIVALENT ✅✅✅✅")
    else:
        print(f"\n⚠️ Found only {len(classes)} inequivalent codes.")

if __name__ == "__main__":
    main()
