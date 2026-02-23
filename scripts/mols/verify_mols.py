#!/usr/bin/env python3
"""
Final verification of the two non-symmetric 5x5 MOLS
"""

def verify_all():
    square1 = [
        [0, 4, 2, 3, 1],
        [3, 2, 0, 1, 4],
        [2, 1, 4, 0, 3],
        [1, 0, 3, 4, 2],
        [4, 3, 1, 2, 0]
    ]
    
    square2 = [
        [4, 3, 2, 0, 1],
        [3, 0, 1, 2, 4],
        [1, 4, 0, 3, 2],
        [0, 2, 4, 1, 3],
        [2, 1, 3, 4, 0]
    ]
    
    n = 5
    all_passed = True
    
    print("="*60)
    print("FINAL VERIFICATION OF TWO NON-SYMMETRIC 5x5 MOLS")
    print("="*60)
    
    # Test 1: Latin square property for Square 1
    print("\n1. Verifying Square 1 is a Latin square...")
    for i in range(n):
        if sorted(square1[i]) != list(range(n)):
            print(f"   ✗ Row {i} failed: {square1[i]}")
            all_passed = False
    for j in range(n):
        col = [square1[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            print(f"   ✗ Column {j} failed: {col}")
            all_passed = False
    if all_passed:
        print("   ✓ Square 1 is a valid Latin square")
    
    # Test 2: Latin square property for Square 2
    print("\n2. Verifying Square 2 is a Latin square...")
    for i in range(n):
        if sorted(square2[i]) != list(range(n)):
            print(f"   ✗ Row {i} failed: {square2[i]}")
            all_passed = False
    for j in range(n):
        col = [square2[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            print(f"   ✗ Column {j} failed: {col}")
            all_passed = False
    if all_passed:
        print("   ✓ Square 2 is a valid Latin square")
    
    # Test 3: Orthogonality
    print("\n3. Verifying orthogonality...")
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (square1[i][j], square2[i][j])
            if pair in pairs:
                print(f"   ✗ Duplicate pair {pair} at position ({i},{j})")
                all_passed = False
            pairs.add(pair)
    
    if len(pairs) == n * n:
        print(f"   ✓ All {n*n} pairs are unique - squares are orthogonal")
    else:
        print(f"   ✗ Only {len(pairs)} unique pairs (expected {n*n})")
        all_passed = False
    
    # Test 4: Non-symmetry of Square 1
    print("\n4. Verifying Square 1 is non-symmetric...")
    is_symmetric = True
    differences = []
    for i in range(n):
        for j in range(i+1, n):
            if square1[i][j] != square1[j][i]:
                differences.append(f"({i},{j}): {square1[i][j]} != {square1[j][i]}")
                is_symmetric = False
    
    if not is_symmetric:
        print(f"   ✓ Square 1 is non-symmetric")
    else:
        print(f"   ✗ Square 1 is symmetric")
        all_passed = False
    
    # Test 5: Non-symmetry of Square 2
    print("\n5. Verifying Square 2 is non-symmetric...")
    is_symmetric = True
    differences = []
    for i in range(n):
        for j in range(i+1, n):
            if square2[i][j] != square2[j][i]:
                differences.append(f"({i},{j}): {square2[i][j]} != {square2[j][i]}")
                is_symmetric = False
    
    if not is_symmetric:
        print(f"   ✓ Square 2 is non-symmetric")
    else:
        print(f"   ✗ Square 2 is symmetric")
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("Successfully found two non-symmetric 5x5 MOLS!")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    # Print squares for latex
    print("\nLaTeX Format:")
    print("Square 1:")
    for row in square1:
        print(" & ".join(str(x) for x in row) + " \\\\")
    print("\nSquare 2:")
    for row in square2:
        print(" & ".join(str(x) for x in row) + " \\\\")

if __name__ == "__main__":
    verify_all()
