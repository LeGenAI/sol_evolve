#!/usr/bin/env python3
"""
Non-Hamming Repair Strategy for Perfect Partition in Λ_n(1^s)

Key insight from previous analysis:
- n=7: 3 bad codewords, 8 escaping vertices → need repair
- n=15: 3 bad codewords, 24 escaping vertices → need repair
- Simple deletion doesn't work for ANY coset!

New strategy: Greedy repair with SAT backup
1. Start from Hamming (or coset)
2. Remove bad codewords
3. Find repair centers for uncovered allowed vertices
4. Repair centers must be:
   - In allowed region
   - Distance ≥ 3 from all remaining centers
   - Cover at least one uncovered vertex

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict
from itertools import combinations


def has_circular_run(v: int, n: int, s: int) -> bool:
    """Check if v has s consecutive circular 1s."""
    if s > n:
        return False
    bits = format(v, f'0{n}b')
    doubled = bits + bits[:-1]
    return '1' * s in doubled


def hamming_weight(v: int) -> int:
    return bin(v).count('1')


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def get_ball(v: int, n: int) -> Set[int]:
    """Get closed ball of radius 1."""
    ball = {v}
    for i in range(n):
        ball.add(v ^ (1 << i))
    return ball


def compute_syndrome(v: int, k: int) -> int:
    """Compute syndrome of v."""
    n = (1 << k) - 1
    syndrome = 0
    for i in range(n):
        if (v >> i) & 1:
            syndrome ^= (i + 1)
    return syndrome


def generate_hamming_code(k: int) -> Set[int]:
    """Generate Hamming code via syndrome check."""
    n = (1 << k) - 1
    return {v for v in range(1 << n) if compute_syndrome(v, k) == 0}


def get_allowed_vertices(n: int, s: int) -> Set[int]:
    """Get all vertices in Λ_n(1^s)."""
    return {v for v in range(1 << n) if not has_circular_run(v, n, s)}


def greedy_repair(base_code: Set[int], n: int, s: int, verbose: bool = True) -> Optional[Set[int]]:
    """
    Greedy repair strategy.

    1. Remove bad codewords from base_code
    2. Find uncovered allowed vertices
    3. Greedily add repair centers
    """
    allowed = get_allowed_vertices(n, s)

    # Remove bad codewords
    good_centers = {c for c in base_code if c in allowed}
    bad_centers = base_code - good_centers

    if verbose:
        print(f"Base code: {len(base_code)}, Good: {len(good_centers)}, Bad: {len(bad_centers)}")

    # Compute current coverage
    covered = set()
    for c in good_centers:
        covered.update(get_ball(c, n) & allowed)

    uncovered = allowed - covered

    if verbose:
        print(f"After removing bad: {len(uncovered)} uncovered vertices")

    if not uncovered:
        return good_centers

    # Find repair candidates
    # Candidate must be:
    # 1. In allowed region
    # 2. Distance ≥ 3 from all current centers
    # 3. Cover at least one uncovered vertex

    repair_centers = set()
    current_centers = good_centers.copy()

    iteration = 0
    while uncovered:
        iteration += 1

        # Find all valid candidates
        candidates = []
        for v in allowed:
            if v in current_centers:
                continue

            # Check distance constraint
            min_dist = min((hamming_distance(v, c) for c in current_centers), default=n+1)
            if min_dist < 3:
                continue

            # Count coverage
            ball = get_ball(v, n)
            covers = ball & uncovered
            if covers:
                candidates.append((v, len(covers), covers))

        if not candidates:
            if verbose:
                print(f"  Iteration {iteration}: No valid candidates! {len(uncovered)} still uncovered")
            return None  # Failed

        # Pick best candidate (covers most uncovered)
        candidates.sort(key=lambda x: -x[1])
        best, best_count, best_covers = candidates[0]

        if verbose and iteration <= 10:
            print(f"  Iteration {iteration}: Adding {format(best, f'0{n}b')} covering {best_count}")

        # Add repair center
        repair_centers.add(best)
        current_centers.add(best)
        covered.update(get_ball(best, n) & allowed)
        uncovered = allowed - covered

    if verbose:
        print(f"Repair complete! Added {len(repair_centers)} centers")

    return current_centers


def verify_perfect_partition(centers: Set[int], n: int, s: int) -> Dict:
    """Verify perfect partition."""
    allowed = get_allowed_vertices(n, s)

    # Check centers in allowed
    invalid = centers - allowed

    # Check packing
    packing_ok = True
    for c1, c2 in combinations(centers, 2):
        if hamming_distance(c1, c2) < 3:
            packing_ok = False
            break

    # Check covering
    covered = defaultdict(list)
    for c in centers:
        for v in get_ball(c, n) & allowed:
            covered[v].append(c)

    uncovered = allowed - set(covered.keys())
    multiply = {v for v, cs in covered.items() if len(cs) > 1}

    return {
        'valid': len(invalid) == 0 and packing_ok and len(uncovered) == 0 and len(multiply) == 0,
        'invalid_centers': len(invalid),
        'packing_ok': packing_ok,
        'uncovered': len(uncovered),
        'multiply_covered': len(multiply),
        'total_centers': len(centers)
    }


def experiment_n7_repair():
    """Test repair strategy for n=7."""
    print("="*70)
    print("Repair Strategy: n=7, s=4")
    print("="*70)

    k = 3
    n = 7
    s = 4

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    print(f"|Hamming| = {len(hamming)}")
    print(f"|Λ_7(1^4)| = {len(allowed)}")

    # Try greedy repair
    print("\nGreedy repair from Hamming:")
    result = greedy_repair(hamming, n, s)

    if result:
        verify = verify_perfect_partition(result, n, s)
        print(f"\nVerification: {verify}")

        if verify['valid']:
            print("\n*** SUCCESS! Found perfect partition via repair! ***")
            print(f"Centers: {len(result)}")

            # Compare with Hamming
            hamming_in_result = result & hamming
            print(f"Overlap with Hamming: {len(hamming_in_result)} ({100*len(hamming_in_result)/len(result):.1f}%)")

            # Show non-Hamming centers
            non_hamming = result - hamming
            print(f"\nNon-Hamming centers ({len(non_hamming)}):")
            for c in sorted(non_hamming):
                print(f"  {format(c, f'0{n}b')} (syndrome={compute_syndrome(c, k)}, weight={hamming_weight(c)})")
    else:
        print("\nGreedy repair failed. Trying all cosets...")

        # Try all cosets
        for shift in range(1 << n):
            coset = {c ^ shift for c in hamming}
            result = greedy_repair(coset, n, s, verbose=False)
            if result:
                verify = verify_perfect_partition(result, n, s)
                if verify['valid']:
                    print(f"\nFound via coset shift={shift}!")
                    print(f"Centers: {len(result)}")
                    break

    return result


def experiment_n15_repair():
    """Test repair strategy for n=15."""
    print("="*70)
    print("Repair Strategy: n=15, s=12")
    print("="*70)

    k = 4
    n = 15
    s = 12

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    print(f"|Hamming| = {len(hamming)}")
    print(f"|Λ_15(1^12)| = {len(allowed)}")

    # Try greedy repair
    print("\nGreedy repair from Hamming:")
    result = greedy_repair(hamming, n, s)

    if result:
        verify = verify_perfect_partition(result, n, s)
        print(f"\nVerification: {verify}")

        if verify['valid']:
            print("\n*** SUCCESS! Found perfect partition via repair! ***")
            print(f"Centers: {len(result)}")

            # Compare with Hamming
            hamming_in_result = result & hamming
            print(f"Overlap with Hamming: {len(hamming_in_result)} ({100*len(hamming_in_result)/len(result):.1f}%)")
    else:
        print("\nGreedy repair failed!")

    return result


def experiment_n15_s11_repair():
    """Test repair strategy for n=15, s=11 (more restrictive)."""
    print("="*70)
    print("Repair Strategy: n=15, s=11")
    print("="*70)

    k = 4
    n = 15
    s = 11

    hamming = generate_hamming_code(k)
    allowed = get_allowed_vertices(n, s)

    print(f"|Hamming| = {len(hamming)}")
    print(f"|Λ_15(1^11)| = {len(allowed)}")

    # Count bad codewords
    bad = {c for c in hamming if c not in allowed}
    print(f"Bad codewords: {len(bad)}")

    # Try greedy repair
    print("\nGreedy repair from Hamming:")
    result = greedy_repair(hamming, n, s)

    if result:
        verify = verify_perfect_partition(result, n, s)
        print(f"\nVerification: {verify}")

        if verify['valid']:
            print("\n*** SUCCESS! ***")
            hamming_in_result = result & hamming
            print(f"Overlap with Hamming: {len(hamming_in_result)} ({100*len(hamming_in_result)/len(result):.1f}%)")

    return result


def analyze_repair_structure(original: Set[int], repaired: Set[int], k: int, n: int, s: int):
    """Analyze the structure of repair."""
    hamming = generate_hamming_code(k)

    removed = original - repaired
    added = repaired - original

    print(f"\nRepair structure analysis:")
    print(f"  Original centers: {len(original)}")
    print(f"  Repaired centers: {len(repaired)}")
    print(f"  Removed: {len(removed)}")
    print(f"  Added: {len(added)}")

    # Syndrome distribution of removed
    print(f"\nRemoved centers (syndrome distribution):")
    syn_dist = defaultdict(int)
    for c in removed:
        syn_dist[compute_syndrome(c, k)] += 1
    for syn, count in sorted(syn_dist.items()):
        print(f"    Syndrome {syn}: {count}")

    # Syndrome distribution of added
    print(f"\nAdded centers (syndrome distribution):")
    syn_dist = defaultdict(int)
    for c in added:
        syn_dist[compute_syndrome(c, k)] += 1
    for syn, count in sorted(syn_dist.items()):
        print(f"    Syndrome {syn}: {count}")

    # Check if result is linear
    # A set is linear if closed under XOR
    result_list = list(repaired)
    is_linear = True
    for i, c1 in enumerate(result_list[:100]):  # Sample
        for c2 in result_list[i+1:i+100]:
            xor_val = c1 ^ c2
            if xor_val not in repaired:
                is_linear = False
                break
        if not is_linear:
            break

    print(f"\nLinearity check (sampled): {'Linear' if is_linear else 'Non-linear'}")


def main():
    """Main experiments."""
    print("Non-Hamming Repair Strategy")
    print("="*70)

    # n=7 first
    result7 = experiment_n7_repair()

    if result7:
        hamming7 = generate_hamming_code(3)
        analyze_repair_structure(hamming7, result7, 3, 7, 4)

    print("\n")

    # n=15
    result15 = experiment_n15_repair()

    if result15:
        hamming15 = generate_hamming_code(4)
        analyze_repair_structure(hamming15, result15, 4, 15, 12)

    print("\n")

    # n=15, s=11
    result15_11 = experiment_n15_s11_repair()

    if result15_11:
        hamming15 = generate_hamming_code(4)
        analyze_repair_structure(hamming15, result15_11, 4, 15, 11)


if __name__ == "__main__":
    main()
