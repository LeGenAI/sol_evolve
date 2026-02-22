#!/usr/bin/env python3
"""
Forbidden Region Lemma v2 - Corrected Characterization

Key insight: A vertex is forbidden iff it contains a circular run of s consecutive 1s.
This is equivalent to: the GAPS between consecutive zeros (circularly) include at least one gap ≥ s.

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

from typing import List, Tuple

# ============================================================================
# Corrected Lemma
# ============================================================================

def has_circular_ones(v: int, n: int, s: int) -> bool:
    """Check if v contains s consecutive 1s in circular form"""
    doubled = v | (v << n)
    mask = (1 << s) - 1
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False


def get_zero_positions(v: int, n: int) -> List[int]:
    """Get positions of zeros in v (0-indexed)"""
    return [i for i in range(n) if not ((v >> i) & 1)]


def max_gap_between_zeros(positions: List[int], n: int) -> int:
    """
    Compute the maximum gap between consecutive zeros (circularly).

    If there are k zeros at positions p_0 < p_1 < ... < p_{k-1},
    the gaps are:
    - p_1 - p_0 - 1 (number of 1s between p_0 and p_1)
    - p_2 - p_1 - 1
    - ...
    - (n - p_{k-1} + p_0) - 1 (wraparound gap)

    Actually, the gap is the number of 1s between consecutive zeros.
    """
    if not positions:
        return n  # All ones, gap is entire length

    if len(positions) == 1:
        return n - 1  # One zero, rest are ones

    k = len(positions)
    positions = sorted(positions)

    max_gap = 0

    for i in range(k):
        next_idx = (i + 1) % k
        # Gap = number of 1s between position[i] and position[next_idx]
        if next_idx > i:
            gap = positions[next_idx] - positions[i] - 1
        else:
            # Wraparound
            gap = (n - positions[i] - 1) + positions[next_idx]

        max_gap = max(max_gap, gap)

    return max_gap


def verify_corrected_lemma(n: int, s: int, verbose: bool = False) -> bool:
    """
    Verify the Corrected Forbidden Region Lemma.

    LEMMA: v is forbidden (contains circular 1^s) iff
           the maximum gap between consecutive zeros (circularly) is ≥ s.

    Equivalently: v is ALLOWED iff all gaps between consecutive zeros are < s.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERIFYING CORRECTED LEMMA FOR n={n}, s={s}")
        print(f"{'='*60}")

    verified = True
    counterexamples = []

    for v in range(1 << n):
        is_forbidden = has_circular_ones(v, n, s)
        zero_pos = get_zero_positions(v, n)
        max_gap = max_gap_between_zeros(zero_pos, n)

        # Lemma: forbidden iff max_gap >= s
        lemma_predicts_forbidden = (max_gap >= s)

        if is_forbidden != lemma_predicts_forbidden:
            verified = False
            counterexamples.append({
                'v': v,
                'binary': bin(v),
                'is_forbidden': is_forbidden,
                'zero_pos': zero_pos,
                'max_gap': max_gap,
                'lemma_predicts': lemma_predicts_forbidden
            })

    if verbose:
        if verified:
            print(f"✓ CORRECTED LEMMA VERIFIED for all 2^{n} = {1 << n} vertices")
        else:
            print(f"✗ Lemma FAILED with {len(counterexamples)} counterexamples")
            for ce in counterexamples[:5]:
                print(f"  v={ce['binary']}: is_forbidden={ce['is_forbidden']}, "
                      f"zero_pos={ce['zero_pos']}, max_gap={ce['max_gap']}, "
                      f"lemma_predicts={ce['lemma_predicts']}")

    return verified


def count_by_gap_structure(n: int, s: int) -> None:
    """
    Analyze the structure of forbidden/allowed vertices.
    """
    print(f"\nGap structure analysis for n={n}, s={s}:")

    gap_counts = {}

    for v in range(1 << n):
        zero_pos = get_zero_positions(v, n)
        max_gap = max_gap_between_zeros(zero_pos, n)

        if max_gap not in gap_counts:
            gap_counts[max_gap] = {'forbidden': 0, 'allowed': 0}

        if has_circular_ones(v, n, s):
            gap_counts[max_gap]['forbidden'] += 1
        else:
            gap_counts[max_gap]['allowed'] += 1

    for gap in sorted(gap_counts.keys()):
        counts = gap_counts[gap]
        status = "FORBIDDEN" if gap >= s else "ALLOWED"
        print(f"  max_gap={gap}: {counts['forbidden']} forbidden, {counts['allowed']} allowed -> {status}")


# ============================================================================
# Main Results
# ============================================================================

def main():
    print("=" * 70)
    print("CORRECTED FORBIDDEN REGION LEMMA")
    print("=" * 70)

    print("""
    LEMMA (Forbidden Region Characterization - Corrected):

    Let Λ_n(1^s) be the generalized Lucas cube avoiding circular 1^s.

    A vertex v ∈ {0,1}^n is FORBIDDEN if and only if
    the maximum gap (run of 1s) between consecutive zeros (circularly) is at least s.

    Equivalently: v is ALLOWED in Λ_n(1^s) iff
    every maximal run of consecutive 1s in v (circularly) has length < s.

    PROOF:

    (⟹) If v is forbidden, it contains s consecutive 1s.
    These s ones form a gap between two zeros (or are all of v if v = 1^n).
    Thus the maximum gap is at least s.

    (⟸) If the maximum gap between consecutive zeros is g ≥ s,
    there exist g consecutive 1s in v, and g ≥ s implies circular 1^s exists.
    Therefore v is forbidden. □
    """)

    # Verify for test cases
    test_cases = [
        (7, 4),   # Λ_7(1^4)
        (7, 3),
        (7, 5),
        (15, 12), # Λ_15(1^12)
        (15, 11),
        (15, 10),
    ]

    all_passed = True

    for n, s in test_cases:
        passed = verify_corrected_lemma(n, s, verbose=True)
        if not passed:
            all_passed = False
        count_by_gap_structure(n, s)

    print("\n" + "=" * 70)
    print(f"ALL VERIFICATIONS PASSED: {all_passed}")
    print("=" * 70)

    # Special case analysis for s = n-3
    print("\n" + "=" * 70)
    print("ANALYSIS FOR s = n-3 (Critical Boundary)")
    print("=" * 70)

    print("""
    For s = n-3, a vertex is forbidden iff max_gap >= n-3.

    This means we need at least n-3 consecutive 1s, leaving at most 3 "space" for zeros.

    Key observations:
    1. All-ones (1^n): max_gap = n >= n-3, FORBIDDEN
    2. Weight-(n-1): one zero, gap = n-1 >= n-3, FORBIDDEN
    3. Weight-(n-2): two zeros, max_gap = max(g1, g2) where g1 + g2 = n-2
       FORBIDDEN iff max(g1, g2) >= n-3, i.e., at least one gap >= n-3

    For two zeros at distance d apart (smaller circular distance):
    - Gaps are: d-1 and n-d-1
    - max_gap = max(d-1, n-d-1)
    - FORBIDDEN iff max(d-1, n-d-1) >= n-3
    - For n=7, s=4: FORBIDDEN iff max(d-1, 6-d) >= 4
      - d=1: max(0, 5) = 5 >= 4 ✓ FORBIDDEN
      - d=2: max(1, 4) = 4 >= 4 ✓ FORBIDDEN
      - d=3: max(2, 3) = 3 < 4 ✗ ALLOWED

    This matches the n=15, s=12 case where only 61 vertices are forbidden.
    """)


if __name__ == "__main__":
    main()
