#!/usr/bin/env python3
"""
Forbidden Region Lemma - Formal Proof

Lemma: A vertex v ∈ {0,1}^n is forbidden in Λ_n(1^s) iff
       all zeros of v lie in a contiguous circular arc of length ≤ (n-s).

This characterizes exactly which vertices are forbidden (contain circular 1^s).

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import List, Tuple, Set

# ============================================================================
# Definitions
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


def is_contiguous_arc(positions: List[int], n: int) -> Tuple[bool, int]:
    """
    Check if positions form a contiguous circular arc.
    Returns (is_contiguous, arc_length).

    Key insight: positions form a contiguous arc iff when we compute the
    circular gaps between consecutive positions, all gaps are 1 except
    possibly one "wraparound" gap.
    """
    if not positions:
        return True, 0

    if len(positions) == 1:
        return True, 1

    k = len(positions)
    positions = sorted(positions)

    # Compute circular gaps
    gaps = []
    for i in range(k):
        next_idx = (i + 1) % k
        gap = (positions[next_idx] - positions[i]) % n
        if gap == 0:
            gap = n  # Same position after wraparound means full circle
        gaps.append(gap)

    # For a contiguous arc of k positions:
    # - (k-1) gaps should be exactly 1 (consecutive)
    # - 1 gap should be (n - k + 1) (the "outside" gap)

    ones = sum(1 for g in gaps if g == 1)
    expected_large = n - k + 1

    if ones == k - 1:
        # Check the remaining gap
        other_gaps = [g for g in gaps if g != 1]
        if len(other_gaps) == 1 and other_gaps[0] == expected_large:
            return True, k

    # Edge case: all k positions (full circle can't happen for k < n)
    # Edge case: k = n would mean all positions, but that's not an "arc"

    return False, k


# ============================================================================
# Lemma Statement and Proof
# ============================================================================

def verify_forbidden_region_lemma(n: int, s: int, verbose: bool = False) -> bool:
    """
    Verify the Forbidden Region Lemma for given n, s.

    Lemma: v is forbidden (contains circular 1^s) iff
           all zeros of v lie in a contiguous arc of length ≤ (n-s).

    Proof direction 1 (⟹):
    If v contains circular 1^s, then there exist s consecutive positions that are all 1s.
    Therefore, all zeros must be in the remaining (n-s) consecutive positions,
    forming a contiguous arc of length ≤ (n-s).

    Proof direction 2 (⟸):
    If all zeros lie in a contiguous arc of length k ≤ (n-s),
    then the remaining (n-k) ≥ s positions are all 1s and consecutive,
    thus v contains circular 1^s.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"VERIFYING FORBIDDEN REGION LEMMA FOR n={n}, s={s}")
        print(f"{'='*60}")
        print(f"Threshold: zeros must be in arc of length ≤ {n-s}")

    verified = True
    counterexamples = []

    # Check all 2^n vertices
    for v in range(1 << n):
        is_forbidden = has_circular_ones(v, n, s)
        zero_pos = get_zero_positions(v, n)
        is_arc, arc_len = is_contiguous_arc(zero_pos, n)

        # Lemma predicts: forbidden iff (is_arc and arc_len ≤ n-s)
        lemma_predicts_forbidden = is_arc and arc_len <= n - s

        if is_forbidden != lemma_predicts_forbidden:
            verified = False
            counterexamples.append({
                'v': v,
                'binary': bin(v),
                'is_forbidden': is_forbidden,
                'zero_pos': zero_pos,
                'is_arc': is_arc,
                'arc_len': arc_len,
                'lemma_predicts': lemma_predicts_forbidden
            })

    if verbose:
        if verified:
            print(f"✓ Lemma VERIFIED for all 2^{n} = {1 << n} vertices")
        else:
            print(f"✗ Lemma FAILED with {len(counterexamples)} counterexamples")
            for ce in counterexamples[:5]:
                print(f"  v={ce['binary']}: is_forbidden={ce['is_forbidden']}, "
                      f"zero_pos={ce['zero_pos']}, is_arc={ce['is_arc']}, "
                      f"arc_len={ce['arc_len']}, lemma_predicts={ce['lemma_predicts']}")

    return verified


def count_forbidden_vertices(n: int, s: int) -> Tuple[int, int]:
    """
    Count forbidden vertices using the lemma.

    By the Forbidden Region Lemma:
    v is forbidden iff all zeros lie in a contiguous arc of length ≤ (n-s).

    For k zeros (k ≤ n-s):
    - Choose starting position: n choices (circular)
    - But k consecutive positions have only n/gcd(n,k) distinct orbits... actually just n starts
    - Total forbidden with exactly k zeros: n * C(k-1, ...) ? No, simpler:

    Actually: For each starting position i, vertices with zeros at positions i, i+1, ..., i+k-1
              form one "forbidden class" for each subset of {i, i+1, ..., i+(n-s)-1}.

    Let's count directly for verification.
    """
    forbidden_count = 0
    allowed_count = 0

    for v in range(1 << n):
        if has_circular_ones(v, n, s):
            forbidden_count += 1
        else:
            allowed_count += 1

    return forbidden_count, allowed_count


def derive_forbidden_formula(n: int, s: int) -> int:
    """
    Derive closed-form formula for |forbidden vertices|.

    By the lemma, forbidden vertices are those with zeros in contiguous arc of length ≤ (n-s).

    For k zeros in contiguous arc (k = 0, 1, ..., n-s):
    - If k = 0: 1 vertex (all ones)
    - If k > 0: n choices for starting position × 1 way to place k consecutive zeros
      BUT this overcounts if k divides n (rotational symmetry)
      Actually no, each starting position gives distinct vertex

    Wait, let me think again:
    - For k zeros in positions {i, i+1, ..., i+k-1} (mod n), there are n choices for i
    - But for k = n, all zeros, there's only 1 vertex
    - For k < n, each of the n starting positions gives a distinct vertex

    So: |forbidden| = Σ_{k=0}^{n-s} (n if k > 0 else 1)
                    = 1 + n * (n - s)
                    = 1 + n(n - s)

    Hmm, that doesn't match. Let me verify computationally.
    """

    # Direct count
    _, _ = count_forbidden_vertices(n, s)

    # Formula attempt: vertices with zeros in arc of length ≤ (n-s)
    # For each arc length k from 0 to n-s:
    # - k=0: just all-ones, 1 vertex
    # - k≥1: n starting positions, each gives 2^k - 1 vertices? No wait...

    # Actually: for arc of length k, we can have any subset of those k positions be zeros
    # But at least one must be zero to have "zeros in the arc"
    # And positions outside the arc must all be ones

    # Wait no, the lemma says "all zeros lie in arc of length ≤ (n-s)"
    # So: for each arc of length m (m = 0 to n-s), count vertices where ALL zeros are within that arc

    # Fix an arc of length m starting at position i
    # Vertices with all zeros in that arc: 2^m choices for which of the m positions are 0
    # (including all 1s within the arc)

    # Total: Σ_{m=0}^{n-s} Σ_{i=0}^{n-1} 2^m = n * Σ_{m=0}^{n-s} 2^m = n * (2^{n-s+1} - 1)

    # But this overcounts! Arc of length 0 at any position gives the same vertex (all 1s)
    # Arc of length m gives same vertex if zeros are same regardless of arc choice

    # Better approach: count by number of zeros k (k = 0 to n-s)
    # For k zeros in contiguous arc:
    # - k=0: 1 vertex
    # - k≥1: n choices for arc start, but we're fixing exactly k zeros

    # Hmm, this is getting complicated. Let's just verify the counts match.

    return 0  # Placeholder, actual formula TBD


# ============================================================================
# Corollaries
# ============================================================================

def verify_all_ones_neighborhood(n: int, s: int) -> bool:
    """
    Corollary: N[1^n] ⊆ Forbidden vertices in Λ_n(1^s) when s ≤ n-1.

    Proof:
    - 1^n itself: all positions are 1, contains 1^s for any s ≤ n. FORBIDDEN.
    - Neighbors of 1^n: weight-(n-1) vertices, have exactly 1 zero.
      By the lemma, 1 zero in arc of length 1 ≤ n-s (when s ≤ n-1). FORBIDDEN.
    """

    print(f"\n{'='*60}")
    print(f"VERIFYING ALL-ONES NEIGHBORHOOD COROLLARY")
    print(f"{'='*60}")

    all_ones = (1 << n) - 1

    # Check all-ones
    all_ones_forbidden = has_circular_ones(all_ones, n, s)
    print(f"All-ones (1^{n}): forbidden = {all_ones_forbidden}")

    # Check neighbors (weight n-1)
    neighbors_forbidden = True
    for i in range(n):
        neighbor = all_ones ^ (1 << i)  # Flip bit i
        if not has_circular_ones(neighbor, n, s):
            neighbors_forbidden = False
            print(f"  Neighbor (flip bit {i}): NOT forbidden!")

    print(f"All neighbors forbidden: {neighbors_forbidden}")

    # Verify with lemma
    # Neighbors have 1 zero, arc length 1 ≤ n-s when s ≤ n-1
    lemma_predicts = (1 <= n - s)
    print(f"Lemma predicts neighbors forbidden when s ≤ {n-1}: {lemma_predicts}")

    return all_ones_forbidden and (neighbors_forbidden == lemma_predicts)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("FORBIDDEN REGION LEMMA - FORMAL VERIFICATION")
    print("=" * 70)

    # Verify for small cases
    test_cases = [
        (7, 4),   # Λ_7(1^4)
        (7, 3),   # Λ_7(1^3) - UNSAT case
        (7, 5),   # Λ_7(1^5)
        (15, 12), # Λ_15(1^12)
        (15, 11), # Λ_15(1^11)
        (15, 10), # Λ_15(1^10)
    ]

    all_passed = True

    for n, s in test_cases:
        passed = verify_forbidden_region_lemma(n, s, verbose=True)
        if not passed:
            all_passed = False

        # Count forbidden/allowed
        forbidden, allowed = count_forbidden_vertices(n, s)
        print(f"Forbidden: {forbidden}, Allowed: {allowed}, Total: {1 << n}")
        print(f"Allowed ratio: {allowed / (1 << n):.6f}")

    print("\n" + "=" * 70)
    print("FORMAL LEMMA STATEMENT")
    print("=" * 70)

    print("""
    LEMMA (Forbidden Region Characterization):

    Let Λ_n(1^s) be the generalized Lucas cube of n-bit strings
    avoiding circular runs of s consecutive 1s.

    A vertex v ∈ {0,1}^n is FORBIDDEN (not in Λ_n(1^s)) if and only if
    all zero-bits of v lie within a contiguous circular arc of length at most (n-s).

    PROOF:

    (⟹) Suppose v is forbidden, i.e., v contains a circular run of s consecutive 1s.
    Let positions p, p+1, ..., p+s-1 (mod n) all be 1.
    Then all zeros of v must lie in the remaining (n-s) consecutive positions
    p+s, p+s+1, ..., p-1 (mod n), which form a contiguous arc of length (n-s).

    (⟸) Suppose all zeros of v lie in a contiguous circular arc of length k ≤ (n-s).
    Then the remaining (n-k) ≥ s positions are all 1s.
    Since these (n-k) positions are consecutive (complement of a contiguous arc),
    v contains at least s consecutive 1s.
    Therefore v is forbidden. □

    COROLLARY: For s = n-3 (the critical case):
    v is forbidden iff all zeros lie in an arc of length ≤ 3.
    This means:
    - All-ones (0 zeros): FORBIDDEN
    - Weight-(n-1) with 1 zero: arc length 1 ≤ 3, FORBIDDEN
    - Weight-(n-2) with 2 zeros: FORBIDDEN iff zeros are adjacent (arc ≤ 3)
    - Weight-(n-3) with 3 zeros: FORBIDDEN iff zeros are in 3 consecutive positions
    """)

    # Verify corollary for n=7, s=4
    print("\n" + "=" * 70)
    print("VERIFYING COROLLARIES")
    print("=" * 70)

    verify_all_ones_neighborhood(7, 4)
    verify_all_ones_neighborhood(15, 12)

    print("\n" + "=" * 70)
    print(f"ALL VERIFICATIONS PASSED: {all_passed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
