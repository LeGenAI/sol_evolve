#!/usr/bin/env python3
"""
Cascade Shift Construction for Perfect Partition in Λ_n(1^s)

Based on SAT solution analysis:
1. Each non-Hamming center is at distance 1 from unique Hamming codeword
2. Shifts propagate through the code in a cascade
3. Weight distribution is preserved

Algorithm:
1. Start with Hamming code, identify bad codewords
2. Build conflict graph: which codewords block repair of which vertices
3. Apply BFS/DFS cascade: shift blockers, propagate changes
4. Resolve new conflicts iteratively

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict, deque
from itertools import combinations
import random


def has_circular_run(v: int, n: int, s: int) -> bool:
    """Check if v has s consecutive circular 1s."""
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
    n = (1 << k) - 1
    syndrome = 0
    for i in range(n):
        if (v >> i) & 1:
            syndrome ^= (i + 1)
    return syndrome


def generate_hamming_code(k: int) -> Set[int]:
    n = (1 << k) - 1
    return {v for v in range(1 << n) if compute_syndrome(v, k) == 0}


def get_allowed_vertices(n: int, s: int) -> Set[int]:
    return {v for v in range(1 << n) if not has_circular_run(v, n, s)}


class CascadeShiftConstructor:
    """
    Cascade shift algorithm for constructing perfect partitions.
    """

    def __init__(self, k: int, s: int, verbose: bool = True):
        self.k = k
        self.n = (1 << k) - 1
        self.s = s
        self.verbose = verbose

        # Precompute
        self.hamming = generate_hamming_code(k)
        self.allowed = get_allowed_vertices(self.n, s)
        self.forbidden = set(range(1 << self.n)) - self.allowed

        # Current state: maps original Hamming codeword -> current position
        # If position == original, codeword hasn't been shifted
        self.codeword_position = {c: c for c in self.hamming}

        # Track which codewords are "active" (included in current solution)
        self.active = set(self.hamming)

        if self.verbose:
            print(f"CascadeShiftConstructor initialized")
            print(f"  n={self.n}, k={self.k}, s={self.s}")
            print(f"  |Hamming| = {len(self.hamming)}")
            print(f"  |Allowed| = {len(self.allowed)}")

    def get_current_centers(self) -> Set[int]:
        """Get current center positions."""
        return {self.codeword_position[c] for c in self.active}

    def get_coverage(self) -> Tuple[Set[int], Set[int]]:
        """Compute current coverage of allowed vertices."""
        centers = self.get_current_centers()
        covered = set()
        for c in centers:
            covered.update(get_ball(c, self.n) & self.allowed)
        uncovered = self.allowed - covered
        return covered, uncovered

    def check_packing(self) -> List[Tuple[int, int, int]]:
        """Check packing constraint, return violations."""
        centers = list(self.get_current_centers())
        violations = []
        for i, c1 in enumerate(centers):
            for c2 in centers[i+1:]:
                d = hamming_distance(c1, c2)
                if d < 3:
                    violations.append((c1, c2, d))
        return violations

    def find_valid_shifts(self, c: int) -> List[Tuple[int, int]]:
        """
        Find all valid single-bit shifts for codeword c.
        Returns list of (bit_position, new_position).
        """
        current_pos = self.codeword_position[c]
        other_centers = self.get_current_centers() - {current_pos}

        valid = []
        for i in range(self.n):
            new_pos = current_pos ^ (1 << i)

            # Must be in allowed region
            if new_pos not in self.allowed:
                continue

            # Must maintain d >= 3 from all other centers
            min_dist = min((hamming_distance(new_pos, o) for o in other_centers), default=self.n + 1)
            if min_dist < 3:
                continue

            valid.append((i, new_pos))

        return valid

    def apply_shift(self, c: int, bit_pos: int) -> bool:
        """Apply shift to codeword c at bit position."""
        old_pos = self.codeword_position[c]
        new_pos = old_pos ^ (1 << bit_pos)

        if new_pos not in self.allowed:
            return False

        self.codeword_position[c] = new_pos
        return True

    def remove_bad_codewords(self):
        """Remove codewords that are in forbidden region."""
        bad = {c for c in self.active if self.codeword_position[c] not in self.allowed}

        if self.verbose:
            print(f"\nRemoving {len(bad)} bad codewords...")

        for c in bad:
            self.active.discard(c)

        return bad

    def find_blocking_codewords(self, uncovered_vertex: int) -> Set[int]:
        """
        Find codewords that block repair of an uncovered vertex.

        A codeword c blocks v if:
        - Some potential repair center for v is at distance < 3 from c's current position
        """
        blockers = set()

        # Potential repair centers: vertices in ball of v that are in allowed
        potential_centers = get_ball(uncovered_vertex, self.n) & self.allowed

        centers = self.get_current_centers()

        for pc in potential_centers:
            for c in self.active:
                pos = self.codeword_position[c]
                if hamming_distance(pc, pos) < 3:
                    blockers.add(c)

        return blockers

    def cascade_repair(self, max_iterations: int = 10000) -> bool:
        """
        Main cascade algorithm.

        1. Remove bad codewords
        2. Find uncovered vertices
        3. For each uncovered, find blockers
        4. Try to shift blockers to make room
        5. Propagate shifts as needed
        """
        # Step 1: Remove bad codewords
        bad = self.remove_bad_codewords()

        # Step 2: Initial assessment
        covered, uncovered = self.get_coverage()

        if self.verbose:
            print(f"After removing bad: {len(uncovered)} uncovered vertices")

        if not uncovered:
            return True

        # Step 3: Build blocking relationships
        # uncovered_vertex -> set of blocking codewords
        blocking = {}
        for v in uncovered:
            blocking[v] = self.find_blocking_codewords(v)

        if self.verbose:
            total_blockers = sum(len(b) for b in blocking.values())
            print(f"Total blocking relationships: {total_blockers}")

        # Step 4: Cascade shift
        iteration = 0
        shift_queue = deque()  # (codeword, target_uncovered)

        # Start with blockers of uncovered vertices
        for v, blockers in blocking.items():
            for b in blockers:
                shift_queue.append((b, v))

        shifts_applied = []
        attempted = set()

        while shift_queue and iteration < max_iterations:
            iteration += 1

            c, target_v = shift_queue.popleft()

            if (c, target_v) in attempted:
                continue
            attempted.add((c, target_v))

            # Find valid shifts for c
            valid_shifts = self.find_valid_shifts(c)

            if not valid_shifts:
                # Can't shift this codeword directly
                # Try to shift its blockers first (cascade)
                current_pos = self.codeword_position[c]
                for i in range(self.n):
                    potential_new = current_pos ^ (1 << i)
                    if potential_new not in self.allowed:
                        continue
                    # Who blocks this shift?
                    other_centers = self.get_current_centers() - {current_pos}
                    for oc in other_centers:
                        if hamming_distance(potential_new, oc) < 3:
                            # oc blocks this shift, add to queue
                            # Find which original codeword oc corresponds to
                            for orig, pos in self.codeword_position.items():
                                if pos == oc and orig in self.active:
                                    shift_queue.append((orig, target_v))
                                    break
                continue

            # Pick best shift (greedy: one that helps most)
            best_shift = None
            best_improvement = -1

            for bit_pos, new_pos in valid_shifts:
                # Temporarily apply shift
                old_pos = self.codeword_position[c]
                self.codeword_position[c] = new_pos

                # Check improvement
                _, new_uncovered = self.get_coverage()
                improvement = len(uncovered) - len(new_uncovered)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_shift = (bit_pos, new_pos)

                # Revert
                self.codeword_position[c] = old_pos

            if best_shift is not None:
                bit_pos, new_pos = best_shift
                old_pos = self.codeword_position[c]
                self.codeword_position[c] = new_pos
                shifts_applied.append((c, old_pos, new_pos))

                if self.verbose and len(shifts_applied) <= 20:
                    print(f"  Shift {len(shifts_applied)}: {format(old_pos, f'0{self.n}b')} -> {format(new_pos, f'0{self.n}b')}")

                # Recompute uncovered
                covered, uncovered = self.get_coverage()

                if not uncovered:
                    if self.verbose:
                        print(f"\n*** All vertices covered after {len(shifts_applied)} shifts! ***")
                    return True

                # Check for new packing violations
                violations = self.check_packing()
                if violations:
                    # Add violating codewords to queue
                    for c1, c2, d in violations:
                        for orig, pos in self.codeword_position.items():
                            if pos == c1 or pos == c2:
                                if orig in self.active:
                                    shift_queue.append((orig, target_v))

        if self.verbose:
            print(f"\nCascade finished after {iteration} iterations")
            print(f"Shifts applied: {len(shifts_applied)}")
            covered, uncovered = self.get_coverage()
            print(f"Final uncovered: {len(uncovered)}")

        return len(uncovered) == 0

    def greedy_coverage_cascade(self, max_iterations: int = 50000) -> bool:
        """
        Alternative: Greedy approach focusing on coverage.

        For each uncovered vertex, try all possible ways to cover it:
        1. Add a new center (shift some codeword to cover it)
        2. Choose the option that minimizes new conflicts
        """
        # Remove bad
        self.remove_bad_codewords()

        covered, uncovered = self.get_coverage()

        if self.verbose:
            print(f"\nGreedy coverage cascade starting with {len(uncovered)} uncovered")

        iteration = 0
        while uncovered and iteration < max_iterations:
            iteration += 1

            # Pick an uncovered vertex
            v = min(uncovered)  # Deterministic choice

            # Find all ways to cover v
            options = []

            # Option type 1: Shift an existing center to cover v
            for c in self.active:
                current_pos = self.codeword_position[c]
                ball_v = get_ball(v, self.n)

                for i in range(self.n):
                    new_pos = current_pos ^ (1 << i)

                    if new_pos not in ball_v:
                        continue  # Doesn't cover v

                    if new_pos not in self.allowed:
                        continue

                    # Check packing
                    other_centers = self.get_current_centers() - {current_pos}
                    min_dist = min((hamming_distance(new_pos, o) for o in other_centers), default=self.n + 1)
                    if min_dist < 3:
                        continue

                    # Valid option
                    options.append((c, i, new_pos))

            if not options:
                if self.verbose and iteration <= 10:
                    print(f"  Iteration {iteration}: No options for {format(v, f'0{self.n}b')}")

                # Try cascade: shift blockers of v
                blockers = self.find_blocking_codewords(v)
                found = False

                for blocker in blockers:
                    # Try to shift blocker
                    valid_shifts = self.find_valid_shifts(blocker)
                    if valid_shifts:
                        # Apply first valid shift
                        bit_pos, new_pos = valid_shifts[0]
                        self.codeword_position[blocker] = new_pos
                        found = True
                        break

                if not found:
                    # Deep cascade needed - skip for now
                    uncovered.discard(v)  # Give up on this vertex temporarily
                    continue

            else:
                # Pick best option (covers most additional uncovered)
                best_option = None
                best_coverage = 0

                for c, bit_pos, new_pos in options[:100]:  # Limit search
                    # Temporarily apply
                    old_pos = self.codeword_position[c]
                    self.codeword_position[c] = new_pos

                    new_covered, new_uncovered = self.get_coverage()
                    coverage_gain = len(covered) - len(new_covered) + len(uncovered) - len(new_uncovered)

                    if len(new_uncovered) < len(uncovered):
                        if best_option is None or len(new_uncovered) < best_coverage:
                            best_option = (c, bit_pos, new_pos)
                            best_coverage = len(new_uncovered)

                    # Revert
                    self.codeword_position[c] = old_pos

                if best_option:
                    c, bit_pos, new_pos = best_option
                    self.codeword_position[c] = new_pos

                    if self.verbose and iteration <= 20:
                        print(f"  Iteration {iteration}: Applied shift, uncovered: {best_coverage}")

            # Update
            covered, uncovered = self.get_coverage()

        # Verify
        violations = self.check_packing()

        if self.verbose:
            print(f"\nFinished after {iteration} iterations")
            print(f"Uncovered: {len(uncovered)}")
            print(f"Packing violations: {len(violations)}")

        return len(uncovered) == 0 and len(violations) == 0

    def verify_solution(self) -> Dict:
        """Verify current solution is valid perfect partition."""
        centers = self.get_current_centers()

        # All centers in allowed?
        invalid = centers - self.allowed

        # Packing
        violations = self.check_packing()

        # Covering
        covered, uncovered = self.get_coverage()

        # Overlap with Hamming
        overlap = centers & self.hamming

        result = {
            'valid': len(invalid) == 0 and len(violations) == 0 and len(uncovered) == 0,
            'centers': len(centers),
            'invalid_centers': len(invalid),
            'packing_violations': len(violations),
            'uncovered': len(uncovered),
            'hamming_overlap': len(overlap),
            'hamming_overlap_pct': 100 * len(overlap) / len(centers) if centers else 0
        }

        return result


def test_cascade_n7():
    """Test cascade on n=7, s=4."""
    print("="*70)
    print("Testing Cascade on n=7, s=4")
    print("="*70)

    constructor = CascadeShiftConstructor(k=3, s=4)

    # Try cascade repair
    print("\n--- Cascade Repair ---")
    success = constructor.cascade_repair(max_iterations=1000)

    if not success:
        print("\nCascade repair failed, trying greedy...")
        # Reset and try greedy
        constructor = CascadeShiftConstructor(k=3, s=4, verbose=True)
        success = constructor.greedy_coverage_cascade(max_iterations=5000)

    result = constructor.verify_solution()
    print(f"\nVerification: {result}")

    return constructor if result['valid'] else None


def test_cascade_n15_s12():
    """Test cascade on n=15, s=12."""
    print("\n" + "="*70)
    print("Testing Cascade on n=15, s=12")
    print("="*70)

    constructor = CascadeShiftConstructor(k=4, s=12)

    print("\n--- Greedy Coverage Cascade ---")
    success = constructor.greedy_coverage_cascade(max_iterations=10000)

    result = constructor.verify_solution()
    print(f"\nVerification: {result}")

    return constructor if result['valid'] else None


def test_cascade_n15_s11():
    """Test cascade on n=15, s=11."""
    print("\n" + "="*70)
    print("Testing Cascade on n=15, s=11")
    print("="*70)

    constructor = CascadeShiftConstructor(k=4, s=11)

    print("\n--- Greedy Coverage Cascade ---")
    success = constructor.greedy_coverage_cascade(max_iterations=20000)

    result = constructor.verify_solution()
    print(f"\nVerification: {result}")

    return constructor if result['valid'] else None


def main():
    """Run all tests."""
    print("Cascade Shift Construction Tests")
    print("="*70)

    # n=7
    result7 = test_cascade_n7()

    # n=15, s=12
    result15_12 = test_cascade_n15_s12()

    # n=15, s=11
    result15_11 = test_cascade_n15_s11()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"n=7, s=4: {'SUCCESS' if result7 else 'FAILED'}")
    print(f"n=15, s=12: {'SUCCESS' if result15_12 else 'FAILED'}")
    print(f"n=15, s=11: {'SUCCESS' if result15_11 else 'FAILED'}")


if __name__ == "__main__":
    main()
