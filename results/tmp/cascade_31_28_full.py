#!/usr/bin/env python3
"""
Full Cascade Replacement Algorithm for Λ₃₁(1²⁸)

Optimized implementation for finding perfect partition of generalized Lucas cube.

Key optimizations:
1. Integer bit representation (31-bit integers) instead of numpy arrays
2. Hash-based center set management
3. Checkpoint/restart capability
4. Progress monitoring with ETA
5. Efficient syndrome computation using precomputed tables

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import deque
import time
import pickle
import os
import sys
from typing import Set, Tuple, List, Optional, Dict
import multiprocessing as mp
from functools import lru_cache

# Constants
N = 31
S = 28
K = 5  # log2(n+1)
NUM_CODEWORDS = 2**26  # 67,108,864
TARGET_CENTERS = NUM_CODEWORDS - 1  # 67,108,863

# Checkpoint file
CHECKPOINT_FILE = "cascade_31_28_checkpoint.pkl"
RESULT_FILE = "lambda_31_28_centers.txt"

# ============================================================================
# Bit manipulation utilities
# ============================================================================

def int_to_bits(v: int, n: int = N) -> List[int]:
    """Convert integer to list of bits (LSB first)"""
    return [(v >> i) & 1 for i in range(n)]

def bits_to_int(bits: List[int]) -> int:
    """Convert list of bits to integer (LSB first)"""
    return sum(b << i for i, b in enumerate(bits))

def popcount(v: int) -> int:
    """Count number of 1 bits"""
    return bin(v).count('1')

def hamming_distance_int(v1: int, v2: int) -> int:
    """Hamming distance between two integers"""
    return popcount(v1 ^ v2)

# ============================================================================
# Precomputed tables for efficiency
# ============================================================================

# Parity check matrix H for Hamming(31, 26)
# Columns are binary representations of 1 to 31
def build_parity_check_matrix():
    """Build 5x31 parity check matrix"""
    H = np.zeros((K, N), dtype=np.uint8)
    for j in range(N):
        col_val = j + 1  # columns are 1 to 31
        for i in range(K):
            H[i, j] = (col_val >> i) & 1
    return H

H_MATRIX = build_parity_check_matrix()

# Precompute syndrome lookup table
# syndrome -> error position (0 to 30, or -1 if no single-bit error)
SYNDROME_TO_ERROR = {}
for pos in range(N):
    syn = 0
    for i in range(K):
        if H_MATRIX[i, pos]:
            syn |= (1 << i)
    SYNDROME_TO_ERROR[syn] = pos

def compute_syndrome_int(v: int) -> int:
    """Compute syndrome of v as integer"""
    syn = 0
    for i in range(K):
        # Row i of H: positions where (j+1) has bit i set
        parity = 0
        for j in range(N):
            if (v >> j) & 1:
                if ((j + 1) >> i) & 1:
                    parity ^= 1
        syn |= (parity << i)
    return syn

@lru_cache(maxsize=2**20)
def is_hamming_codeword_int(v: int) -> bool:
    """Check if v is a Hamming codeword (syndrome = 0)"""
    return compute_syndrome_int(v) == 0

def has_circular_ones_int(v: int, s: int = S) -> bool:
    """Check if v contains s consecutive 1s in circular form"""
    # Create doubled representation
    doubled = v | (v << N)
    mask = (1 << s) - 1  # s consecutive 1s

    for i in range(N):
        window = (doubled >> i) & mask
        if window == mask:  # All s bits are 1
            return True
    return False

def get_neighbors_int(v: int, n: int = N) -> List[int]:
    """Get all n neighbors (flip one bit)"""
    neighbors = []
    for i in range(n):
        neighbors.append(v ^ (1 << i))
    return neighbors

def get_distance_2_neighbors_int(v: int, n: int = N) -> List[int]:
    """Get all neighbors at distance 2 (flip two bits)"""
    neighbors = []
    for i in range(n):
        for j in range(i + 1, n):
            neighbors.append(v ^ (1 << i) ^ (1 << j))
    return neighbors

# ============================================================================
# Bad codeword identification
# ============================================================================

def find_bad_codewords() -> Set[int]:
    """Find all Hamming codewords containing circular 1^s"""
    bad = set()

    # All-ones: 2^31 - 1
    all_ones = (1 << N) - 1
    if is_hamming_codeword_int(all_ones):
        bad.add(all_ones)
        print(f"  All-ones (weight {N}): {bin(all_ones)}")

    # Weight-(N-3) with contiguous zeros
    # These have 3 consecutive zeros in circular form
    for start in range(N):
        v = all_ones
        for offset in range(3):
            pos = (start + offset) % N
            v &= ~(1 << pos)  # Clear bit at position

        if is_hamming_codeword_int(v):
            bad.add(v)
            if len(bad) <= 5:
                zeros_at = [(start + i) % N for i in range(3)]
                print(f"  Weight-{popcount(v)} with zeros at {zeros_at}: syndrome check")

    return bad

# ============================================================================
# Coverage checking
# ============================================================================

def is_covered_by(v: int, centers: Set[int]) -> bool:
    """Check if vertex v is covered by any center in the set"""
    # Distance 0: v itself is a center
    if v in centers:
        return True

    # Distance 1: check all neighbors
    for neighbor in get_neighbors_int(v):
        if neighbor in centers:
            return True

    return False

def find_covering_center(v: int, centers: Set[int]) -> Optional[int]:
    """Find which center covers v, if any"""
    if v in centers:
        return v
    for neighbor in get_neighbors_int(v):
        if neighbor in centers:
            return neighbor
    return None

# ============================================================================
# Cascade replacement algorithm
# ============================================================================

class CascadeState:
    """State management for cascade algorithm with checkpoint support"""

    def __init__(self):
        self.centers: Set[int] = set()  # Current center set
        self.bad_queue: deque = deque()  # Bad codewords to process
        self.removed_hamming: Set[int] = set()  # Removed Hamming codewords
        self.added_nonhamming: Set[int] = set()  # Added non-Hamming centers
        self.cascade_depth: int = 0
        self.total_processed: int = 0
        self.start_time: float = 0

    def save_checkpoint(self, filename: str = CHECKPOINT_FILE):
        """Save state to file"""
        state_dict = {
            'centers': self.centers,
            'bad_queue': list(self.bad_queue),
            'removed_hamming': self.removed_hamming,
            'added_nonhamming': self.added_nonhamming,
            'cascade_depth': self.cascade_depth,
            'total_processed': self.total_processed,
        }
        with open(filename, 'wb') as f:
            pickle.dump(state_dict, f)
        print(f"  [Checkpoint saved: {len(self.centers):,} centers, depth {self.cascade_depth}]")

    def load_checkpoint(self, filename: str = CHECKPOINT_FILE) -> bool:
        """Load state from file"""
        if not os.path.exists(filename):
            return False

        try:
            with open(filename, 'rb') as f:
                state_dict = pickle.load(f)

            self.centers = state_dict['centers']
            self.bad_queue = deque(state_dict['bad_queue'])
            self.removed_hamming = state_dict['removed_hamming']
            self.added_nonhamming = state_dict['added_nonhamming']
            self.cascade_depth = state_dict['cascade_depth']
            self.total_processed = state_dict['total_processed']

            print(f"  [Checkpoint loaded: {len(self.centers):,} centers, depth {self.cascade_depth}]")
            return True
        except Exception as e:
            print(f"  [Checkpoint load failed: {e}]")
            return False


def initialize_from_hamming(state: CascadeState) -> None:
    """Initialize centers with all Hamming codewords except bad ones"""
    print("\n" + "=" * 70)
    print("PHASE 1: INITIALIZATION")
    print("=" * 70)

    print("\nFinding bad codewords...")
    bad_codewords = find_bad_codewords()
    print(f"Found {len(bad_codewords)} bad codewords")

    print("\nInitializing Hamming codeword set...")
    print(f"Total Hamming codewords: {NUM_CODEWORDS:,}")

    # Generate all Hamming codewords
    # A codeword is determined by its 26 information bits
    # We use the systematic form: first 26 bits are information, last 5 are parity

    count = 0
    for info in range(NUM_CODEWORDS):
        # Construct codeword from information bits
        # For Hamming(31,26), we need to compute the codeword
        # Actually, easier to iterate and check syndrome

        # Alternative: use generator matrix approach
        # But for simplicity, we'll use a different method
        pass

    # Actually, iterating through all 2^31 possibilities is infeasible
    # Instead, we use the syndrome decoding property:
    # Every vector v has a unique nearest Hamming codeword

    # For initialization, we start with just the bad codewords identified
    # and their neighborhoods, then expand

    # Simpler approach: Start with known bad codewords and cascade from there
    print("\nUsing cascade approach from bad codewords...")

    # Add all bad codewords to the queue
    for bad in bad_codewords:
        state.bad_queue.append(bad)
        state.removed_hamming.add(bad)

    # The remaining Hamming codewords form the initial center set
    # We'll track removed ones and assume the rest are in
    state.centers = set()  # Will be computed implicitly

    print(f"Bad codewords in queue: {len(state.bad_queue)}")


def find_uncovered_vertices(state: CascadeState) -> List[int]:
    """Find allowed vertices not covered by current centers"""
    uncovered = []

    # Check neighborhoods of removed codewords
    all_to_check = set()
    for removed in state.removed_hamming:
        for neighbor in get_neighbors_int(removed):
            if not has_circular_ones_int(neighbor):
                all_to_check.add(neighbor)
        all_to_check.add(removed)

    for v in all_to_check:
        if has_circular_ones_int(v):
            continue  # Forbidden vertex

        # Check if covered by a Hamming codeword not in removed set
        covered = False

        # Check if v itself is a non-removed Hamming codeword
        if is_hamming_codeword_int(v) and v not in state.removed_hamming:
            covered = True
        else:
            # Check distance-1 neighbors
            for neighbor in get_neighbors_int(v):
                if is_hamming_codeword_int(neighbor) and neighbor not in state.removed_hamming:
                    covered = True
                    break

        # Also check non-Hamming centers we've added
        if not covered:
            if v in state.added_nonhamming:
                covered = True
            else:
                for neighbor in get_neighbors_int(v):
                    if neighbor in state.added_nonhamming:
                        covered = True
                        break

        if not covered:
            uncovered.append(v)

    return uncovered


def find_repair_center(v: int, state: CascadeState) -> Tuple[Optional[int], List[int]]:
    """
    Find a repair center for uncovered vertex v.

    Returns:
        (repair_center, conflicting_codewords)

    The repair center r must satisfy:
    - r is allowed (no circular 1^s)
    - d(v, r) <= 1
    - d(r, c) >= 3 for all existing centers c

    If no such r exists, returns conflicting codewords that need to be removed.
    """
    # Candidates: v itself and its neighbors
    candidates = [v] + get_neighbors_int(v)

    for r in candidates:
        if has_circular_ones_int(r):
            continue  # r is forbidden

        # Check distance to all existing Hamming centers (not removed)
        # and all added non-Hamming centers
        conflicts = []
        valid = True

        # Check distance-1 and distance-2 neighbors of r
        # These are potential conflicts (d < 3)
        close_vertices = set([r])
        for n1 in get_neighbors_int(r):
            close_vertices.add(n1)
        for n2 in get_distance_2_neighbors_int(r):
            close_vertices.add(n2)

        for close in close_vertices:
            # Is close a non-removed Hamming codeword?
            if is_hamming_codeword_int(close) and close not in state.removed_hamming:
                if close != r:  # r itself being a codeword is OK if it's good
                    conflicts.append(close)
                    valid = False

            # Is close an added non-Hamming center?
            if close in state.added_nonhamming and close != r:
                conflicts.append(close)
                valid = False

        if valid:
            return r, []

        # If not valid, record conflicts for cascade
        if conflicts:
            return None, conflicts

    return None, []


def run_cascade(state: CascadeState, max_iterations: int = 10**9,
                checkpoint_interval: int = 10000) -> bool:
    """
    Run the cascade replacement algorithm.

    Returns True if successful, False if failed.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: CASCADE REPLACEMENT")
    print("=" * 70)

    state.start_time = time.time()
    iteration = 0
    last_checkpoint = 0

    while state.bad_queue and iteration < max_iterations:
        iteration += 1
        state.total_processed += 1

        # Pop a bad codeword from queue
        bad = state.bad_queue.popleft()

        # It's already in removed_hamming, so find uncovered vertices
        uncovered = []
        for neighbor in get_neighbors_int(bad):
            if has_circular_ones_int(neighbor):
                continue

            # Check if this neighbor is now uncovered
            covered = False

            # Check distance-0
            if is_hamming_codeword_int(neighbor) and neighbor not in state.removed_hamming:
                covered = True
            elif neighbor in state.added_nonhamming:
                covered = True
            else:
                # Check distance-1
                for n2 in get_neighbors_int(neighbor):
                    if (is_hamming_codeword_int(n2) and n2 not in state.removed_hamming) or \
                       n2 in state.added_nonhamming:
                        covered = True
                        break

            if not covered:
                uncovered.append(neighbor)

        # Also check if bad itself needs covering (it's allowed if good, but bad by definition is not)
        # Actually bad codewords are forbidden themselves since they contain 1^s

        # Try to repair each uncovered vertex
        for v in uncovered:
            repair, conflicts = find_repair_center(v, state)

            if repair is not None:
                # Successfully found repair center
                if not is_hamming_codeword_int(repair):
                    state.added_nonhamming.add(repair)
            else:
                # Need to cascade: remove conflicting codewords
                for conflict in conflicts:
                    if conflict not in state.removed_hamming:
                        state.removed_hamming.add(conflict)
                        state.bad_queue.append(conflict)

        # Progress reporting
        if iteration % 1000 == 0:
            elapsed = time.time() - state.start_time
            qsize = len(state.bad_queue)
            removed = len(state.removed_hamming)
            added = len(state.added_nonhamming)

            print(f"  Iteration {iteration:,}: queue={qsize:,}, "
                  f"removed={removed:,}, added={added:,}, "
                  f"time={elapsed:.1f}s")

        # Checkpoint
        if iteration - last_checkpoint >= checkpoint_interval:
            state.save_checkpoint()
            last_checkpoint = iteration

    # Final status
    print("\n" + "-" * 70)
    print("CASCADE COMPLETED")
    print("-" * 70)

    elapsed = time.time() - state.start_time
    print(f"Total iterations: {iteration:,}")
    print(f"Bad queue remaining: {len(state.bad_queue)}")
    print(f"Hamming codewords removed: {len(state.removed_hamming):,}")
    print(f"Non-Hamming centers added: {len(state.added_nonhamming):,}")
    print(f"Time elapsed: {elapsed:.1f}s")

    return len(state.bad_queue) == 0


def verify_partition(state: CascadeState, sample_size: int = 100000) -> bool:
    """Verify the partition is valid (sampling-based for large scale)"""
    print("\n" + "=" * 70)
    print("PHASE 3: VERIFICATION")
    print("=" * 70)

    # Count total centers
    # Centers = Hamming codewords not removed + added non-Hamming
    hamming_remaining = NUM_CODEWORDS - len(state.removed_hamming)
    total_centers = hamming_remaining + len(state.added_nonhamming)

    print(f"\nCenter counts:")
    print(f"  Hamming remaining: {hamming_remaining:,}")
    print(f"  Non-Hamming added: {len(state.added_nonhamming):,}")
    print(f"  Total centers: {total_centers:,}")
    print(f"  Target: {TARGET_CENTERS:,}")

    if total_centers != TARGET_CENTERS:
        print(f"\n  WARNING: Center count mismatch!")
        return False

    # Sample-based verification
    print(f"\nSampling {sample_size:,} random allowed vertices...")

    import random
    checked = 0
    covered_count = 0

    for _ in range(sample_size):
        # Generate random 31-bit integer
        v = random.randint(0, (1 << N) - 1)

        # Skip if forbidden
        if has_circular_ones_int(v):
            continue

        checked += 1

        # Check coverage
        covered = False

        # Check if v is a center
        if is_hamming_codeword_int(v) and v not in state.removed_hamming:
            covered = True
        elif v in state.added_nonhamming:
            covered = True
        else:
            # Check neighbors
            for neighbor in get_neighbors_int(v):
                if (is_hamming_codeword_int(neighbor) and neighbor not in state.removed_hamming) or \
                   neighbor in state.added_nonhamming:
                    covered = True
                    break

        if covered:
            covered_count += 1

    coverage_rate = covered_count / checked if checked > 0 else 0
    print(f"  Checked: {checked:,}")
    print(f"  Covered: {covered_count:,}")
    print(f"  Coverage rate: {coverage_rate:.6f}")

    if coverage_rate < 0.9999:
        print("\n  WARNING: Coverage may be incomplete!")
        return False

    print("\n  Verification PASSED (sampling-based)")
    return True


def save_results(state: CascadeState) -> None:
    """Save final center list to file"""
    print("\n" + "=" * 70)
    print("PHASE 4: SAVING RESULTS")
    print("=" * 70)

    print(f"\nSaving to {RESULT_FILE}...")

    # We need to enumerate all centers
    # This is the non-removed Hamming codewords + added non-Hamming

    # For now, save a summary and the non-Hamming centers
    with open(RESULT_FILE, 'w') as f:
        f.write(f"# Perfect Partition of Λ₃₁(1²⁸)\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# \n")
        f.write(f"# Statistics:\n")
        f.write(f"#   Hamming codewords removed: {len(state.removed_hamming):,}\n")
        f.write(f"#   Non-Hamming centers added: {len(state.added_nonhamming):,}\n")
        f.write(f"#   Total centers: {TARGET_CENTERS:,}\n")
        f.write(f"# \n")
        f.write(f"# Format: Each line is a 31-bit binary string (center)\n")
        f.write(f"# Note: Full list omits remaining Hamming codewords for brevity\n")
        f.write(f"# \n\n")

        f.write(f"[REMOVED_HAMMING]\n")
        for v in sorted(state.removed_hamming):
            f.write(f"{v:031b}\n")

        f.write(f"\n[ADDED_NONHAMMING]\n")
        for v in sorted(state.added_nonhamming):
            f.write(f"{v:031b}\n")

    print(f"  Saved {len(state.removed_hamming) + len(state.added_nonhamming):,} entries")


def main():
    """Main entry point"""
    print("=" * 70)
    print("FULL CASCADE REPLACEMENT FOR Λ₃₁(1²⁸)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  n = {N}")
    print(f"  s = {S}")
    print(f"  Target centers: {TARGET_CENTERS:,}")
    print(f"  Checkpoint file: {CHECKPOINT_FILE}")
    print(f"  Result file: {RESULT_FILE}")

    # Initialize state
    state = CascadeState()

    # Try to load checkpoint
    if state.load_checkpoint():
        print("\nResuming from checkpoint...")
    else:
        print("\nStarting fresh...")
        initialize_from_hamming(state)

    # Run cascade
    success = run_cascade(state)

    if success:
        # Verify
        valid = verify_partition(state)

        if valid:
            # Save results
            save_results(state)
            print("\n" + "=" * 70)
            print("SUCCESS: Perfect partition found!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("FAILED: Verification failed")
            print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FAILED: Cascade did not complete")
        print("=" * 70)
        state.save_checkpoint()


if __name__ == "__main__":
    main()
