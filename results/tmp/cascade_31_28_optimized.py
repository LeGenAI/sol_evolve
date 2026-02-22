#!/usr/bin/env python3
"""
Optimized Cascade Replacement Algorithm for Λ₃₁(1²⁸)

This version uses generator matrix approach to efficiently enumerate
all Hamming codewords and track the cascade.

Key insight: Instead of checking if arbitrary integers are codewords,
we generate codewords directly from their 26-bit information vectors.

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from collections import deque
import time
import pickle
import os
from typing import Set, Tuple, List, Optional
from functools import lru_cache

# Constants
N = 31
S = 28
K = 5  # Redundancy bits
INFO_BITS = N - K  # 26 information bits
NUM_CODEWORDS = 2**INFO_BITS  # 67,108,864
TARGET_CENTERS = NUM_CODEWORDS - 1  # 67,108,863

CHECKPOINT_FILE = "cascade_31_28_opt_checkpoint.pkl"
RESULT_FILE = "lambda_31_28_centers_opt.txt"

# ============================================================================
# Generator Matrix Construction
# ============================================================================

def build_generator_matrix():
    """
    Build the systematic generator matrix G for Hamming(31, 26).

    G is 26 x 31 matrix in systematic form: G = [I_26 | P]
    where P is the 26 x 5 parity matrix.

    Codeword c = m * G for 26-bit message m.
    """
    # Parity check matrix H (5 x 31)
    # Column j (0-indexed) is binary representation of j+1
    H = np.zeros((K, N), dtype=np.uint8)
    for j in range(N):
        col_val = j + 1
        for i in range(K):
            H[i, j] = (col_val >> i) & 1

    # For systematic form, we need to rearrange columns
    # Non-parity positions: columns where col_val has >1 bit set
    # Parity positions: columns 0, 1, 3, 7, 15 (powers of 2 minus 1)
    # Actually in standard Hamming: positions 1,2,4,8,16 are parity (1-indexed)
    # So 0-indexed: 0,1,3,7,15 are parity positions

    parity_positions = [0, 1, 3, 7, 15]  # 0-indexed positions of parity bits
    info_positions = [i for i in range(N) if i not in parity_positions]

    # Generator matrix in standard form
    # For encoding: info bits go to info_positions, parity bits computed

    G = np.zeros((INFO_BITS, N), dtype=np.uint8)

    # Identity part: each info bit maps to its position
    for idx, pos in enumerate(info_positions):
        G[idx, pos] = 1

    # Parity part: compute from H
    # c[parity_pos[k]] = sum of c[j] for j in info_positions where H[k,j]=1
    for k, ppos in enumerate(parity_positions):
        for idx, ipos in enumerate(info_positions):
            if H[k, ipos] == 1:
                G[idx, ppos] = 1

    return G, info_positions, parity_positions

G_MATRIX, INFO_POSITIONS, PARITY_POSITIONS = build_generator_matrix()

def info_to_codeword(info: int) -> int:
    """Convert 26-bit information to 31-bit codeword"""
    codeword = 0

    # Place info bits
    for idx, pos in enumerate(INFO_POSITIONS):
        if (info >> idx) & 1:
            codeword |= (1 << pos)

    # Compute parity bits
    for k, ppos in enumerate(PARITY_POSITIONS):
        parity = 0
        for idx, ipos in enumerate(INFO_POSITIONS):
            if G_MATRIX[idx, ppos] == 1 and ((info >> idx) & 1):
                parity ^= 1
        if parity:
            codeword |= (1 << ppos)

    return codeword

def codeword_to_info(cw: int) -> int:
    """Extract 26-bit information from 31-bit codeword"""
    info = 0
    for idx, pos in enumerate(INFO_POSITIONS):
        if (cw >> pos) & 1:
            info |= (1 << idx)
    return info

# Verify the construction
def verify_generator():
    """Verify generator matrix produces valid codewords"""
    print("Verifying generator matrix...")

    # Build parity check matrix
    H = np.zeros((K, N), dtype=np.uint8)
    for j in range(N):
        col_val = j + 1
        for i in range(K):
            H[i, j] = (col_val >> i) & 1

    # Test a few codewords
    test_count = 1000
    valid = 0
    for info in range(test_count):
        cw = info_to_codeword(info)

        # Check syndrome
        bits = np.array([(cw >> i) & 1 for i in range(N)], dtype=np.uint8)
        syn = H @ bits % 2

        if np.all(syn == 0):
            valid += 1

            # Verify round-trip
            recovered = codeword_to_info(cw)
            if recovered != info:
                print(f"  Round-trip failed: {info} -> {cw} -> {recovered}")

    print(f"  Tested {test_count} codewords: {valid} valid (expected {test_count})")
    return valid == test_count

# ============================================================================
# Optimized bit operations
# ============================================================================

def popcount(v: int) -> int:
    return bin(v).count('1')

def has_circular_ones_int(v: int, s: int = S) -> bool:
    """Check if v contains s consecutive 1s in circular form"""
    doubled = v | (v << N)
    mask = (1 << s) - 1

    for i in range(N):
        window = (doubled >> i) & mask
        if window == mask:
            return True
    return False

def get_neighbors_int(v: int) -> List[int]:
    return [v ^ (1 << i) for i in range(N)]

def compute_syndrome(v: int) -> int:
    """Compute syndrome as 5-bit integer"""
    syn = 0
    for i in range(K):
        parity = 0
        for j in range(N):
            if (v >> j) & 1:
                if ((j + 1) >> i) & 1:
                    parity ^= 1
        syn |= (parity << i)
    return syn

def syndrome_decode(v: int) -> Tuple[int, int]:
    """
    Decode v to nearest codeword.
    Returns (codeword, distance).
    """
    syn = compute_syndrome(v)
    if syn == 0:
        return v, 0

    # Syndrome gives error position (1-indexed)
    error_pos = syn - 1
    if 0 <= error_pos < N:
        corrected = v ^ (1 << error_pos)
        return corrected, 1

    return v, -1  # Should not happen for single-bit errors

# ============================================================================
# Main cascade algorithm
# ============================================================================

def find_bad_codewords() -> Set[int]:
    """Find all bad codewords (contain circular 1^s)"""
    bad = set()

    print("  Scanning for bad codewords...")
    start = time.time()

    count = 0
    for info in range(NUM_CODEWORDS):
        cw = info_to_codeword(info)
        if has_circular_ones_int(cw):
            bad.add(cw)
            count += 1

        if info % 10_000_000 == 0 and info > 0:
            elapsed = time.time() - start
            progress = info / NUM_CODEWORDS * 100
            print(f"    Progress: {progress:.1f}%, found {count} bad, time: {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"  Scan complete: {len(bad)} bad codewords in {elapsed:.1f}s")

    return bad


def find_uncovered_from_removed(removed: Set[int], added: Set[int]) -> List[int]:
    """Find allowed vertices not covered after removing codewords"""
    uncovered = []

    # Check neighborhood of removed codewords
    checked = set()

    for r in removed:
        # Check the removed codeword itself (if allowed)
        if not has_circular_ones_int(r) and r not in checked:
            checked.add(r)

            # Is it covered?
            covered = False
            if r in added:
                covered = True
            else:
                cw, dist = syndrome_decode(r)
                if dist == 0 and cw not in removed:
                    covered = True
                elif dist == 1 and cw not in removed:
                    covered = True
                else:
                    # Check if any added center covers it
                    for neighbor in get_neighbors_int(r):
                        if neighbor in added:
                            covered = True
                            break

            if not covered:
                uncovered.append(r)

        # Check neighbors of removed codeword
        for neighbor in get_neighbors_int(r):
            if has_circular_ones_int(neighbor):
                continue
            if neighbor in checked:
                continue

            checked.add(neighbor)

            # Is neighbor covered?
            covered = False
            if neighbor in added:
                covered = True
            else:
                cw, dist = syndrome_decode(neighbor)
                if dist == 0:
                    if cw not in removed:
                        covered = True
                elif dist == 1:
                    if cw not in removed:
                        covered = True

                if not covered:
                    # Check added centers
                    for n2 in get_neighbors_int(neighbor):
                        if n2 in added:
                            covered = True
                            break

            if not covered:
                uncovered.append(neighbor)

    return uncovered


def find_conflicts(v: int, removed: Set[int], added: Set[int]) -> List[int]:
    """Find codewords at distance < 3 from v that would conflict"""
    conflicts = []

    # Distance 1
    for n1 in get_neighbors_int(v):
        cw, dist = syndrome_decode(n1)
        if dist == 0 and cw not in removed and not has_circular_ones_int(cw):
            conflicts.append(cw)
        elif dist == 1:
            # cw is at distance 2 from v
            if cw not in removed and not has_circular_ones_int(cw):
                conflicts.append(cw)

    # Distance 2
    for i in range(N):
        for j in range(i+1, N):
            n2 = v ^ (1 << i) ^ (1 << j)
            cw, dist = syndrome_decode(n2)
            if dist == 0 and cw not in removed and not has_circular_ones_int(cw):
                if cw not in conflicts:
                    conflicts.append(cw)

    return conflicts


class CascadeStateOpt:
    """Optimized cascade state"""

    def __init__(self):
        self.removed: Set[int] = set()  # Removed Hamming codewords
        self.added: Set[int] = set()     # Added non-Hamming centers
        self.queue: deque = deque()      # Codewords to process
        self.processed: int = 0
        self.start_time: float = 0

    def save(self, filename: str = CHECKPOINT_FILE):
        with open(filename, 'wb') as f:
            pickle.dump({
                'removed': self.removed,
                'added': self.added,
                'queue': list(self.queue),
                'processed': self.processed,
            }, f)
        print(f"  [Checkpoint: removed={len(self.removed):,}, added={len(self.added):,}]")

    def load(self, filename: str = CHECKPOINT_FILE) -> bool:
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.removed = data['removed']
            self.added = data['added']
            self.queue = deque(data['queue'])
            self.processed = data['processed']
            print(f"  [Loaded: removed={len(self.removed):,}, added={len(self.added):,}]")
            return True
        except:
            return False


def run_cascade_optimized(max_depth: int = 100, checkpoint_interval: int = 10000):
    """Run optimized cascade algorithm"""

    print("=" * 70)
    print("OPTIMIZED CASCADE FOR Λ₃₁(1²⁸)")
    print("=" * 70)

    state = CascadeStateOpt()

    # Try loading checkpoint
    if state.load():
        print("\nResuming from checkpoint...")
    else:
        print("\nInitializing...")

        # Find all bad codewords
        bad = find_bad_codewords()

        # Initialize queue with bad codewords
        for b in bad:
            state.removed.add(b)
            state.queue.append(b)

        print(f"Initial bad codewords: {len(bad)}")

    # Main cascade loop
    print("\n" + "=" * 70)
    print("RUNNING CASCADE")
    print("=" * 70)

    state.start_time = time.time()
    depth = 0
    last_checkpoint = 0

    while state.queue:
        depth += 1
        if depth > max_depth:
            print(f"\nReached max depth {max_depth}")
            break

        # Process current depth
        current_batch = list(state.queue)
        state.queue.clear()

        print(f"\nDepth {depth}: processing {len(current_batch):,} codewords")

        # Find all uncovered vertices
        uncovered = find_uncovered_from_removed(state.removed, state.added)
        print(f"  Uncovered vertices: {len(uncovered):,}")

        if not uncovered:
            print("  All vertices covered!")
            break

        # For each uncovered vertex, try to repair or cascade
        new_conflicts = set()
        repaired = 0

        for v in uncovered:
            state.processed += 1

            # Try v itself as repair center
            if not has_circular_ones_int(v):
                conflicts = find_conflicts(v, state.removed, state.added)

                if not conflicts:
                    # Can use v as repair center
                    state.added.add(v)
                    repaired += 1
                else:
                    # Need to remove conflicting codewords
                    for c in conflicts:
                        if c not in state.removed:
                            new_conflicts.add(c)

        print(f"  Repaired: {repaired:,}")
        print(f"  New conflicts: {len(new_conflicts):,}")

        # Add new conflicts to queue
        for c in new_conflicts:
            state.removed.add(c)
            state.queue.append(c)

        # Status
        elapsed = time.time() - state.start_time
        print(f"  Total removed: {len(state.removed):,}")
        print(f"  Total added: {len(state.added):,}")
        print(f"  Time: {elapsed:.1f}s")

        # Checkpoint
        if state.processed - last_checkpoint >= checkpoint_interval:
            state.save()
            last_checkpoint = state.processed

    # Final status
    print("\n" + "=" * 70)
    print("CASCADE COMPLETE")
    print("=" * 70)

    hamming_remaining = NUM_CODEWORDS - len(state.removed)
    total_centers = hamming_remaining + len(state.added)

    print(f"\nFinal statistics:")
    print(f"  Hamming remaining: {hamming_remaining:,} ({100*hamming_remaining/NUM_CODEWORDS:.2f}%)")
    print(f"  Non-Hamming added: {len(state.added):,}")
    print(f"  Total centers: {total_centers:,}")
    print(f"  Target: {TARGET_CENTERS:,}")
    print(f"  Difference: {total_centers - TARGET_CENTERS:,}")

    # Save final checkpoint
    state.save()

    return state


def main():
    # Verify generator matrix first
    if not verify_generator():
        print("Generator matrix verification failed!")
        return

    # Run cascade
    state = run_cascade_optimized(max_depth=50)


if __name__ == "__main__":
    main()
