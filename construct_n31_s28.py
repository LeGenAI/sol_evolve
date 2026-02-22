#!/usr/bin/env python3
"""
Syndrome-Based Construction for n=31, s=28.

Based on SYNDROME_SHIFT_DISCOVERY.md:
- For s = n-3, allowed syndromes are {1, 2, 3, n-3, n-2, n-1, n}
- For n=31, s=28: syndromes {1, 2, 3, 28, 29, 30, 31} → bits {0, 1, 2, 27, 28, 29, 30}

Strategy:
1. Generate Hamming codewords incrementally (via message encoding)
2. Apply greedy syndrome shift assignment
3. Use spatial hashing for fast conflict detection
4. Verify covering by sampling

Author: Jae-Hyun Baek
Date: 2025-11-26
"""

import numpy as np
from typing import Set, Dict, List, Tuple, Optional
from collections import defaultdict
import time
import sys


def has_circular_run(v: int, n: int, s: int) -> bool:
    """Check if v has s consecutive 1s (circular)."""
    bits = format(v, f'0{n}b')
    doubled = bits + bits[:-1]
    return '1' * s in doubled


def hamming_weight(v: int) -> int:
    return bin(v).count('1')


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def is_allowed(v: int, n: int, s: int) -> bool:
    return not has_circular_run(v, n, s)


def get_ball(v: int, n: int) -> List[int]:
    """Get all vertices at distance ≤ 1 from v."""
    ball = [v]
    for i in range(n):
        ball.append(v ^ (1 << i))
    return ball


class SpatialHash:
    """
    Spatial hash for fast conflict detection.

    Key insight: Two vectors conflict (distance < 3) if they share at least n-2 bits.
    We hash by "neighborhoods" of bit patterns.
    """

    def __init__(self, n: int):
        self.n = n
        self.centers = set()
        # Hash by different bit masks for faster lookup
        self.by_prefix = defaultdict(set)  # First 16 bits
        self.by_suffix = defaultdict(set)  # Last 16 bits

    def add(self, v: int):
        self.centers.add(v)
        self.by_prefix[v >> 16].add(v)
        self.by_suffix[v & 0xFFFF].add(v)

    def has_conflict(self, v: int) -> bool:
        """Check if v has distance < 3 from any center."""
        # Check exact match and distance-1 neighbors
        if v in self.centers:
            return True

        # Check distance 1 (flip one bit)
        for i in range(self.n):
            neighbor = v ^ (1 << i)
            if neighbor in self.centers:
                return True

        # Check distance 2 (flip two bits)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                neighbor = v ^ (1 << i) ^ (1 << j)
                if neighbor in self.centers:
                    return True

        return False

    def has_conflict_fast(self, v: int) -> bool:
        """Faster conflict check using spatial hashing."""
        # Get candidate centers from same hash buckets
        candidates = self.by_prefix.get(v >> 16, set()) | self.by_suffix.get(v & 0xFFFF, set())

        # For remaining centers, we need to check all (for correctness)
        # But most conflicts should be caught by spatial proximity
        for c in candidates:
            if hamming_distance(v, c) < 3:
                return True

        # Full check for centers not in candidates (slower)
        remaining = self.centers - candidates
        for c in remaining:
            if hamming_distance(v, c) < 3:
                return True

        return False


def generate_hamming_codeword(msg: int, k: int) -> int:
    """
    Generate a single Hamming codeword from message.

    For Ham(n, n-k) where n = 2^k - 1:
    - Message has n-k bits
    - Codeword has n bits (message + k parity bits)
    """
    n = (1 << k) - 1
    n_k = n - k

    # Parity positions are 1, 2, 4, 8, ... (1-indexed)
    parity_positions = [(1 << i) for i in range(k)]

    # Data positions are non-powers of 2 (1-indexed)
    data_positions = [i for i in range(1, n + 1) if i not in parity_positions]

    codeword = 0

    # Place message bits in data positions
    for bit_idx, pos in enumerate(data_positions):
        if (msg >> bit_idx) & 1:
            codeword |= (1 << (pos - 1))  # Convert to 0-indexed

    # Compute parity bits
    for p_idx in range(k):
        parity_pos = parity_positions[p_idx]
        parity = 0
        for i in range(1, n + 1):
            if i & parity_pos:
                if (codeword >> (i - 1)) & 1:
                    parity ^= 1
        if parity:
            codeword |= (1 << (parity_pos - 1))

    return codeword


def construct_n31_s28(progress_interval: int = 1000000,
                      checkpoint_interval: int = 10000000,
                      max_codewords: Optional[int] = None):
    """
    Construct perfect partition for Λ₃₁(1²⁸).

    Uses streaming approach:
    1. Generate Hamming codewords one at a time
    2. Apply greedy syndrome shift
    3. Track statistics
    """
    k = 5
    n = 31
    s = 28

    # Predicted allowed shift bits (from syndrome pattern)
    # Syndromes {1, 2, 3, 28, 29, 30, 31} → bits {0, 1, 2, 27, 28, 29, 30}
    shift_bits = [0, 1, 2, 27, 28, 29, 30]

    print("="*70)
    print(f"Constructing Perfect Partition for Λ_{n}(1^{s})")
    print("="*70)
    print(f"k = {k}, n = {n}, s = {s}")
    print(f"Allowed shift bits: {shift_bits}")
    print(f"|Hamming| = 2^{n-k} = {1 << (n-k):,}")

    total_codewords = 1 << (n - k)  # 2^26 = 67,108,864
    if max_codewords:
        total_codewords = min(total_codewords, max_codewords)
        print(f"Processing first {total_codewords:,} codewords")

    # Initialize
    spatial_hash = SpatialHash(n)

    stats = {
        'stayed': 0,
        'shifted': defaultdict(int),  # syndrome -> count
        'removed': 0,
        'processed': 0,
        'bad_codewords': 0
    }

    start_time = time.time()
    last_checkpoint = start_time

    print("\nProcessing codewords...")

    for msg in range(total_codewords):
        codeword = generate_hamming_codeword(msg, k)
        stats['processed'] += 1

        # Check if codeword is valid (not forbidden)
        codeword_valid = is_allowed(codeword, n, s)
        if not codeword_valid:
            stats['bad_codewords'] += 1

        placed = False

        # Strategy 1: Try to stay if valid and no conflict
        if codeword_valid:
            if not spatial_hash.has_conflict(codeword):
                spatial_hash.add(codeword)
                stats['stayed'] += 1
                placed = True

        # Strategy 2: Try shifts in order of preference
        if not placed:
            for bit in shift_bits:
                shifted = codeword ^ (1 << bit)

                # Check if shifted position is valid
                if not is_allowed(shifted, n, s):
                    continue

                # Check for conflicts
                if not spatial_hash.has_conflict(shifted):
                    spatial_hash.add(shifted)
                    syndrome = bit + 1
                    stats['shifted'][syndrome] += 1
                    placed = True
                    break

        # Strategy 3: Mark as removed if nothing works
        if not placed:
            stats['removed'] += 1

        # Progress reporting
        if stats['processed'] % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = stats['processed'] / elapsed
            eta = (total_codewords - stats['processed']) / rate if rate > 0 else 0

            print(f"  {stats['processed']:,} / {total_codewords:,} ({100*stats['processed']/total_codewords:.1f}%)"
                  f" | Centers: {len(spatial_hash.centers):,}"
                  f" | Removed: {stats['removed']:,}"
                  f" | Rate: {rate:.0f}/s | ETA: {eta/60:.1f}m")

        # Checkpoint
        if stats['processed'] % checkpoint_interval == 0:
            current_time = time.time()
            checkpoint_elapsed = current_time - last_checkpoint
            last_checkpoint = current_time

            print(f"\n--- Checkpoint at {stats['processed']:,} ---")
            print(f"Centers: {len(spatial_hash.centers):,}")
            print(f"Stayed: {stats['stayed']:,}")
            print(f"Shifted: {sum(stats['shifted'].values()):,}")
            print(f"  By syndrome: {dict(stats['shifted'])}")
            print(f"Removed: {stats['removed']:,}")
            print(f"Bad codewords: {stats['bad_codewords']:,}")
            print(f"Checkpoint time: {checkpoint_elapsed:.1f}s\n")

    # Final summary
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Total processed: {stats['processed']:,}")
    print(f"Total centers: {len(spatial_hash.centers):,}")
    print(f"Expected centers: {total_codewords - 1:,} (minus all-ones)")
    print(f"\nDistribution:")
    print(f"  Stayed (syndrome 0): {stats['stayed']:,} ({100*stats['stayed']/stats['processed']:.1f}%)")

    total_shifted = sum(stats['shifted'].values())
    print(f"  Shifted: {total_shifted:,} ({100*total_shifted/stats['processed']:.1f}%)")
    for syn in sorted(stats['shifted'].keys()):
        count = stats['shifted'][syn]
        print(f"    Syndrome {syn}: {count:,} ({100*count/total_shifted:.1f}% of shifts)")

    print(f"  Removed: {stats['removed']:,} ({100*stats['removed']/stats['processed']:.2f}%)")
    print(f"  Bad codewords: {stats['bad_codewords']:,}")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Verify packing (sample)
    print("\n--- Verification (sampling) ---")
    centers_list = list(spatial_hash.centers)
    sample_size = min(10000, len(centers_list))

    min_dist = float('inf')
    violations = 0
    np.random.seed(42)
    sample_indices = np.random.choice(len(centers_list), size=sample_size, replace=False)

    for i in range(min(1000, sample_size)):
        for j in range(i + 1, min(i + 100, sample_size)):
            c1 = centers_list[sample_indices[i]]
            c2 = centers_list[sample_indices[j]]
            d = hamming_distance(c1, c2)
            min_dist = min(min_dist, d)
            if d < 3:
                violations += 1

    print(f"Sampled min distance: {min_dist}")
    print(f"Sampled packing violations: {violations}")

    if stats['removed'] <= 1 and violations == 0:
        print("\n*** LIKELY SUCCESS! ***")
        print(f"Only {stats['removed']} codewords removed (expected: 1 for all-ones)")

        # Save centers
        output_file = f'constructed_n{n}_s{s}_centers.npy'
        np.save(output_file, np.array(centers_list, dtype=np.int64))
        print(f"Saved {len(centers_list):,} centers to {output_file}")
    else:
        print(f"\n*** INCOMPLETE: {stats['removed']} removed, {violations} violations ***")

    return spatial_hash.centers, stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Construct perfect partition for n=31, s=28')
    parser.add_argument('--max', type=int, default=None,
                        help='Max codewords to process (default: all)')
    parser.add_argument('--progress', type=int, default=1000000,
                        help='Progress interval')
    parser.add_argument('--checkpoint', type=int, default=10000000,
                        help='Checkpoint interval')

    args = parser.parse_args()

    construct_n31_s28(
        progress_interval=args.progress,
        checkpoint_interval=args.checkpoint,
        max_codewords=args.max
    )


if __name__ == "__main__":
    main()
