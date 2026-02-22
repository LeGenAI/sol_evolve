#!/usr/bin/env python3
"""
Analyze bad codewords for n=31, s=28.

Key question: How many Hamming codewords have 28 consecutive 1s (circular)?
"""

import numpy as np


def has_circular_run(v: int, n: int, s: int) -> bool:
    bits = format(v, f'0{n}b')
    doubled = bits + bits[:-1]
    return '1' * s in doubled


def hamming_weight(v: int) -> int:
    return bin(v).count('1')


def generate_hamming_codeword(msg: int, k: int) -> int:
    n = (1 << k) - 1
    n_k = n - k
    parity_positions = [(1 << i) for i in range(k)]
    data_positions = [i for i in range(1, n + 1) if i not in parity_positions]

    codeword = 0
    for bit_idx, pos in enumerate(data_positions):
        if (msg >> bit_idx) & 1:
            codeword |= (1 << (pos - 1))

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


def main():
    k = 5
    n = 31
    s = 28

    print(f"Analyzing bad codewords for n={n}, s={s}")
    print(f"|Hamming| = 2^{n-k} = {1 << (n-k):,}")

    # For s=28, we need 28 consecutive 1s
    # This requires weight >= 28
    # Hamming codewords have specific weight distribution

    print("\nChecking high-weight codewords...")

    # Sample to find weight distribution
    sample_size = 100000
    weight_dist = {}

    for msg in range(sample_size):
        cw = generate_hamming_codeword(msg, k)
        w = hamming_weight(cw)
        weight_dist[w] = weight_dist.get(w, 0) + 1

    print(f"\nWeight distribution (sample of {sample_size:,}):")
    for w in sorted(weight_dist.keys()):
        print(f"  Weight {w:2d}: {weight_dist[w]:,}")

    # Check specific high-weight codewords
    print("\n--- Checking for bad codewords (weight >= 28) ---")

    # Enumerate all 2^26 codewords (too many, so sample or reason)
    # For weight 28, need exactly 3 zeros
    # For weight 29, need exactly 2 zeros
    # For weight 30, need exactly 1 zero
    # For weight 31, need all ones (only 1 such codeword)

    # The all-ones codeword
    all_ones = (1 << n) - 1
    cw_all_ones = None

    # Find all-ones in Hamming
    for msg in range(1 << (n - k)):
        cw = generate_hamming_codeword(msg, k)
        if cw == all_ones:
            cw_all_ones = cw
            print(f"Found all-ones codeword: {format(cw, '031b')}")
            print(f"  Weight: {hamming_weight(cw)}")
            print(f"  Has {s} consecutive 1s: {has_circular_run(cw, n, s)}")
            break

    if cw_all_ones is None:
        # Check syndrome of all-ones
        syndrome = 0
        for i in range(n):
            if (all_ones >> i) & 1:
                syndrome ^= (i + 1)
        print(f"All-ones syndrome: {syndrome}")
        if syndrome == 0:
            print("All-ones IS a Hamming codeword!")
        else:
            print("All-ones is NOT a Hamming codeword")

    # For circular 28 consecutive 1s, we need:
    # - At least 28 consecutive 1s when wrapped
    # - This means at most 3 zeros, and they must be consecutive (circular)

    print("\n--- Mathematical analysis ---")
    print(f"For {s} circular consecutive 1s:")
    print(f"  Need weight >= {s}")
    print(f"  With n={n}, need at most {n-s} zeros")
    print(f"  For circular pattern, zeros must be 'bunched' to allow 28 consecutive 1s")

    # Count codewords with weight >= 28
    print("\nEnumerating high-weight codewords...")
    high_weight_count = 0
    bad_count = 0

    total = 1 << (n - k)
    check_every = total // 100  # 1% progress

    for msg in range(total):
        if msg % check_every == 0 and msg > 0:
            print(f"  Progress: {100*msg/total:.0f}%", end='\r')

        cw = generate_hamming_codeword(msg, k)
        w = hamming_weight(cw)

        if w >= s:
            high_weight_count += 1
            if has_circular_run(cw, n, s):
                bad_count += 1
                if bad_count <= 10:
                    print(f"\n  Bad codeword: {format(cw, '031b')} (weight {w})")

    print(f"\n\nResults:")
    print(f"  High-weight (>= {s}) codewords: {high_weight_count:,}")
    print(f"  Bad codewords (circular {s} 1s): {bad_count:,}")

    if bad_count == 0:
        print(f"\n*** NO BAD CODEWORDS! Hamming code is already valid! ***")
    elif bad_count == 1:
        print(f"\n*** Only 1 bad codeword (likely all-ones) ***")


if __name__ == "__main__":
    main()
