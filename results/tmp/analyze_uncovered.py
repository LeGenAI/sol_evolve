#!/usr/bin/env python3
"""
Analyze uncovered vertices from greedy algorithm
"""

import numpy as np
from collections import Counter, defaultdict

def hamming_weight(x):
    return bin(x).count('1')

def hamming_distance(a, b):
    return bin(a ^ b).count('1')

def has_circular_consecutive_ones(n, bits, s):
    doubled = (n << bits) | n
    mask = (1 << s) - 1
    for i in range(bits):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def main():
    n, s = 15, 12

    print(f"{'='*60}")
    print(f"Analyzing Uncovered Vertices")
    print(f"{'='*60}")

    # Load data
    uncovered = np.load('uncovered.npy')
    centers = np.load('greedy_centers.npy')

    print(f"Uncovered vertices: {len(uncovered)}")
    print(f"Greedy centers: {len(centers)}")

    # Weight distribution
    print(f"\n1. Weight Distribution of Uncovered Vertices:")
    weights = [hamming_weight(v) for v in uncovered]
    weight_counts = Counter(weights)
    print(f"   {dict(sorted(weight_counts.items()))}")

    # Analyze why they're uncovered
    print(f"\n2. Distance to Nearest Center:")
    min_dists = []
    for v in uncovered:
        min_d = min(hamming_distance(v, c) for c in centers)
        min_dists.append(min_d)

    dist_counts = Counter(min_dists)
    print(f"   {dict(sorted(dist_counts.items()))}")

    # All uncovered are at distance >= 2 from centers (else they'd be covered)
    # But we need to understand WHY greedy didn't cover them

    # Find which uncovered vertices could potentially be centers
    print(f"\n3. Can Uncovered Vertices Be Centers?")

    # Generate all vertices for reference
    vertices = [v for v in range(1 << n) if not has_circular_consecutive_ones(v, n, s)]
    vertices_set = set(vertices)

    can_be_center = []
    for v in uncovered[:100]:  # Sample
        # Check if v is at distance >= 3 from all current centers
        valid = all(hamming_distance(v, c) >= 3 for c in centers)
        if valid:
            can_be_center.append(v)

    print(f"   Sample of 100 uncovered: {len(can_be_center)} could still be centers")

    # Check pairwise distances among uncovered
    print(f"\n4. Pairwise Distances Among Uncovered (sample):")
    sample_size = min(200, len(uncovered))
    sample = uncovered[:sample_size]

    pair_dists = []
    for i, v1 in enumerate(sample):
        for v2 in sample[i+1:]:
            pair_dists.append(hamming_distance(v1, v2))

    pair_dist_counts = Counter(pair_dists)
    print(f"   {dict(sorted(pair_dist_counts.items()))}")

    # How many pairs have distance < 3?
    close_pairs = sum(1 for d in pair_dists if d < 3)
    print(f"   Close pairs (d < 3): {close_pairs}")

    # Check clustering
    print(f"\n5. Clustering Analysis:")

    # Group by weight
    by_weight = defaultdict(list)
    for v in uncovered:
        by_weight[hamming_weight(v)].append(v)

    for w in sorted(by_weight.keys()):
        vlist = by_weight[w]
        # Check how many are close to each other
        close = 0
        for i, v1 in enumerate(vlist[:50]):
            for v2 in vlist[i+1:50]:
                if hamming_distance(v1, v2) < 3:
                    close += 1
        print(f"   Weight {w}: {len(vlist)} vertices, {close} close pairs (in sample)")

    # Theoretical analysis
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    # Calculate how many more centers we need
    # Each new center can cover up to 16 vertices
    # But due to packing constraints, we need to be careful

    print(f"""
Current state:
- Covered: {len(vertices) - len(uncovered)} vertices
- Uncovered: {len(uncovered)} vertices
- Centers: {len(centers)}

If perfect partition exists with k centers:
- Total ball coverage = sum of ball sizes = |V| = 32707
- Average ball size ≈ 32707 / k

From greedy:
- Ball sizes used: 14 (24), 15 (9), 16 (1878)
- Total coverage = 24*14 + 9*15 + 1878*16 = {24*14 + 9*15 + 1878*16}

Remaining coverage needed: {len(uncovered)}
Min additional centers: ceil({len(uncovered)} / 16) = {(len(uncovered) + 15) // 16}

So we need approximately {len(centers) + (len(uncovered) + 15) // 16} centers
(This is a lower bound; actual may be higher due to distance constraints)
""")

    # Check if the uncovered vertices form an independent set
    # that can be covered by non-overlapping balls
    print("Checking if uncovered vertices can form valid center set...")

    # Build graph of uncovered vertices with edges for distance < 3
    uncovered_set = set(uncovered)

    # Find maximum independent set approximation
    independent = []
    remaining = list(uncovered)

    while remaining:
        v = remaining[0]
        independent.append(v)
        # Remove v and all close vertices
        remaining = [u for u in remaining[1:] if hamming_distance(u, v) >= 3]

    print(f"Max independent set (greedy) among uncovered: {len(independent)}")
    print(f"These could potentially be added as centers")

    # Save analysis
    np.save('potential_new_centers.npy', np.array(independent))
    print(f"\nSaved potential new centers to potential_new_centers.npy")

if __name__ == "__main__":
    main()
