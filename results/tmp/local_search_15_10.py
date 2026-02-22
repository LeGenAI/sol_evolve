#!/usr/bin/env python3
"""
Local Search for Lambda_15(1^10) Perfect Partition.

Strategy:
1. Start with a greedy solution (maximizing coverage)
2. Use Simulated Annealing to escape local minima
3. Moves: swap, add, remove centers
4. Energy: uncovered_count + overlap_count * penalty

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import random
import time
import math

def has_circ(v, n, s):
    doubled = (v << n) | v
    mask = (1 << s) - 1
    for i in range(n):
        if ((doubled >> i) & mask) == mask:
            return True
    return False

def hamming(a, b):
    return bin(a ^ b).count('1')

def to_bin(v, n):
    return format(v, f'0{n}b')

class LocalSearch:
    def __init__(self, n=15, s=10):
        self.n = n
        self.s = s

        print(f"Initializing Lambda_{n}(1^{s})...")
        self.verts = [v for v in range(1 << n) if not has_circ(v, n, s)]
        self.vset = set(self.verts)
        self.num_verts = len(self.verts)
        print(f"|V| = {self.num_verts}")

        # Precompute balls and membership
        print("Precomputing balls...")
        self.balls = {}
        self.ball_sizes = {}
        self.membership = defaultdict(list)  # vertex -> list of centers that cover it

        for v in self.verts:
            ball = [v]
            for i in range(n):
                nb = v ^ (1 << i)
                if nb in self.vset:
                    ball.append(nb)
            self.balls[v] = ball
            self.ball_sizes[v] = len(ball)
            for u in ball:
                self.membership[u].append(v)

        # Precompute close pairs (d < 3)
        print("Precomputing close pairs...")
        self.close_to = defaultdict(set)  # v -> set of vertices at distance < 3
        for v in self.verts:
            # distance 1
            for i in range(n):
                nb = v ^ (1 << i)
                if nb in self.vset:
                    self.close_to[v].add(nb)
            # distance 2
            for i in range(n):
                for j in range(i + 1, n):
                    nb = v ^ (1 << i) ^ (1 << j)
                    if nb in self.vset:
                        self.close_to[v].add(nb)

        print("Initialization complete.")

    def compute_state(self, centers):
        """Compute coverage state given a set of centers."""
        center_set = set(centers)
        coverage = defaultdict(int)  # vertex -> count of covering centers

        for c in centers:
            for v in self.balls[c]:
                coverage[v] += 1

        uncovered = [v for v in self.verts if coverage[v] == 0]
        overlaps = sum(1 for v in self.verts if coverage[v] > 1)

        return coverage, uncovered, overlaps

    def energy(self, coverage, uncovered_count, overlap_count, center_count):
        """Compute energy (lower is better)."""
        # Primary: minimize uncovered + overlaps
        # Secondary: prefer fewer centers
        return uncovered_count * 100 + overlap_count * 100 + center_count * 0.1

    def greedy_initial(self, prioritize_small_balls=True):
        """Generate initial solution using greedy algorithm."""
        print("\nGenerating greedy initial solution...")

        centers = []
        center_set = set()
        covered = set()
        blocked = set()  # vertices that can't be centers (too close)

        # Sort by ball size (smaller first if prioritize_small_balls)
        if prioritize_small_balls:
            candidates = sorted(self.verts, key=lambda v: (self.ball_sizes[v], v))
        else:
            candidates = sorted(self.verts, key=lambda v: (-self.ball_sizes[v], v))

        for c in candidates:
            if c in blocked:
                continue

            ball = self.balls[c]

            # Check if would cause overlap
            if any(v in covered for v in ball):
                continue

            # Add center
            centers.append(c)
            center_set.add(c)
            covered.update(ball)

            # Block close vertices
            blocked.update(self.close_to[c])

        coverage, uncovered, overlaps = self.compute_state(centers)
        print(f"Greedy result: {len(centers)} centers, {len(uncovered)} uncovered, {overlaps} overlaps")

        return centers

    def find_valid_additions(self, center_set, uncovered_set):
        """Find centers that can be added without violating packing."""
        valid = []
        for v in self.verts:
            if v in center_set:
                continue
            # Check packing constraint
            if any(c in self.close_to[v] for c in center_set):
                continue
            # Check if it covers any uncovered vertex
            ball = self.balls[v]
            covers_uncovered = any(u in uncovered_set for u in ball)
            if covers_uncovered:
                valid.append(v)
        return valid

    def find_swap_candidates(self, center, center_set, coverage):
        """Find vertices that could replace a center."""
        # Find vertices covered by this center
        ball = self.balls[center]

        # Temporarily remove center
        temp_centers = center_set - {center}

        # Find candidates that:
        # 1. Are not too close to remaining centers
        # 2. Cover at least some of the vertices that would become uncovered
        candidates = []

        would_uncover = [v for v in ball if coverage[v] == 1]  # only covered by this center

        for c in self.verts:
            if c in temp_centers:
                continue
            if any(tc in self.close_to[c] for tc in temp_centers):
                continue

            # Check how many it would cover
            new_ball = self.balls[c]
            recovery = len(set(new_ball) & set(would_uncover))
            if recovery > 0:
                candidates.append((c, recovery))

        return sorted(candidates, key=lambda x: -x[1])

    def simulated_annealing(self, initial_centers, max_iter=100000, T0=100, alpha=0.9995):
        """Simulated Annealing optimization."""
        print(f"\nStarting Simulated Annealing (max_iter={max_iter}, T0={T0})...")

        centers = list(initial_centers)
        center_set = set(centers)
        coverage, uncovered, overlaps = self.compute_state(centers)
        uncovered_set = set(uncovered)

        current_energy = self.energy(coverage, len(uncovered), overlaps, len(centers))
        best_centers = centers.copy()
        best_energy = current_energy
        best_uncovered = len(uncovered)

        T = T0
        accepted = 0
        improved = 0

        start_time = time.time()

        for iteration in range(max_iter):
            if iteration % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {iteration}: E={current_energy:.1f}, uncovered={len(uncovered)}, "
                      f"overlaps={overlaps}, centers={len(centers)}, T={T:.2f}, "
                      f"time={elapsed:.1f}s")

                if len(uncovered) == 0 and overlaps == 0:
                    print("  PERFECT PARTITION FOUND!")
                    return centers, True

            # Choose move type
            move_type = random.choice(['swap', 'add', 'remove'])

            new_centers = centers.copy()
            new_center_set = center_set.copy()

            if move_type == 'swap' and centers:
                # Swap: remove one center, add another
                old_center = random.choice(centers)
                candidates = self.find_swap_candidates(old_center, center_set, coverage)

                if candidates:
                    new_center, _ = random.choice(candidates[:10])  # top 10
                    new_centers.remove(old_center)
                    new_centers.append(new_center)
                    new_center_set.remove(old_center)
                    new_center_set.add(new_center)
                else:
                    continue

            elif move_type == 'add':
                # Add: add a new center
                valid = self.find_valid_additions(center_set, uncovered_set)
                if valid:
                    new_center = random.choice(valid[:20])  # random from top candidates
                    new_centers.append(new_center)
                    new_center_set.add(new_center)
                else:
                    continue

            elif move_type == 'remove' and len(centers) > 1:
                # Remove: remove a center (risky but can help escape)
                # Prefer removing centers that cause overlaps or cover already-covered vertices
                candidates = []
                for c in centers:
                    ball = self.balls[c]
                    multi_covered = sum(1 for v in ball if coverage[v] > 1)
                    candidates.append((c, multi_covered))

                candidates.sort(key=lambda x: -x[1])
                if candidates[0][1] > 0:
                    old_center = candidates[0][0]
                else:
                    old_center = random.choice(centers)

                new_centers.remove(old_center)
                new_center_set.remove(old_center)

            else:
                continue

            # Evaluate new state
            new_coverage, new_uncovered, new_overlaps = self.compute_state(new_centers)
            new_energy = self.energy(new_coverage, len(new_uncovered), new_overlaps, len(new_centers))

            # Accept or reject
            delta = new_energy - current_energy

            if delta < 0 or random.random() < math.exp(-delta / T):
                centers = new_centers
                center_set = new_center_set
                coverage = new_coverage
                uncovered = new_uncovered
                uncovered_set = set(uncovered)
                overlaps = new_overlaps
                current_energy = new_energy
                accepted += 1

                if new_energy < best_energy:
                    best_centers = centers.copy()
                    best_energy = new_energy
                    best_uncovered = len(uncovered)
                    improved += 1

                    if len(new_uncovered) == 0 and new_overlaps == 0:
                        print(f"  PERFECT PARTITION FOUND at iteration {iteration}!")
                        return centers, True

            # Cool down
            T *= alpha
            if T < 0.01:
                T = 0.01

        print(f"\nSA finished. Best: {len(best_centers)} centers, {best_uncovered} uncovered")
        print(f"Accepted: {accepted}/{max_iter}, Improved: {improved}")

        return best_centers, (best_uncovered == 0)

    def verify(self, centers):
        """Verify if centers form a perfect partition."""
        center_set = set(centers)

        # Check all centers are valid
        for c in centers:
            if c not in self.vset:
                return False, f"Invalid center {c}"

        # Check packing
        for i, c1 in enumerate(centers):
            for c2 in centers[i+1:]:
                if hamming(c1, c2) < 3:
                    return False, f"Centers too close: d({to_bin(c1,self.n)}, {to_bin(c2,self.n)})={hamming(c1,c2)}"

        # Check coverage
        coverage, uncovered, overlaps = self.compute_state(centers)

        if uncovered:
            return False, f"{len(uncovered)} uncovered vertices"

        if overlaps > 0:
            return False, f"{overlaps} overlapping vertices"

        return True, "Perfect partition verified!"

    def save_solution(self, centers, filename_prefix):
        """Save solution to files."""
        np.save(f"{filename_prefix}_centers.npy", np.array(centers, dtype=np.uint32))

        with open(f"{filename_prefix}_solution.txt", 'w') as f:
            f.write(f"Lambda_{self.n}(1^{self.s}) Perfect Partition\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"|V| = {self.num_verts}\n")
            f.write(f"Centers = {len(centers)}\n\n")
            f.write("Centers (binary):\n")
            for i, c in enumerate(sorted(centers)):
                f.write(f"{i+1:4d}. {to_bin(c, self.n)} (ball size {self.ball_sizes[c]})\n")

        print(f"Solution saved to {filename_prefix}*")


def main():
    random.seed(42)

    ls = LocalSearch(n=15, s=10)

    # Try multiple restarts
    best_solution = None
    best_uncovered = float('inf')

    for restart in range(5):
        print(f"\n{'='*60}")
        print(f"Restart {restart + 1}")
        print("=" * 60)

        # Different initialization strategies
        if restart % 2 == 0:
            initial = ls.greedy_initial(prioritize_small_balls=True)
        else:
            initial = ls.greedy_initial(prioritize_small_balls=False)

        # Run SA
        solution, found = ls.simulated_annealing(
            initial,
            max_iter=200000,
            T0=50,
            alpha=0.99995
        )

        coverage, uncovered, overlaps = ls.compute_state(solution)

        if len(uncovered) < best_uncovered:
            best_uncovered = len(uncovered)
            best_solution = solution.copy()

        if found:
            ok, msg = ls.verify(solution)
            print(f"\nVerification: {msg}")

            if ok:
                ls.save_solution(solution,
                    "/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10")
                return

    print(f"\n{'='*60}")
    print(f"Best solution: {len(best_solution)} centers, {best_uncovered} uncovered")
    print("=" * 60)

    # Save best (even if not perfect)
    ls.save_solution(best_solution,
        "/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_best")


if __name__ == "__main__":
    main()
