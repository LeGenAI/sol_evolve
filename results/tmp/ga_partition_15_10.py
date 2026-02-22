#!/usr/bin/env python3
"""
Genetic Algorithm for Lambda_15(1^10) Perfect Partition.

Strategy:
1. Population: sets of centers (encoded as bitmasks or lists)
2. Fitness: |covered| - overlap_penalty - packing_penalty
3. Crossover: spatial partitioning (take centers from different regions)
4. Mutation: add/remove/swap centers
5. Selection: tournament selection with elitism

Key insight: Start from ball-14 vertices (most constrained)
Force them to be covered first, then fill in the rest.

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import random
import time
import sys

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

class GeneticPartition:
    def __init__(self, n=15, s=10):
        self.n = n
        self.s = s

        print(f"Initializing Lambda_{n}(1^{s})...")
        self.verts = [v for v in range(1 << n) if not has_circ(v, n, s)]
        self.vset = set(self.verts)
        self.num_verts = len(self.verts)
        self.vert_to_idx = {v: i for i, v in enumerate(self.verts)}
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

        # Group by ball size
        self.by_ball_size = defaultdict(list)
        for v in self.verts:
            self.by_ball_size[self.ball_sizes[v]].append(v)

        print(f"Ball sizes: {dict(Counter(self.ball_sizes.values()))}")

        # Precompute close pairs (d < 3) for fast packing check
        print("Precomputing close pairs...")
        self.close_to = defaultdict(set)
        for v in self.verts:
            for i in range(n):
                nb = v ^ (1 << i)
                if nb in self.vset:
                    self.close_to[v].add(nb)
            for i in range(n):
                for j in range(i + 1, n):
                    nb = v ^ (1 << i) ^ (1 << j)
                    if nb in self.vset:
                        self.close_to[v].add(nb)

        # Priority order for covering: small ball size first
        self.priority_order = sorted(self.verts, key=lambda v: (self.ball_sizes[v], v))

        print("Initialization complete.")

    def evaluate(self, centers):
        """Evaluate fitness of a center set."""
        center_set = set(centers)

        # Check packing violations
        packing_violations = 0
        center_list = list(centers)
        for i, c1 in enumerate(center_list):
            for c2 in center_list[i+1:]:
                if c2 in self.close_to[c1]:
                    packing_violations += 1

        # Check coverage
        coverage = defaultdict(int)
        for c in centers:
            for v in self.balls[c]:
                coverage[v] += 1

        covered = len([v for v in self.verts if coverage[v] >= 1])
        overlaps = sum(coverage[v] - 1 for v in self.verts if coverage[v] > 1)
        uncovered = self.num_verts - covered

        # Fitness: maximize coverage, penalize violations
        # Perfect partition: covered = num_verts, overlaps = 0, packing_violations = 0
        fitness = covered - overlaps * 10 - packing_violations * 100

        return fitness, covered, overlaps, packing_violations, uncovered

    def greedy_individual(self, start_idx=0, prioritize_small=True):
        """Generate an individual using greedy heuristic."""
        centers = []
        center_set = set()
        covered = set()
        blocked = set()  # Vertices at distance < 3 from any center

        # Choose starting point
        if prioritize_small:
            candidates = self.priority_order.copy()
        else:
            candidates = random.sample(self.verts, len(self.verts))

        # Rotate based on start_idx for diversity
        candidates = candidates[start_idx:] + candidates[:start_idx]

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
            blocked.update(self.close_to[c])
            blocked.add(c)

        return centers

    def random_individual(self, target_centers=2050):
        """Generate a random individual."""
        centers = []
        center_set = set()
        blocked = set()

        candidates = random.sample(self.verts, len(self.verts))

        for c in candidates:
            if c in blocked:
                continue

            centers.append(c)
            center_set.add(c)
            blocked.update(self.close_to[c])
            blocked.add(c)

            if len(centers) >= target_centers:
                break

        return centers

    def crossover(self, parent1, parent2):
        """Crossover two parents by spatial partitioning."""
        # Random bit position to split by
        bit_pos = random.randint(0, self.n - 1)

        # Take centers from parent1 where bit is 0, from parent2 where bit is 1
        child_centers = []
        used = set()
        blocked = set()

        for c in parent1:
            if (c >> bit_pos) & 1 == 0:
                if c not in blocked:
                    child_centers.append(c)
                    used.add(c)
                    blocked.update(self.close_to[c])
                    blocked.add(c)

        for c in parent2:
            if (c >> bit_pos) & 1 == 1:
                if c not in blocked:
                    child_centers.append(c)
                    used.add(c)
                    blocked.update(self.close_to[c])
                    blocked.add(c)

        return child_centers

    def mutate(self, centers, mutation_rate=0.1):
        """Mutate an individual."""
        centers = list(centers)
        center_set = set(centers)

        # Compute current state
        covered = set()
        for c in centers:
            covered.update(self.balls[c])

        uncovered = [v for v in self.verts if v not in covered]

        # Mutation: add centers to cover uncovered
        if uncovered and random.random() < mutation_rate:
            # Pick an uncovered vertex
            target = random.choice(uncovered)

            # Find candidates that can cover it
            candidates = [c for c in self.membership[target]
                          if c not in center_set and
                          not any(existing in self.close_to[c] for existing in center_set)]

            if candidates:
                new_center = random.choice(candidates)
                centers.append(new_center)
                center_set.add(new_center)

        # Mutation: remove a random center
        if len(centers) > 1 and random.random() < mutation_rate * 0.5:
            to_remove = random.choice(centers)
            centers.remove(to_remove)
            center_set.remove(to_remove)

        # Mutation: swap a center for a nearby alternative
        if centers and random.random() < mutation_rate:
            old_center = random.choice(centers)
            centers.remove(old_center)
            center_set.remove(old_center)

            # Find alternatives at distance 3 (valid packing distance)
            alternatives = []
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    for k in range(j + 1, self.n):
                        alt = old_center ^ (1 << i) ^ (1 << j) ^ (1 << k)
                        if alt in self.vset and alt not in center_set:
                            if not any(existing in self.close_to[alt] for existing in center_set):
                                alternatives.append(alt)

            if alternatives:
                new_center = random.choice(alternatives[:10])  # Random from top 10
                centers.append(new_center)
                center_set.add(new_center)
            else:
                # Revert
                centers.append(old_center)

        return centers

    def tournament_select(self, population, fitnesses, tournament_size=3):
        """Tournament selection."""
        indices = random.sample(range(len(population)), tournament_size)
        winner = max(indices, key=lambda i: fitnesses[i])
        return population[winner]

    def run(self, pop_size=50, generations=1000, elite_size=5):
        """Run genetic algorithm."""
        print(f"\nStarting Genetic Algorithm (pop={pop_size}, gen={generations})...")

        # Initialize population
        print("Initializing population...")
        population = []

        # Add greedy individuals with different starting points
        for i in range(pop_size // 2):
            start_idx = (i * len(self.verts)) // (pop_size // 2)
            ind = self.greedy_individual(start_idx=start_idx)
            population.append(ind)

        # Add random individuals
        for _ in range(pop_size - len(population)):
            ind = self.random_individual()
            population.append(ind)

        # Evaluate initial population
        fitnesses = []
        stats = []
        for ind in population:
            f, cov, ovl, pack, unc = self.evaluate(ind)
            fitnesses.append(f)
            stats.append((cov, ovl, pack, unc))

        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]
        best_stats = stats[best_idx]

        print(f"Initial best: fitness={best_fitness}, covered={best_stats[0]}/{self.num_verts}, "
              f"overlaps={best_stats[1]}, packing_violations={best_stats[2]}")

        start_time = time.time()

        for gen in range(generations):
            # Create new population
            new_population = []

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate rest through crossover and mutation
            while len(new_population) < pop_size:
                parent1 = self.tournament_select(population, fitnesses)
                parent2 = self.tournament_select(population, fitnesses)

                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutation_rate=0.2)

                new_population.append(child)

            population = new_population

            # Evaluate
            fitnesses = []
            stats = []
            for ind in population:
                f, cov, ovl, pack, unc = self.evaluate(ind)
                fitnesses.append(f)
                stats.append((cov, ovl, pack, unc))

            gen_best_idx = np.argmax(fitnesses)
            gen_best = fitnesses[gen_best_idx]
            gen_stats = stats[gen_best_idx]

            if gen_best > best_fitness:
                best_fitness = gen_best
                best_individual = population[gen_best_idx]
                best_stats = gen_stats

            # Progress report
            if gen % 50 == 0 or gen_stats[3] == 0:  # or perfect solution
                elapsed = time.time() - start_time
                print(f"Gen {gen}: best_fitness={best_fitness}, covered={best_stats[0]}/{self.num_verts}, "
                      f"uncovered={best_stats[3]}, overlaps={best_stats[1]}, packing={best_stats[2]}, "
                      f"time={elapsed:.1f}s")

                # Check for perfect partition
                if best_stats[0] == self.num_verts and best_stats[1] == 0 and best_stats[2] == 0:
                    print("\n*** PERFECT PARTITION FOUND! ***")
                    return best_individual, True

        print(f"\nGA finished. Best: {len(best_individual)} centers, "
              f"covered={best_stats[0]}, uncovered={best_stats[3]}, "
              f"overlaps={best_stats[1]}, packing={best_stats[2]}")

        return best_individual, False

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
        coverage = defaultdict(int)
        for c in centers:
            for v in self.balls[c]:
                coverage[v] += 1

        uncovered = [v for v in self.verts if coverage[v] == 0]
        overlaps = [v for v in self.verts if coverage[v] > 1]

        if uncovered:
            return False, f"{len(uncovered)} uncovered vertices"

        if overlaps:
            return False, f"{len(overlaps)} overlapping vertices"

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
    random.seed(int(time.time()))
    np.random.seed(int(time.time()) % (2**32))

    ga = GeneticPartition(n=15, s=10)

    # Run multiple times with different seeds
    for run in range(3):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}")
        print("=" * 60)

        solution, found = ga.run(pop_size=100, generations=500, elite_size=10)

        if found:
            ok, msg = ga.verify(solution)
            print(f"\nVerification: {msg}")

            if ok:
                ga.save_solution(solution,
                    f"/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_ga_run{run}")
                print("SOLUTION FOUND! Exiting...")
                return

        # Random seed for next run
        random.seed(int(time.time() * 1000) % (2**31))

    print("\nNo perfect partition found in any run.")


if __name__ == "__main__":
    main()
