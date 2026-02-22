#!/usr/bin/env python3
"""
Constraint Propagation for Lambda_15(1^10) Perfect Partition.

Key insight: Ball size 14 vertices are the bottleneck.
Each ball-14 vertex can only be covered by 14 possible centers.
If we fix some centers, propagate constraints to reduce choices.

Strategy:
1. Start with ball-14 vertices (most constrained)
2. Use arc consistency to prune impossible assignments
3. Backtrack when stuck

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

import numpy as np
from collections import defaultdict, Counter
import random
import time
import copy

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

class ConstraintPropagation:
    def __init__(self, n=15, s=10):
        self.n = n
        self.s = s

        print(f"Initializing Lambda_{n}(1^{s})...")
        self.verts = [v for v in range(1 << n) if not has_circ(v, n, s)]
        self.vset = set(self.verts)
        self.num_verts = len(self.verts)
        print(f"|V| = {self.num_verts}")

        # Precompute balls
        print("Precomputing balls...")
        self.balls = {}
        self.ball_sizes = {}
        self.membership = defaultdict(list)

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

    def propagate(self, domains, selected, blocked):
        """
        Propagate constraints.
        domains[v] = set of possible centers that can cover v
        selected = set of centers already chosen
        blocked = set of vertices that can't be centers (too close to selected)

        Returns True if consistent, False if contradiction
        """
        changed = True
        while changed:
            changed = False

            for v in self.verts:
                if v in domains and len(domains[v]) == 0:
                    return False  # No way to cover v

                # Remove blocked centers from domain
                if v in domains:
                    new_domain = domains[v] - blocked
                    if len(new_domain) < len(domains[v]):
                        domains[v] = new_domain
                        changed = True

                    # If domain has only one element, it must be selected
                    if len(domains[v]) == 1:
                        forced_center = list(domains[v])[0]
                        if forced_center not in selected:
                            # Check if we can select it
                            if forced_center in blocked:
                                return False  # Contradiction

                            # Select this center
                            selected.add(forced_center)
                            changed = True

                            # Block close vertices
                            for i in range(self.n):
                                nb = forced_center ^ (1 << i)
                                if nb in self.vset:
                                    blocked.add(nb)
                            for i in range(self.n):
                                for j in range(i + 1, self.n):
                                    nb = forced_center ^ (1 << i) ^ (1 << j)
                                    if nb in self.vset:
                                        blocked.add(nb)

                            # Mark all vertices in this ball as covered
                            for u in self.balls[forced_center]:
                                if u in domains:
                                    domains[u] = {forced_center}

        return True

    def solve_with_backtrack(self, max_backtracks=100000):
        """Main solving routine with backtracking."""
        print("\nStarting constraint propagation with backtracking...")

        # Initial domains: each vertex can be covered by any center in its membership
        initial_domains = {}
        for v in self.verts:
            initial_domains[v] = set(self.membership[v])

        # Stack for backtracking: (domains, selected, blocked, next_choice)
        stack = [(copy.deepcopy(initial_domains), set(), set(), None)]

        backtracks = 0
        best_coverage = 0
        iterations = 0

        start_time = time.time()

        while stack and backtracks < max_backtracks:
            iterations += 1

            if iterations % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Iter {iterations}: stack={len(stack)}, backtracks={backtracks}, "
                      f"best_coverage={best_coverage}/{self.num_verts}, time={elapsed:.1f}s")

            domains, selected, blocked, choice = stack.pop()

            # If there's a choice to make, make it
            if choice is not None:
                v, center = choice
                if center in blocked:
                    backtracks += 1
                    continue

                # Select this center
                selected.add(center)

                # Update blocked
                for i in range(self.n):
                    nb = center ^ (1 << i)
                    if nb in self.vset:
                        blocked.add(nb)
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        nb = center ^ (1 << i) ^ (1 << j)
                        if nb in self.vset:
                            blocked.add(nb)

                # Mark covered
                for u in self.balls[center]:
                    domains[u] = {center}

            # Propagate
            if not self.propagate(domains, selected, blocked):
                backtracks += 1
                continue

            # Check if solved
            covered = set()
            for c in selected:
                covered.update(self.balls[c])

            if len(covered) > best_coverage:
                best_coverage = len(covered)
                print(f"  New best: {best_coverage}/{self.num_verts} with {len(selected)} centers")

            if covered == self.vset:
                print(f"\nSOLUTION FOUND with {len(selected)} centers!")
                return list(selected), True

            # Find next branching variable (MRV heuristic)
            uncovered = [v for v in self.verts if v not in covered]
            if not uncovered:
                continue

            # Choose the uncovered vertex with smallest remaining domain
            mrv_vertex = min(uncovered, key=lambda v: len(domains.get(v, set())))
            mrv_domain = domains.get(mrv_vertex, set())

            if not mrv_domain:
                backtracks += 1
                continue

            # Branch on each possible center for this vertex
            for center in sorted(mrv_domain, key=lambda c: -self.ball_sizes[c]):
                # Push choice to stack
                new_domains = copy.deepcopy(domains)
                new_selected = selected.copy()
                new_blocked = blocked.copy()
                stack.append((new_domains, new_selected, new_blocked, (mrv_vertex, center)))

        print(f"\nSearch ended. Backtracks: {backtracks}, Best coverage: {best_coverage}")
        return None, False

    def verify(self, centers):
        center_set = set(centers)

        for c in centers:
            if c not in self.vset:
                return False, f"Invalid center {c}"

        for i, c1 in enumerate(centers):
            for c2 in centers[i+1:]:
                if hamming(c1, c2) < 3:
                    return False, f"Centers too close"

        covered = set()
        for c in centers:
            for v in self.balls[c]:
                if v in covered:
                    return False, "Overlap"
                covered.add(v)

        if covered != self.vset:
            return False, f"Missing {len(self.vset - covered)} vertices"

        return True, "OK"


def main():
    random.seed(42)

    cp = ConstraintPropagation(n=15, s=10)
    solution, found = cp.solve_with_backtrack(max_backtracks=100000)

    if found:
        ok, msg = cp.verify(solution)
        print(f"\nVerification: {msg}")

        if ok:
            np.save("/Users/baegjaehyeon/CodeEvolve/results/tmp/lambda_15_10_cp_centers.npy",
                    np.array(solution, dtype=np.uint32))
            print("Solution saved!")


if __name__ == "__main__":
    main()
