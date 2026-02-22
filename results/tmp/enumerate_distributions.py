#!/usr/bin/env python3
"""
Enumerate all feasible (a, b, d, C) distributions for Lambda_15(1^10) perfect partition.

Based on ChatGPT's ILP analysis:
- a: number of ball-14 centers
- b: number of ball-15 centers
- d: number of ball-16 centers
- C = a + b + d: total centers

Constraints:
1. 14a + 15b + 16d = 32527 (exact cover)
2. 0 <= a <= 435 (supply of ball-14 vertices)
3. 0 <= b <= 1305 (supply of ball-15 vertices)
4. 0 <= d <= 30787 (supply of ball-16 vertices)

Key insight: 2a + b = D(C) where D(C) = 16C - 32527

Author: Jae-Hyun Baek (with Claude Code)
Date: 2025-11-27
"""

from collections import defaultdict
import sys

# Constants
V_TOTAL = 32527
N14 = 435   # number of ball-14 vertices
N15 = 1305  # number of ball-15 vertices
N16 = 30787 # number of ball-16 vertices

def enumerate_all_distributions():
    """Enumerate all feasible (C, a, b, d) combinations."""
    solutions = []

    # For each possible a (ball-14 centers)
    for a in range(N14 + 1):
        # For each possible b (ball-15 centers)
        for b in range(N15 + 1):
            # Calculate remainder that must be covered by ball-16
            rem = V_TOTAL - 14 * a - 15 * b

            if rem < 0:
                break  # No point continuing with larger b

            if rem % 16 != 0:
                continue  # d must be integer

            d = rem // 16

            if d < 0 or d > N16:
                continue

            C = a + b + d
            solutions.append((C, a, b, d))

    return solutions

def analyze_distributions(solutions):
    """Analyze and summarize the distributions."""

    # Group by C
    by_C = defaultdict(list)
    for C, a, b, d in solutions:
        by_C[C].append((a, b, d))

    min_C = min(by_C.keys())
    max_C = max(by_C.keys())

    print("=" * 70)
    print("Lambda_15(1^10) Perfect Partition - Feasible Type Distributions")
    print("=" * 70)
    print(f"\n|V| = {V_TOTAL}")
    print(f"Ball-14 vertices (supply): {N14}")
    print(f"Ball-15 vertices (supply): {N15}")
    print(f"Ball-16 vertices (supply): {N16}")
    print(f"\nTotal feasible (C, a, b, d) combinations: {len(solutions)}")
    print(f"Center count range: {min_C} <= C <= {max_C}")
    print(f"Number of distinct C values: {len(by_C)}")

    # Summary by C
    print("\n" + "=" * 70)
    print("Summary by Center Count C")
    print("=" * 70)
    print(f"{'C':>6} {'D=16C-V':>8} {'#combos':>8} {'a_range':>12} {'b_range':>12} {'d_range':>12}")
    print("-" * 70)

    for C in sorted(by_C.keys()):
        combos = by_C[C]
        D = 16 * C - V_TOTAL
        a_vals = [x[0] for x in combos]
        b_vals = [x[1] for x in combos]
        d_vals = [x[2] for x in combos]

        a_range = f"[{min(a_vals)},{max(a_vals)}]"
        b_range = f"[{min(b_vals)},{max(b_vals)}]"
        d_range = f"[{min(d_vals)},{max(d_vals)}]"

        print(f"{C:>6} {D:>8} {len(combos):>8} {a_range:>12} {b_range:>12} {d_range:>12}")

    # Detailed view for small C values
    print("\n" + "=" * 70)
    print("Detailed Distributions for C <= 2050")
    print("=" * 70)

    for C in sorted(by_C.keys()):
        if C > 2050:
            break

        combos = by_C[C]
        D = 16 * C - V_TOTAL

        print(f"\nC = {C} (D = {D}, deficit equation: 2a + b = {D})")
        print(f"  {len(combos)} combinations:")

        for a, b, d in sorted(combos):
            # Verify: 2a + b should equal D
            assert 2*a + b == D, f"Verification failed: 2*{a} + {b} != {D}"
            print(f"    (a={a:>3}, b={b:>4}, d={d:>4}) -> 14×{a} + 15×{b} + 16×{d} = {14*a + 15*b + 16*d}")

    # Special analysis: which distributions use no ball-14 centers?
    print("\n" + "=" * 70)
    print("Distributions with a=0 (no ball-14 centers)")
    print("=" * 70)

    a0_solutions = [(C, a, b, d) for C, a, b, d in solutions if a == 0]
    print(f"Total: {len(a0_solutions)} combinations")
    print(f"C range when a=0: [{min(C for C,_,_,_ in a0_solutions)}, {max(C for C,_,_,_ in a0_solutions)}]")

    print("\nFirst 20 (a=0) distributions:")
    for C, a, b, d in sorted(a0_solutions)[:20]:
        print(f"  C={C}: (a=0, b={b}, d={d})")

    # Special analysis: distributions that maximize ball-14 usage
    print("\n" + "=" * 70)
    print("Distributions with maximum ball-14 usage (a=435)")
    print("=" * 70)

    max_a_solutions = [(C, a, b, d) for C, a, b, d in solutions if a == N14]
    if max_a_solutions:
        print(f"Total: {len(max_a_solutions)} combinations")
        print(f"C range when a={N14}: [{min(C for C,_,_,_ in max_a_solutions)}, {max(C for C,_,_,_ in max_a_solutions)}]")

        print(f"\nAll (a={N14}) distributions:")
        for C, a, b, d in sorted(max_a_solutions):
            print(f"  C={C}: (a={a}, b={b}, d={d})")
    else:
        print("No distributions with a=435")

    return by_C

def save_for_search(solutions, filename_prefix):
    """Save promising distributions for SAT/local search."""

    # Save distributions for C <= 2050 (likely candidates)
    promising = [(C, a, b, d) for C, a, b, d in solutions if C <= 2050]

    with open(f"{filename_prefix}_C_le_2050.txt", 'w') as f:
        f.write("# Lambda_15(1^10) feasible distributions with C <= 2050\n")
        f.write("# Format: C, a (ball-14), b (ball-15), d (ball-16)\n")
        f.write(f"# Total: {len(promising)} combinations\n\n")

        for C, a, b, d in sorted(promising):
            f.write(f"{C},{a},{b},{d}\n")

    print(f"\nSaved {len(promising)} distributions to {filename_prefix}_C_le_2050.txt")

    # Save all distributions
    with open(f"{filename_prefix}_all.txt", 'w') as f:
        f.write("# Lambda_15(1^10) all feasible distributions\n")
        f.write("# Format: C, a (ball-14), b (ball-15), d (ball-16)\n")
        f.write(f"# Total: {len(solutions)} combinations\n\n")

        for C, a, b, d in sorted(solutions):
            f.write(f"{C},{a},{b},{d}\n")

    print(f"Saved {len(solutions)} distributions to {filename_prefix}_all.txt")

def main():
    print("Enumerating all feasible (C, a, b, d) distributions...")
    solutions = enumerate_all_distributions()

    by_C = analyze_distributions(solutions)

    save_for_search(solutions, "/Users/baegjaehyeon/CodeEvolve/results/tmp/distributions_15_10")

    # Key insight summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. If C = 2033 (minimum centers), then (a,b,d) = (0,1,2032) is the ONLY option.
   - All ball-14 vertices must be covered as neighbors, not centers
   - Exactly ONE ball-15 center, rest are ball-16

2. As C increases, more flexibility in (a,b,d) distribution:
   - C = 2034: 9 combinations
   - C = 2035: 17 combinations
   - ...

3. The constraint 2a + b = 16C - 32527 determines which combinations work.

4. For practical search, focus on:
   - C = 2033: unique distribution, try direct SAT
   - C = 2034-2040: few combinations, enumerate all
   - Larger C: use heuristics to select promising distributions
""")

if __name__ == "__main__":
    main()
