# Analysis: Perfect Partition for Λ₃₁(1²⁸)

**Date**: 2025-11-26
**Author**: Jae-Hyun Baek
**Status**: Open Problem Identified

---

## Executive Summary

The perfect partition construction for n=31, s=28 is **more complex** than initially anticipated. Simple "Hamming minus bad" approaches fail due to covering issues.

---

## Key Findings

### 1. Bad Codewords in Ham(31,26)

Only **3 bad codewords** out of 67,108,864:

| Codeword | Weight | Pattern |
|----------|--------|---------|
| `0011111111111111111111111111110` | 28 | 28 consecutive 1s at positions 1-28 |
| `1111111111111111111111111111000` | 28 | 28 consecutive 1s at positions 3-30 |
| `1111111111111111111111111111111` | 31 | All ones |

### 2. Coverage Problem

Removing the 3 bad codewords leaves **56 allowed vertices uncovered**:
- 28 neighbors of bad1 (weight 27)
- 28 neighbors of bad2 (weight 27)
- 0 neighbors of all_ones (all weight-30 neighbors are forbidden)

### 3. Packing Constraint Violation

The 56 uncovered vertices have **minimum pairwise distance = 2**, so they cannot ALL be centers simultaneously.

### 4. Center Conflicts

Each uncovered vertex v:
- Has syndrome s > 0 (not a Hamming codeword)
- Its only nearby Hamming codeword is at distance 1 (a bad codeword)
- Any alternative center at distance 1 is at distance 1 from a GOOD Hamming codeword (conflict!)

---

## Why Simple Constructions Fail

### Approach 1: Ham(31,26) - {bad}
- **Fails**: 56 vertices uncovered

### Approach 2: Ham - {bad} + {uncovered as centers}
- **Fails**: Uncovered vertices at pairwise d=2

### Approach 3: Shift bad codewords
- Bad codewords can shift to valid positions
- But shifted positions are at d=1 from good Hamming codewords
- **Fails**: Packing violation with existing centers

---

## Implications

### For n=31, s=28

The construction requires **global reorganization** like n=15, s=12:
- Many good Hamming codewords must shift
- Not just 3 bad codewords, but their neighborhoods
- The SAT approach would work but scale is prohibitive (67M codewords)

### Pattern Analysis

For s = n - 3:
- n=7, s=4: Works with simple shifts (SAT verified)
- n=15, s=12: Requires ~67% shifts (SAT verified)
- n=31, s=28: Expected to require significant shifts

---

## Construction Strategy for n=31

### Option 1: Streaming Greedy (Current)
- Process codewords in order
- Shift when needed
- May leave gaps in coverage

### Option 2: Local SAT Repair
- Use greedy for most codewords
- Apply SAT solver for problematic neighborhoods
- Incremental conflict resolution

### Option 3: Theoretical Proof
- Show existence via probabilistic argument
- Lovász Local Lemma style proof
- Don't need explicit construction

---

## Open Questions

1. What is the minimum number of shifts needed for n=31, s=28?
2. Is there a closed-form construction based on syndrome patterns?
3. Can we prove existence without explicit construction?

---

## Files

- `construct_n31_s28.py`: Streaming greedy construction
- `analyze_n31_bad_codewords.py`: Bad codeword enumeration
- `N31_S28_ANALYSIS.md`: This analysis document

---

*Last updated: 2025-11-26*
