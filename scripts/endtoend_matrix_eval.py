#!/usr/bin/env python3
"""Shared evaluator for the end-to-end matrix-evolution baselines.

The evolved artifact is a full program that must define
``build_generator() -> 10x43 binary matrix`` (any full-rank form). Programs are
executed in an isolated subprocess with a wall-clock limit, and the returned
matrix is verified deterministically in the parent: GF(2) rank, exact d_min and
weight distribution over all 1023 nonzero codewords.

Staged objective (minimized):
  invalid program / wrong shape / rank < 10  -> 1e9
  d_min < 16 -> (16 - d_min) * 100_000 + (# codewords of weight < 16)
  d_min >= 16 -> A_16   (0 would simultaneously prove a [43,10,17] code)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

N, K, TARGET_D = 43, 10, 16
INVALID = 1e9

HARNESS = '''
import json, sys
class _FS:
    @staticmethod
    def run(f):
        return f
    @staticmethod
    def evolve(f):
        return f
funsearch = _FS()
namespace = {"funsearch": _FS()}
exec(compile(open(sys.argv[1]).read(), "candidate", "exec"), namespace)
matrix = namespace["build_generator"]()
print(json.dumps([[int(v) & 1 for v in row] for row in matrix]))
'''


def gf2_rank(matrix: np.ndarray) -> int:
    m = matrix.copy().astype(np.uint8)
    rank = 0
    for col in range(m.shape[1]):
        pivot = None
        for row in range(rank, m.shape[0]):
            if m[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(m.shape[0]):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
    return rank


def codeword_weights(matrix: np.ndarray) -> np.ndarray:
    messages = np.arange(1, 1 << K, dtype=np.uint32)
    bits = ((messages[:, None] >> np.arange(K)[None, :]) & 1).astype(np.uint8)
    return (bits @ matrix % 2).sum(axis=1)


def run_candidate_program(code: str, *, timeout_sec: int = 90) -> dict:
    """Execute the candidate in a subprocess and verify the emitted matrix."""
    record: dict = {"valid": False, "objective": INVALID}
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "candidate.py"
        candidate.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", HARNESS, str(candidate)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            record["error"] = "timeout"
            return record
    if proc.returncode != 0:
        record["error"] = (proc.stderr or "nonzero exit").strip()[-300:]
        return record
    try:
        matrix = np.array(json.loads(proc.stdout.strip().splitlines()[-1]), dtype=np.uint8)
    except Exception as exc:
        record["error"] = f"unparseable output: {exc}"
        return record
    if matrix.shape != (K, N):
        record["error"] = f"shape {matrix.shape} != ({K},{N})"
        return record
    if gf2_rank(matrix) != K:
        record["error"] = "rank deficient"
        return record
    weights = codeword_weights(matrix)
    d_min = int(weights.min())
    a16 = int((weights == TARGET_D).sum())
    below = int((weights < TARGET_D).sum())
    objective = float((TARGET_D - d_min) * 100_000 + below) if d_min < TARGET_D else float(a16)
    record.update(
        {
            "valid": True,
            "d_min": d_min,
            "A16": a16,
            "codewords_below_16": below,
            "objective": objective,
            "matrix": matrix.astype(int).tolist(),
        }
    )
    return record


TASK_DESCRIPTION = (
    "Construct a binary linear [43,10] error-correcting code with minimum distance "
    "at least 16 and as few weight-16 codewords (A16) as possible. Implement "
    "build_generator() returning a 10x43 matrix of 0/1 integers whose rows are "
    "linearly independent over GF(2). The deterministic evaluator enumerates all "
    "1023 nonzero codewords: while d_min < 16 the objective is "
    "(16 - d_min)*100000 + (#codewords of weight < 16); once d_min >= 16 the "
    "objective is A16 (lower is better; the best known value is 225 and any "
    "improvement is significant). The function must be deterministic and finish "
    "within 90 seconds; numpy, math, and itertools are available. Known algebraic "
    "constructions (BCH, quasi-cyclic, shortening/puncturing of larger codes) are "
    "legitimate strategies."
)

SEED_BODY_LINES = [
    "# Systematic seed [I_10 | P] with a fixed pseudo-random parity part.",
    "state = 88172645463325252",
    "bits = []",
    "while len(bits) < 10 * 33:",
    "    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF",
    "    state ^= state >> 7",
    "    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF",
    "    bits.append(state & 1)",
    "rows = []",
    "for i in range(10):",
    "    identity = [1 if j == i else 0 for j in range(10)]",
    "    parity = bits[i * 33:(i + 1) * 33]",
    "    rows.append(identity + parity)",
    "return rows",
]
