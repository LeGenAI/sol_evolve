#!/usr/bin/env python3
"""Reconstruct and verify the binary (GF(2)) self-orthogonal embeddings of the SO table.

Instances:
- 22_11: append t=7 columns S to the [15,11,3] Hamming code H4 so that
  [H4 | S] is self-orthogonal with d_min >= 6 (SAT), and check the d_min >= 7
  obligation for the same fixed base (expected UNSAT).
- 52_26: append t=21 columns S to the [31,26,3] Hamming code H5 so that
  [H5 | S] is self-orthogonal (SAT on the Gram condition alone; distance
  constraints are out of reach at k=26), enumerate several solutions with
  blocking clauses, and measure the exact d_min of each via a
  meet-in-the-middle enumeration of all 2^26 codewords.

All verification (self-orthogonality, minimum distance, weight distribution)
is recomputed independently with numpy after solving. Deterministic; no LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.engines.sat_solver_interface import SATSolver  # noqa: E402
from solevolve.so_claim_repro import resolve_cadical  # noqa: E402


def hamming_generator(r: int) -> np.ndarray:
    """Systematic generator [I_k | A] of the [2^r-1, 2^r-1-r, 3] Hamming code."""
    columns = [np.array([(v >> bit) & 1 for bit in range(r)], dtype=np.uint8) for v in range(1, 2**r)]
    weight_ge2 = [c for c in columns if int(c.sum()) >= 2]
    k = 2**r - 1 - r
    assert len(weight_ge2) == k
    a = np.stack(weight_ge2)  # k x r
    return np.concatenate([np.eye(k, dtype=np.uint8), a], axis=1)


class CNFBuilder:
    def __init__(self) -> None:
        self.counter = 0
        self.clauses: list[list[int]] = []

    def new_var(self) -> int:
        self.counter += 1
        return self.counter

    def add(self, clause: list[int]) -> None:
        self.clauses.append(clause)

    def and_var(self, a: int, b: int) -> int:
        out = self.new_var()
        self.add([-out, a])
        self.add([-out, b])
        self.add([out, -a, -b])
        return out

    def xor_chain(self, lits: list[int], constant: int) -> int | None:
        """Return an aux var equal to XOR(lits) XOR constant; None for empty."""
        if not lits:
            return None
        acc = lits[0]
        for lit in lits[1:]:
            out = self.new_var()
            # out <-> acc XOR lit
            self.add([-out, acc, lit])
            self.add([-out, -acc, -lit])
            self.add([out, -acc, lit])
            self.add([out, acc, -lit])
            acc = out
        if constant:
            out = self.new_var()
            self.add([-out, -acc])
            self.add([out, acc])
            acc = out
        return acc

    def assert_xor(self, lits: list[int], parity: int) -> None:
        """Assert XOR(lits) == parity via a chain and a unit clause."""
        acc = self.xor_chain(lits, 0)
        if acc is None:
            if parity:
                self.add([])  # unsatisfiable
            return
        self.add([acc] if parity else [-acc])

    def at_least(self, lits: list[int], threshold: int) -> None:
        if threshold <= 0:
            return
        drop = len(lits) - threshold + 1
        for subset in combinations(lits, drop):
            self.add(list(subset))


def encode_gram(builder: CNFBuilder, s_var, k: int, t: int, gram: np.ndarray) -> None:
    """Assert S S^T == gram over GF(2), with S the k x t variable matrix."""
    for i in range(k):
        builder.assert_xor([s_var(i, r) for r in range(t)], int(gram[i, i]))
        for j in range(i + 1, k):
            products = [builder.and_var(s_var(i, r), s_var(j, r)) for r in range(t)]
            builder.assert_xor(products, int(gram[i, j]))


def encode_distance(builder: CNFBuilder, s_var, base: np.ndarray, k: int, t: int, target_d: int) -> None:
    """Assert weight([m G_base | m S]) >= target_d for every nonzero message m."""
    for message in range(1, 1 << k):
        rows = [i for i in range(k) if message & (1 << i)]
        base_word = np.zeros(base.shape[1], dtype=np.uint8)
        for i in rows:
            base_word ^= base[i]
        need = target_d - int(base_word.sum())
        if need <= 0:
            continue
        ys = []
        for r in range(t):
            y = builder.xor_chain([s_var(i, r) for i in rows], 0)
            ys.append(y)
        builder.at_least(ys, need)


def solve(builder: CNFBuilder, cnf_path: Path, cadical: str, timeout: int) -> dict:
    with cnf_path.open("w", encoding="utf-8") as handle:
        handle.write(f"p cnf {builder.counter} {len(builder.clauses)}\n")
        for clause in builder.clauses:
            handle.write(" ".join(map(str, clause)) + " 0\n")
    solver = SATSolver(solver_type="cadical", solver_path=cadical)
    return solver.solve(str(cnf_path), timeout=timeout, verbose=False)


def extract_s(model, k: int, t: int) -> np.ndarray:
    if isinstance(model, dict):
        assignment = {int(var): bool(value) for var, value in model.items()}
    else:
        assignment = {abs(lit): lit > 0 for lit in model}
    return np.array([[1 if assignment.get(i * t + r + 1, False) else 0 for r in range(t)] for i in range(k)], dtype=np.uint8)


def dmin_meet_in_middle(matrix: np.ndarray) -> tuple[int, dict[int, int]]:
    """Exact d_min and weight distribution via split enumeration (uint64 packing)."""
    k, n = matrix.shape
    assert n <= 64
    row_words = np.zeros(k, dtype=np.uint64)
    for i in range(k):
        word = 0
        for j in range(n):
            word |= int(matrix[i, j]) << j
        row_words[i] = word
    half = k // 2
    def span(rows: np.ndarray) -> np.ndarray:
        words = np.zeros(1 << len(rows), dtype=np.uint64)
        for idx, row in enumerate(rows):
            step = 1 << idx
            words[step : 2 * step] = words[:step] ^ row
        return words
    left = span(row_words[:half])
    right = span(row_words[half:])
    counts: dict[int, int] = {}
    for r_word in right:
        weights = np.bitwise_count(left ^ r_word)
        values, tallies = np.unique(weights, return_counts=True)
        for value, tally in zip(values.tolist(), tallies.tolist()):
            counts[int(value)] = counts.get(int(value), 0) + int(tally)
    nonzero = [w for w in counts if w > 0]
    return min(nonzero), dict(sorted(counts.items()))


def verify(base: np.ndarray, s: np.ndarray) -> dict:
    ext = np.concatenate([base, s], axis=1).astype(np.uint8)
    gram = (ext @ ext.T) % 2
    d_min, distribution = dmin_meet_in_middle(ext)
    return {
        "self_orthogonal": bool(not gram.any()),
        "d_min": int(d_min),
        "weight_distribution": distribution,
        "n": int(ext.shape[1]),
        "k": int(ext.shape[0]),
        "matrix": ext.astype(int).tolist(),
    }


def run_22_11(out_dir: Path, cadical: str) -> dict:
    base = hamming_generator(4)  # [15,11]
    k, t = 11, 7
    gram = (base @ base.T) % 2
    s_var = lambda i, r: i * t + r + 1  # noqa: E731
    report: dict = {"instance": "22_11", "base": "[15,11,3] Hamming H4", "t": t}

    builder = CNFBuilder()
    builder.counter = k * t
    encode_gram(builder, s_var, k, t, gram)
    encode_distance(builder, s_var, base, k, t, target_d=6)
    started = time.perf_counter()
    result = solve(builder, out_dir / "so_22_11_d6.cnf", cadical, timeout=600)
    report["d6_status"] = result.get("status")
    report["d6_solve_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if result.get("status") == "SAT" and result.get("model"):
        s = extract_s(result["model"], k, t)
        report["d6_verification"] = verify(base, s)

    builder = CNFBuilder()
    builder.counter = k * t
    encode_gram(builder, s_var, k, t, gram)
    encode_distance(builder, s_var, base, k, t, target_d=7)
    started = time.perf_counter()
    result = solve(builder, out_dir / "so_22_11_d7.cnf", cadical, timeout=1800)
    report["d7_obligation_status"] = result.get("status")
    report["d7_solve_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return report


def run_52_26(out_dir: Path, cadical: str, *, solutions: int) -> dict:
    base = hamming_generator(5)  # [31,26]
    k = 26
    gram = (base @ base.T) % 2
    t = gf2_rank(gram)
    s_var = lambda i, r: i * t + r + 1  # noqa: E731
    report: dict = {"instance": "52_26", "base": "[31,26,3] Hamming H5", "t": t, "solutions": []}

    builder = CNFBuilder()
    builder.counter = k * t
    encode_gram(builder, s_var, k, t, gram)
    for attempt in range(solutions):
        started = time.perf_counter()
        result = solve(builder, out_dir / f"so_52_26_sol{attempt}.cnf", cadical, timeout=600)
        if result.get("status") != "SAT" or not result.get("model"):
            report["solutions"].append({"attempt": attempt, "status": result.get("status")})
            break
        s = extract_s(result["model"], k, t)
        verification = verify(base, s)
        matrix = verification.pop("matrix")
        matrix_path = out_dir / f"so_52_26_sol{attempt}_matrix.json"
        matrix_path.write_text(json.dumps(matrix) + "\n", encoding="utf-8")
        verification["matrix_path"] = str(matrix_path)
        report["solutions"].append(
            {
                "attempt": attempt,
                "status": "SAT",
                "solve_ms": round((time.perf_counter() - started) * 1000, 3),
                **verification,
            }
        )
        print(f"[52_26] attempt {attempt}: SO={verification['self_orthogonal']} d_min={verification['d_min']}", file=sys.stderr)
        blocking = [-(s_var(i, r)) if s[i, r] else s_var(i, r) for i in range(k) for r in range(t)]
        builder.add(blocking)
    return report


def gf2_rank(matrix: np.ndarray) -> int:
    m = matrix.copy().astype(np.uint8)
    rank = 0
    rows, cols = m.shape
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if m[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
    return rank


def rank_one_elimination(base: np.ndarray, *, pick: str = "first", hyper: str = "ei", rng=None) -> np.ndarray:
    """Manuscript Algorithm 1 with the stated pivot freedom made explicit."""
    k = base.shape[0]
    residual = (base @ base.T) % 2
    columns = []
    while residual.any():
        diag = np.flatnonzero(np.diag(residual))
        if len(diag):
            if pick == "first":
                i = int(diag[0])
            elif pick == "last":
                i = int(diag[-1])
            else:
                i = int(rng.choice(diag))
            v = residual[:, i].copy()
        else:
            pairs = np.argwhere(np.triu(residual, 1))
            sel = pairs[0] if pick != "random" else pairs[rng.integers(0, len(pairs))]
            i, j = map(int, sel)
            v = np.zeros(k, dtype=np.uint8)
            v[i] = 1
            if hyper == "eij":
                v[j] = 1
        columns.append(v.astype(np.uint8))
        residual = (residual + np.outer(v, v)) % 2
    s = np.stack(columns, axis=1)
    return np.concatenate([base, s], axis=1).astype(np.uint8)


def run_rank_one_sweep(out_dir: Path) -> dict:
    """(t, d_min) landscape of Algorithm 1 over pivot choices, for both Hamming bases."""
    report: dict = {"instance": "rank_one_sweep", "results": []}
    for r, label in ((4, "H4 [15,11,3]"), (5, "H5 [31,26,3]")):
        base = hamming_generator(r)
        configs: list[tuple[str, str, int | None]] = [
            ("first", "ei", None),
            ("last", "ei", None),
            ("first", "eij", None),
            ("last", "eij", None),
        ]
        for seed in range(6):
            configs.append(("random", "ei", seed))
            configs.append(("random", "eij", seed))
        for pick, hyper, seed in configs:
            rng = np.random.default_rng(seed) if seed is not None else None
            ext = rank_one_elimination(base, pick=pick, hyper=hyper, rng=rng)
            gram_zero = not ((ext @ ext.T) % 2).any()
            d_min, _ = dmin_meet_in_middle(ext)
            report["results"].append(
                {
                    "base": label,
                    "pick": pick if seed is None else f"{pick}{seed}",
                    "hyperbolic": hyper,
                    "t": int(ext.shape[1] - base.shape[1]),
                    "n": int(ext.shape[1]),
                    "self_orthogonal": bool(gram_zero),
                    "d_min": int(d_min),
                }
            )
            print(f"[rank1] {label} {pick}{'' if seed is None else seed}/{hyper}: t={report['results'][-1]['t']} d_min={d_min}", file=sys.stderr)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="22_11,52_26,rank_one_sweep")
    parser.add_argument("--solutions", type=int, default=5, help="Blocked SS^T=M solutions to try for 52_26.")
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cadical = resolve_cadical(args.cadical_path)
    if not cadical:
        raise SystemExit("CaDiCaL binary not found")

    reports = []
    for instance in [item.strip() for item in args.instances.split(",") if item.strip()]:
        if instance == "22_11":
            reports.append(run_22_11(out_dir, cadical))
        elif instance == "52_26":
            reports.append(run_52_26(out_dir, cadical, solutions=args.solutions))
        elif instance == "rank_one_sweep":
            reports.append(run_rank_one_sweep(out_dir))
        else:
            raise SystemExit(f"unknown instance {instance}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Reconstruction and independent verification of the binary SO embeddings.",
        "args": vars(args),
        "reports": reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
