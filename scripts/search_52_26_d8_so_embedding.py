#!/usr/bin/env python3
"""Search for a binary [52,26] self-orthogonal embedding of H5 with d_min >= 8.

Strategy: solutions of S S^T = G G^T are closed under S -> S Q for any GF(2)
orthogonal Q. Transvections Q = I + u u^T with even-weight u are orthogonal, so
the Gram-preserving move S -> S + (S u) u^T explores the solution space exactly.
Starting from SAT-constructed seed solutions, simulated annealing minimizes the
number of codewords of weight < 8.

Only messages whose H5 base codeword has weight <= 7 can violate d >= 8 (the
extension adds nonnegative weight), so the objective enumerates exactly those
codewords once and evaluates each candidate S with a single mod-2 matrix product.
Any reported witness is re-verified exhaustively (all 2^26 codewords) and for
self-orthogonality. Deterministic given --seed.
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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from reconstruct_f2_so_embeddings import (  # noqa: E402
    CNFBuilder,
    dmin_meet_in_middle,
    encode_gram,
    extract_s,
    gf2_rank,
    hamming_generator,
    solve,
)
from solevolve.so_claim_repro import resolve_cadical  # noqa: E402

K, T, TARGET_D = 26, 21, 8


def low_weight_messages(base: np.ndarray, max_weight: int) -> tuple[np.ndarray, np.ndarray]:
    """Messages of the systematic H5 code whose base codewords have weight <= max_weight.

    Codewords of weight w correspond to w-subsets of the parity-check columns
    that XOR to zero. H = [A^T | I_5] for G = [I_26 | A]; column j < 26 is row j
    of A, column 26+b is the unit vector e_b. The message support is the subset
    of selected columns with index < 26.
    """
    a = base[:, K:]  # 26 x 5
    columns = [int(sum(int(a[j, b]) << b for b in range(5))) for j in range(K)]
    columns += [1 << b for b in range(5)]
    n = len(columns)
    supports = []
    for w in range(3, max_weight + 1):
        for combo in combinations(range(n), w):
            acc = 0
            for idx in combo:
                acc ^= columns[idx]
            if acc == 0:
                supports.append(combo)
    messages = np.zeros((len(supports), K), dtype=np.uint8)
    base_weights = np.zeros(len(supports), dtype=np.int64)
    for row, combo in enumerate(supports):
        base_weights[row] = len(combo)
        for idx in combo:
            if idx < K:
                messages[row, idx] = 1
    return messages, base_weights


def violations(messages: np.ndarray, base_weights: np.ndarray, s: np.ndarray) -> tuple[int, int]:
    ext = (messages @ s) % 2
    totals = base_weights + ext.sum(axis=1)
    deficit = np.maximum(0, TARGET_D - totals)
    return int((totals < TARGET_D).sum()), int(deficit.sum())


def gram_seed_solutions(base: np.ndarray, cadical: str, out_dir: Path, count: int) -> list[np.ndarray]:
    gram = (base @ base.T) % 2
    assert gf2_rank(gram) == T
    s_var = lambda i, r: i * T + r + 1  # noqa: E731
    builder = CNFBuilder()
    builder.counter = K * T
    encode_gram(builder, s_var, K, T, gram)
    seeds = []
    for attempt in range(count):
        result = solve(builder, out_dir / f"seed_{attempt}.cnf", cadical, timeout=600)
        if result.get("status") != "SAT" or not result.get("model"):
            break
        s = extract_s(result["model"], K, T)
        seeds.append(s)
        builder.add([-(s_var(i, r)) if s[i, r] else s_var(i, r) for i in range(K) for r in range(T)])
    return seeds


def anneal(
    s0: np.ndarray,
    messages: np.ndarray,
    base_weights: np.ndarray,
    *,
    rng: np.random.Generator,
    iterations: int,
    report_every: int = 200_000,
) -> tuple[np.ndarray, int, int]:
    s = s0.copy()
    count, deficit = violations(messages, base_weights, s)
    best_s, best = s.copy(), (count, deficit)
    temp_hi, temp_lo = 4.0, 0.05
    for it in range(iterations):
        if best[0] == 0:
            break
        weight = 2 if rng.random() < 0.7 else 4
        u = np.zeros(T, dtype=np.uint8)
        u[rng.choice(T, size=weight, replace=False)] = 1
        su = (s @ u) % 2
        candidate = (s + np.outer(su, u)) % 2
        c_count, c_deficit = violations(messages, base_weights, candidate)
        temp = temp_hi * (temp_lo / temp_hi) ** (it / max(1, iterations - 1))
        delta = c_deficit - deficit
        if delta <= 0 or rng.random() < np.exp(-delta / temp):
            s, count, deficit = candidate, c_count, c_deficit
            if (count, deficit) < best:
                best_s, best = s.copy(), (count, deficit)
        if report_every and it % report_every == 0:
            print(f"    iter {it}: current viol={count} deficit={deficit} best={best}", file=sys.stderr)
    return best_s, best[0], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=400_000)
    parser.add_argument("--sat-seeds", type=int, default=4)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cadical = resolve_cadical(args.cadical_path)
    if not cadical:
        raise SystemExit("CaDiCaL binary not found")

    base = hamming_generator(5)
    started = time.perf_counter()
    messages, base_weights = low_weight_messages(base, TARGET_D - 1)
    counts = {int(w): int((base_weights == w).sum()) for w in sorted(set(base_weights.tolist()))}
    print(f"[d8] low-weight base codewords (w<=7): {len(messages)} by weight {counts} "
          f"({time.perf_counter() - started:.1f}s)", file=sys.stderr)
    assert counts.get(3) == 155 and counts.get(4) == 1085, "Hamming(31,26) low-weight counts mismatch"

    seeds = gram_seed_solutions(base, cadical, out_dir, args.sat_seeds)
    print(f"[d8] SAT seed solutions: {len(seeds)}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    attempts = []
    witness = None
    for restart in range(args.restarts):
        s0 = seeds[restart % len(seeds)]
        best_s, count, deficit = anneal(
            s0, messages, base_weights, rng=rng, iterations=args.iterations
        )
        attempts.append({"restart": restart, "violations": count, "deficit": deficit})
        print(f"[d8] restart {restart}: violations={count} deficit={deficit}", file=sys.stderr)
        if count == 0:
            witness = best_s
            break

    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Gram-preserving transvection annealing for a [52,26] SO embedding with d>=8.",
        "args": vars(args),
        "low_weight_base_counts": counts,
        "attempts": attempts,
        "witness_found": witness is not None,
    }
    if witness is not None:
        ext = np.concatenate([base, witness], axis=1).astype(np.uint8)
        gram_zero = not ((ext @ ext.T) % 2).any()
        d_min, distribution = dmin_meet_in_middle(ext)
        report["verification"] = {
            "self_orthogonal": bool(gram_zero),
            "d_min": int(d_min),
            "weight_distribution": {int(w): int(c) for w, c in distribution.items()},
        }
        (out_dir / "so_52_26_d8_matrix.json").write_text(json.dumps(ext.astype(int).tolist()) + "\n", encoding="utf-8")
        report["matrix_path"] = str(out_dir / "so_52_26_d8_matrix.json")
        print(f"[d8] WITNESS: SO={gram_zero} d_min={d_min}", file=sys.stderr)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
