#!/usr/bin/env python3
"""SAT-driven construction of the binary [52,26] self-orthogonal embedding with d_min >= 8.

This is the architecture-faithful construction: the extension S (26 x 21) is the
SAT unknown, self-orthogonality is the Gram condition S S^T = G G^T, and minimum
distance is enforced through per-codeword cardinality constraints. Because a
self-orthogonal binary code has only even weights, enforcing weight >= 7 already
yields d >= 8, so constraints are needed only for the 29,016 base codewords of
weight <= 6 (weight-7 base codewords are covered by parity).

Two modes:
- eager: encode all 29,016 distance constraints up front (single SAT query).
- cegar: start from the base-weight <= 4 constraints only, then run the
  SolEvolve-style verifier loop: CaDiCaL proposes S, the deterministic verifier
  scans every low-weight codeword, and the violated codewords are fed back as
  new constraints for the next SAT query, until the verifier accepts.

Any accepted witness is re-verified exhaustively (all 2^26 codewords) and for
self-orthogonality, then frozen to JSON. Deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
from search_52_26_d8_so_embedding import low_weight_messages  # noqa: E402
from solevolve.so_claim_repro import resolve_cadical  # noqa: E402

K, T, TARGET_D = 26, 21, 8


def encode_distance_for_messages(
    builder: CNFBuilder,
    s_var,
    messages: np.ndarray,
    base_weights: np.ndarray,
    rows_of: list[np.ndarray],
) -> None:
    """weight(extension of message m) >= 7 - w_base(m) for each given message."""
    for idx in range(len(messages)):
        need = 7 - int(base_weights[idx])
        if need <= 0:
            continue
        rows = rows_of[idx]
        ys = []
        for r in range(T):
            y = builder.xor_chain([s_var(int(i), r) for i in rows], 0)
            ys.append(y)
        builder.at_least(ys, need)


def build_and_solve(
    base: np.ndarray,
    gram: np.ndarray,
    messages: np.ndarray,
    base_weights: np.ndarray,
    active: np.ndarray,
    rows_of: list[np.ndarray],
    *,
    cnf_path: Path,
    cadical: str,
    timeout: int,
) -> tuple[dict, dict]:
    s_var = lambda i, r: i * T + r + 1  # noqa: E731
    builder = CNFBuilder()
    builder.counter = K * T
    encode_gram(builder, s_var, K, T, gram)
    idxs = np.flatnonzero(active)
    encode_distance_for_messages(
        builder, s_var, messages[idxs], base_weights[idxs], [rows_of[i] for i in idxs]
    )
    stats = {"variables": builder.counter, "clauses": len(builder.clauses), "active_constraints": int(len(idxs))}
    started = time.perf_counter()
    result = solve(builder, cnf_path, cadical, timeout=timeout)
    stats["solve_ms"] = round((time.perf_counter() - started) * 1000, 3)
    stats["status"] = result.get("status")
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cegar", "eager"], default="cegar")
    parser.add_argument("--initial-max-base-weight", type=int, default=4,
                        help="cegar: encode constraints for base codewords up to this weight initially.")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--keep-cnf", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cadical = resolve_cadical(args.cadical_path)
    if not cadical:
        raise SystemExit("CaDiCaL binary not found")

    base = hamming_generator(5)
    gram = (base @ base.T) % 2
    assert gf2_rank(gram) == T
    messages, base_weights = low_weight_messages(base, TARGET_D - 1)
    counts = {int(w): int((base_weights == w).sum()) for w in sorted(set(base_weights.tolist()))}
    assert counts.get(3) == 155 and counts.get(4) == 1085
    rows_of = [np.flatnonzero(messages[i]) for i in range(len(messages))]
    constrainable = base_weights <= 6  # parity covers base weight 7
    print(f"[sat-d8] low-weight base codewords: {len(messages)} by weight {counts}; "
          f"constrainable (w<=6): {int(constrainable.sum())}", file=sys.stderr)

    if args.mode == "eager":
        active = constrainable.copy()
    else:
        active = constrainable & (base_weights <= args.initial_max_base_weight)

    rounds = []
    witness = None
    for round_idx in range(args.max_rounds):
        result, stats = build_and_solve(
            base, gram, messages, base_weights, active, rows_of,
            cnf_path=out_dir / f"round_{round_idx}.cnf",
            cadical=cadical,
            timeout=args.timeout_sec,
        )
        print(f"[sat-d8] round {round_idx}: {stats}", file=sys.stderr)
        if not args.keep_cnf:
            (out_dir / f"round_{round_idx}.cnf").unlink(missing_ok=True)
            Path(str(out_dir / f"round_{round_idx}.cnf") + ".cadical.out").unlink(missing_ok=True)
        round_record = {"round": round_idx, **stats}
        if result.get("status") != "SAT" or not result.get("model"):
            rounds.append(round_record)
            break
        s = extract_s(result["model"], K, T)
        ext_weights = (messages @ s % 2).sum(axis=1)
        totals = base_weights + ext_weights
        violated = np.flatnonzero(totals < TARGET_D)
        round_record["verifier_violations"] = int(len(violated))
        rounds.append(round_record)
        print(f"[sat-d8] round {round_idx}: verifier violations = {len(violated)}", file=sys.stderr)
        if len(violated) == 0:
            witness = s
            break
        newly = violated[~active[violated]]
        if len(newly) == 0:
            raise SystemExit("verifier found violations but all are already constrained; encoding bug")
        active[newly] = True

    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "SAT-driven [52,26] SO embedding with d>=8: Gram condition plus per-codeword distance constraints; cegar mode feeds verifier-found violations back as constraints.",
        "args": vars(args),
        "low_weight_base_counts": counts,
        "rounds": rounds,
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
        matrix_path = out_dir / "so_52_26_d8_matrix.json"
        matrix_path.write_text(json.dumps(ext.astype(int).tolist()) + "\n", encoding="utf-8")
        report["matrix_path"] = str(matrix_path)
        print(f"[sat-d8] WITNESS: SO={gram_zero} d_min={d_min}", file=sys.stderr)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
