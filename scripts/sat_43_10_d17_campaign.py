#!/usr/bin/env python3
"""[43,10,17] campaign: close the codetables gap for binary [43,10] (lb=16, ub=17).

Either outcome is decisive: SAT yields a new best-known code (d=17 meets the
upper bound), UNSAT proves d=16 optimal. The search runs over standard-form
generators G=[I_10 | P] (complete for existence up to equivalence), with:

- per-codeword distance constraints  weight(m P) >= d - w(m)  via sequential-
  counter cardinality encoding;
- full column-lexicographic symmetry breaking over the 33 parity columns of P
  (sound for both SAT and UNSAT: every column-permutation class keeps its
  lex-sorted representative);
- CEGAR refinement (SolEvolve verifier loop): start from the low-weight
  messages only, verify each SAT candidate exhaustively over all 1023
  codewords, and feed violated messages back as constraints. Because any
  constraint-subset UNSAT implies full UNSAT, the lazy loop can also finish
  the impossibility proof early.

Deterministic; any witness is re-verified independently and frozen.
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

from reconstruct_f2_so_embeddings import CNFBuilder, extract_s, solve  # noqa: E402
from solevolve.so_claim_repro import resolve_cadical  # noqa: E402

N, K = 43, 10
M = N - K  # 33 parity columns


def p_var(i: int, j: int) -> int:
    return i * M + j + 1


def at_most_k(builder: CNFBuilder, lits: list[int], k: int) -> None:
    """Sinz sequential-counter encoding of sum(lits) <= k."""
    n = len(lits)
    if k >= n:
        return
    if k == 0:
        for lit in lits:
            builder.add([-lit])
        return
    s = [[builder.new_var() for _ in range(k)] for _ in range(n)]
    builder.add([-lits[0], s[0][0]])
    for j in range(1, k):
        builder.add([-s[0][j]])
    for i in range(1, n):
        builder.add([-lits[i], s[i][0]])
        builder.add([-s[i - 1][0], s[i][0]])
        for j in range(1, k):
            builder.add([-lits[i], -s[i - 1][j - 1], s[i][j]])
            builder.add([-s[i - 1][j], s[i][j]])
        builder.add([-lits[i], -s[i - 1][k - 1]])


def at_least_t(builder: CNFBuilder, lits: list[int], t: int) -> None:
    """sum(lits) >= t  <=>  sum(~lits) <= n - t."""
    if t <= 0:
        return
    n = len(lits)
    if t > n:
        builder.add([])
        return
    at_most_k(builder, [-lit for lit in lits], n - t)


def add_message_constraint(builder: CNFBuilder, message: int, target_d: int) -> None:
    rows = [i for i in range(K) if message & (1 << i)]
    need = target_d - len(rows)
    if need <= 0:
        return
    ys = [builder.xor_chain([p_var(i, j) for i in rows], 0) for j in range(M)]
    at_least_t(builder, ys, need)


def add_column_lex_chain(builder: CNFBuilder) -> None:
    """Enforce column_j >=_lex column_{j+1} for all adjacent parity columns."""
    for j in range(M - 1):
        a = [p_var(i, j) for i in range(K)]
        b = [p_var(i, j + 1) for i in range(K)]
        prev_eq = None
        for i in range(K):
            if prev_eq is None:
                builder.add([a[i], -b[i]])  # a_0 >= b_0
            else:
                builder.add([-prev_eq, a[i], -b[i]])
            eq = builder.new_var()
            # eq -> (a_i = b_i) and eq -> prev_eq
            builder.add([-eq, a[i], -b[i]])
            builder.add([-eq, -a[i], b[i]])
            if prev_eq is not None:
                builder.add([-eq, prev_eq])
                builder.add([eq, -prev_eq, -a[i], -b[i]])
                builder.add([eq, -prev_eq, a[i], b[i]])
            else:
                builder.add([eq, -a[i], -b[i]])
                builder.add([eq, a[i], b[i]])
            prev_eq = eq


def build_cnf(active_messages: list[int], target_d: int, *, use_lex: bool) -> CNFBuilder:
    builder = CNFBuilder()
    builder.counter = K * M
    if use_lex:
        add_column_lex_chain(builder)
    for message in active_messages:
        add_message_constraint(builder, message, target_d)
    return builder


def weights_all_messages(p_matrix: np.ndarray) -> np.ndarray:
    """Codeword weights for all 1023 nonzero messages of G=[I|P]."""
    messages = np.arange(1, 1 << K, dtype=np.uint32)
    msg_bits = ((messages[:, None] >> np.arange(K)[None, :]) & 1).astype(np.uint8)
    ext = msg_bits @ p_matrix % 2
    return msg_bits.sum(axis=1) + ext.sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-d", type=int, default=17)
    parser.add_argument("--mode", choices=["cegar", "eager"], default="cegar")
    parser.add_argument("--initial-max-weight", type=int, default=2,
                        help="cegar: initially constrain messages of weight <= this.")
    parser.add_argument("--max-new-per-round", type=int, default=200)
    parser.add_argument("--max-rounds", type=int, default=60)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--no-lex", action="store_true", help="Disable column-lex symmetry breaking.")
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cadical = resolve_cadical(args.cadical_path)
    if not cadical:
        raise SystemExit("CaDiCaL binary not found")

    all_messages = list(range(1, 1 << K))
    if args.mode == "eager":
        active = set(all_messages)
    else:
        active = {m for m in all_messages if bin(m).count("1") <= args.initial_max_weight}

    rounds = []
    witness = None
    status_final = None
    for round_idx in range(args.max_rounds):
        builder = build_cnf(sorted(active), args.target_d, use_lex=not args.no_lex)
        stats = {
            "round": round_idx,
            "active_messages": len(active),
            "variables": builder.counter,
            "clauses": len(builder.clauses),
        }
        started = time.perf_counter()
        result = solve(builder, out_dir / f"round_{round_idx}.cnf", cadical, timeout=args.timeout_sec)
        stats["solve_ms"] = round((time.perf_counter() - started) * 1000, 3)
        stats["status"] = result.get("status")
        (out_dir / f"round_{round_idx}.cnf").unlink(missing_ok=True)
        Path(str(out_dir / f"round_{round_idx}.cnf") + ".cadical.out").unlink(missing_ok=True)
        print(f"[d17] round {round_idx}: {stats}", file=sys.stderr, flush=True)

        if result.get("status") == "UNSAT":
            status_final = "UNSAT"
            stats["conclusion"] = (
                "UNSAT with a constraint subset implies the full instance is UNSAT: "
                f"no standard-form [{N},{K},{args.target_d}] code exists, so d={args.target_d - 1} is optimal."
                if args.mode == "cegar" else "UNSAT."
            )
            rounds.append(stats)
            break
        if result.get("status") != "SAT" or not result.get("model"):
            status_final = str(result.get("status"))
            rounds.append(stats)
            break

        p_matrix = extract_s(result["model"], K, M)
        weights = weights_all_messages(p_matrix)
        d_min = int(weights.min())
        stats["candidate_d_min"] = d_min
        violated = [all_messages[idx] for idx in np.flatnonzero(weights < args.target_d)]
        stats["verifier_violations"] = len(violated)
        rounds.append(stats)
        print(f"[d17] round {round_idx}: candidate d_min={d_min}, violations={len(violated)}",
              file=sys.stderr, flush=True)
        if not violated:
            witness = p_matrix
            status_final = "SAT"
            break
        fresh = [m for m in violated if m not in active][: args.max_new_per_round]
        if not fresh:
            raise SystemExit("violations found but all already constrained; encoding bug")
        active.update(fresh)

    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": f"[{N},{K},{args.target_d}] campaign: codetables gap [43,10] lb=16 ub=17.",
        "args": vars(args),
        "rounds": rounds,
        "final_status": status_final,
        "witness_found": witness is not None,
    }
    if witness is not None:
        g = np.concatenate([np.eye(K, dtype=np.uint8), witness], axis=1)
        weights = weights_all_messages(witness)
        distribution = {int(w): int(c) for w, c in zip(*np.unique(weights, return_counts=True))}
        report["verification"] = {
            "d_min": int(weights.min()),
            "weight_distribution": {0: 1, **distribution},
        }
        matrix_path = out_dir / f"witness_43_10_{args.target_d}.json"
        matrix_path.write_text(json.dumps(g.astype(int).tolist()) + "\n", encoding="utf-8")
        report["matrix_path"] = str(matrix_path)
        print(f"[d17] WITNESS FOUND: d_min={weights.min()}", file=sys.stderr, flush=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
