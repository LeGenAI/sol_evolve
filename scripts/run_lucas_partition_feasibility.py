#!/usr/bin/env python3
"""Deterministic SAT feasibility reproduction for generalized Lucas cube partitions.

For each instance Lambda_n(1^s), encodes "does a perfect partition by closed
radius-1 balls exist?" as an exactly-one SAT problem (one selector variable per
vertex; every vertex covered by exactly one selected ball) and runs CaDiCaL.
Reproduces the SAT/UNSAT statuses of the manuscript's Lucas table, including the
UNSAT rows; wall-clock is re-measured and hardware-dependent.

The vertex count of every instance is asserted against the manuscript's |V|
column, so the graph construction itself is cross-checked.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.engines.sat_solver_interface import SATSolver  # noqa: E402
from solevolve.lucas_repro import lucas_vertices  # noqa: E402
from solevolve.so_claim_repro import resolve_cadical  # noqa: E402

# (n, s) -> (expected |V|, manuscript status, per-instance timeout seconds)
INSTANCES = {
    (7, 4): (99, "SAT", 60),
    (7, 3): (71, "UNSAT", 60),
    (7, 2): (29, "UNSAT", 60),
    (15, 2): (1364, "UNSAT", 300),
    (15, 3): (9327, "UNSAT", 1800),
    (15, 4): (18842, "UNSAT", 3600),
    (15, 12): (32707, "SAT", 1800),
    (15, 11): (32647, "SAT", 1800),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        default="7_4,7_3,7_2,15_2,15_3,15_4",
        help="Comma-separated n_s pairs; defaults to the small SAT sanity row plus all UNSAT rows.",
    )
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--keep-cnf", action="store_true")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def encode_perfect_partition(vertices: list[str], n: int) -> tuple[list[list[int]], int]:
    index = {vertex: idx + 1 for idx, vertex in enumerate(vertices)}
    vertex_set = set(vertices)
    clauses: list[list[int]] = []
    for vertex in vertices:
        ball = [index[vertex]]
        for bit in range(n):
            flipped = vertex[:bit] + ("1" if vertex[bit] == "0" else "0") + vertex[bit + 1 :]
            if flipped in vertex_set:
                ball.append(index[flipped])
        clauses.append(ball)  # covered at least once
        for i in range(len(ball)):
            for j in range(i + 1, len(ball)):
                clauses.append([-ball[i], -ball[j]])  # covered at most once
    return clauses, len(vertices)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_cadical(args.cadical_path)
    if not resolved:
        raise SystemExit("CaDiCaL binary not found")

    requested = []
    for token in args.instances.split(","):
        n_str, s_str = token.strip().split("_")
        requested.append((int(n_str), int(s_str)))

    records = []
    for n, s in requested:
        expected_v, expected_status, timeout = INSTANCES[(n, s)]
        build_started = time.perf_counter()
        vertices = lucas_vertices(n, s)
        if len(vertices) != expected_v:
            raise SystemExit(f"Lambda_{n}(1^{s}): built {len(vertices)} vertices, manuscript reports {expected_v}")
        clauses, num_vars = encode_perfect_partition(vertices, n)
        cnf_path = out_dir / f"lucas_{n}_{s}.cnf"
        with cnf_path.open("w", encoding="utf-8") as handle:
            handle.write(f"p cnf {num_vars} {len(clauses)}\n")
            for clause in clauses:
                handle.write(" ".join(str(lit) for lit in clause) + " 0\n")
        encode_ms = round((time.perf_counter() - build_started) * 1000, 3)

        solver = SATSolver(solver_type="cadical", solver_path=resolved)
        result = solver.solve(str(cnf_path), timeout=timeout, verbose=False)
        status = str(result.get("status") or "UNKNOWN")
        if not args.keep_cnf:
            cnf_path.unlink(missing_ok=True)
            Path(str(cnf_path) + ".cadical.out").unlink(missing_ok=True)
        records.append(
            {
                "instance": f"Lambda_{n}(1^{s})",
                "n": n,
                "s": s,
                "vertices": len(vertices),
                "variables": num_vars,
                "clauses": len(clauses),
                "encode_ms": encode_ms,
                "solver_status": status,
                "solve_ms": round(float(result.get("time") or 0.0) * 1000, 3),
                "timeout_sec": timeout,
                "manuscript_status": expected_status,
                "status_matches_manuscript": status == expected_status,
            }
        )
        print(
            f"[lucas] Lambda_{n}(1^{s}): |V|={len(vertices)} clauses={len(clauses)} "
            f"status={status} (manuscript: {expected_status}) {records[-1]['solve_ms']:.0f} ms",
            file=sys.stderr,
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "SAT feasibility reproduction for generalized Lucas cube perfect partitions (exactly-one ball-cover encoding, CaDiCaL).",
        "args": vars(args),
        "records": records,
        "all_statuses_match": all(record["status_matches_manuscript"] for record in records),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
