#!/usr/bin/env python3
"""Encoding-choice ablation: measure CNF size and solver behavior per encoding variant.

For each encoding variant, encodes the direct feasibility query "does a binary
[n,k,d] code exist?" and runs CaDiCaL, recording variables, clauses, encode time,
solve time, and solver status. Deterministic; no LLM calls.

Variants toggle the encoder options actually exposed by CNFEncoder:
- systematic generator form on/off
- symmetry-breaking constraint on/off
- all-one-vector (self-complementary) constraint on/off
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.engines.cnf_encoder import CNFEncoder  # noqa: E402
from solevolve.engines.sat_solver_interface import SATSolver  # noqa: E402
from solevolve.so_claim_repro import resolve_cadical  # noqa: E402

VARIANTS = {
    "production": {"systematic": True, "symmetry": True, "all_one": False},
    "no_symmetry": {"systematic": True, "symmetry": False, "all_one": False},
    "non_systematic": {"systematic": False, "symmetry": True, "all_one": False},
    "with_all_one": {"systematic": True, "symmetry": True, "all_one": True},
}

INSTANCES = {
    "22_11_7": {"n": 22, "k": 11, "d": 7, "timeout": 60},
    "43_10_16": {"n": 43, "k": 10, "d": 16, "timeout": 300},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default=",".join(sorted(INSTANCES)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--repeats", type=int, default=3, help="Solve repeats for timing stability.")
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "cnf").mkdir(parents=True, exist_ok=True)
    resolved = resolve_cadical(args.cadical_path)
    if not resolved:
        raise SystemExit("CaDiCaL binary not found")

    instances = [item.strip() for item in args.instances.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    records = []
    for instance in instances:
        cfg = INSTANCES[instance]
        for variant in variants:
            opts = VARIANTS[variant]
            encode_started = time.perf_counter()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                encoder = CNFEncoder(cfg["n"], cfg["k"], cfg["d"], systematic=opts["systematic"])
                encoder.encode_all_constraints(
                    include_all_one=opts["all_one"],
                    add_symmetry_breaking=opts["symmetry"],
                )
                cnf_path = out_dir / "cnf" / f"{instance}_{variant}.cnf"
                encoder.export_to_dimacs(str(cnf_path))
            encode_ms = round((time.perf_counter() - encode_started) * 1000, 3)
            num_vars = encoder.var_counter - 1
            num_clauses = len(encoder.clauses)

            solve_times = []
            statuses = []
            for repeat in range(args.repeats):
                with contextlib.redirect_stdout(io.StringIO()):
                    solver = SATSolver(solver_type="cadical", solver_path=resolved)
                    result = solver.solve(str(cnf_path), timeout=cfg["timeout"], verbose=False)
                statuses.append(str(result.get("status") or "UNKNOWN"))
                solve_times.append(round(float(result.get("time") or 0.0) * 1000, 3))
            records.append(
                {
                    "instance": instance,
                    "variant": variant,
                    "options": opts,
                    "variables": num_vars,
                    "clauses": num_clauses,
                    "encode_ms": encode_ms,
                    "solve_ms_runs": solve_times,
                    "solve_ms_median": statistics.median(solve_times),
                    "statuses": statuses,
                    "timeout_sec": cfg["timeout"],
                }
            )
            print(
                f"[encoding] {instance} {variant}: vars={num_vars} clauses={num_clauses} "
                f"median_solve={statistics.median(solve_times):.0f} ms statuses={statuses}",
                file=sys.stderr,
            )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Encoding-choice ablation on direct [n,k,d] feasibility queries (CaDiCaL).",
        "args": vars(args),
        "records": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
