#!/usr/bin/env python3
"""Matched-seed ablation of the GA's A_d objective term.

Seeds the GA population from an archived SAT-verified witness and evolves it under
two selection objectives on the same random seeds:

- d_min_only:  score = 10_000 * d_min                 (no minimum-weight-count pressure)
- d_min_a_d:   score = 10_000 * d_min - A_d           (production objective)

The primary metric is the A_d of the final delivered best candidate (verified
d_min >= target); the per-generation best_A_d trajectory is recorded for both.
Deterministic: no LLM calls, no SAT repair (repair_interval=0).
"""

from __future__ import annotations

import argparse
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

import solevolve.hybrid_ga as hybrid_ga  # noqa: E402
from solevolve.contracts import HybridGAPolicy  # noqa: E402
from solevolve.hybrid_ga import (  # noqa: E402
    _diagnostics,
    _load_hybrid_artifact,
    evolve_population,
    initialize_population,
)

OBJECTIVES = ("d_min_only", "d_min_a_d")

INSTANCES = {
    "22_11_7": {"n": 22, "k": 11, "d": 7},
    "43_10_16": {"n": 43, "k": 10, "d": 16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=sorted(INSTANCES), required=True)
    parser.add_argument("--objectives", default=",".join(OBJECTIVES))
    parser.add_argument("--runs", type=int, default=30, help="Matched random seeds per objective.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument(
        "--artifact-path",
        default=str(REPO_ROOT / "artifacts" / "reviewer_core_claims.json"),
        help="Reviewer core-claims JSON with archived witnesses.",
    )
    parser.add_argument(
        "--seed-matrix-json",
        default=None,
        help="Optional path to an explicit seed generator matrix (overrides the archived witness).",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_seed_matrix(instance: str, artifact_path: Path) -> tuple[list[list[int]], dict]:
    if instance == "22_11_7":
        payload, resolved, error = _load_hybrid_artifact(
            artifact_dir=artifact_path.parent,
            artifact_path=artifact_path,
        )
        if error or payload is None:
            raise RuntimeError(error or f"could not load hybrid artifact from {resolved}")
        matrix = payload.get("generator_matrix")
        if not isinstance(matrix, list):
            raise RuntimeError("reviewer_core_claims.json is missing the verified final witness matrix")
        return [[int(value) & 1 for value in row] for row in matrix], {"source": "verified_final_witness"}
    if instance == "43_10_16":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        solutions = payload["binary_codes"]["binary_ad_43_10_16"]["solutions"]
        first = min(solutions, key=lambda item: float(item.get("discovery_time_sec") or 0.0))
        return first["generator_matrix"], {
            "source": "archived_first_sat_witness",
            "seed_expected_A_d": first.get("expected_A_d"),
        }
    raise ValueError(instance)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = INSTANCES[args.instance]
    n, k, d = cfg["n"], cfg["k"], cfg["d"]
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    unknown = set(objectives) - set(OBJECTIVES)
    if unknown:
        raise SystemExit(f"unknown objectives: {sorted(unknown)}")

    if args.seed_matrix_json:
        seed_matrix = json.loads(Path(args.seed_matrix_json).read_text())
        seed_meta = {"source": args.seed_matrix_json}
    else:
        seed_matrix, seed_meta = load_seed_matrix(args.instance, Path(args.artifact_path))
    seed_diag = _diagnostics(seed_matrix, n=n, k=k, d=d, expected_a_d=None)
    if int(seed_diag.get("minimum_distance") or 0) < d:
        raise SystemExit(f"seed matrix has d_min={seed_diag.get('minimum_distance')} < target {d}")
    print(
        f"[ad-ablation] instance={args.instance} seed d_min={seed_diag.get('minimum_distance')} "
        f"A_d={seed_diag.get('A_d')} ({seed_meta})",
        file=sys.stderr,
    )

    records = []
    for objective in objectives:
        hybrid_ga.SCORE_OBJECTIVE = objective
        for run_idx in range(args.runs):
            run_seed = args.seed_start + run_idx
            policy = HybridGAPolicy(
                enabled=True,
                seed=run_seed,
                population=args.population,
                generations=args.generations,
                repair_interval=0,
                target_distance=d,
            )
            started = time.perf_counter()
            init = initialize_population(seed_matrix=seed_matrix, policy=policy, n=n, k=k)
            evolved = evolve_population(
                population=init["population"],
                policy=policy,
                n=n,
                k=k,
                target_d=d,
                expected_a_d=0,  # unreachable: disables the early-exit so all generations run
            )
            best = evolved["best_matrix"]
            final_diag = _diagnostics(best, n=n, k=k, d=d, expected_a_d=None)
            trajectory = [
                {"generation": s.get("generation"), "best_d_min": s.get("best_d_min"), "best_A_d": s.get("best_A_d")}
                for s in evolved.get("generation_summaries", [])
            ]
            records.append(
                {
                    "instance": args.instance,
                    "objective": objective,
                    "run_seed": run_seed,
                    "final_d_min": final_diag.get("minimum_distance"),
                    "final_A_d": final_diag.get("A_d"),
                    "target_achieved": int(final_diag.get("minimum_distance") or 0) >= d,
                    "seed_A_d": seed_diag.get("A_d"),
                    "trajectory": trajectory,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            print(
                f"[ad-ablation] {objective} run {run_idx + 1}/{args.runs} "
                f"final d_min={final_diag.get('minimum_distance')} A_d={final_diag.get('A_d')} "
                f"({records[-1]['elapsed_ms']:.0f} ms)",
                file=sys.stderr,
            )

    # Aggregate: paired per-seed comparison on runs where both objectives kept d_min >= target.
    summaries = []
    for objective in objectives:
        rows = [r for r in records if r["objective"] == objective]
        achieved = [r for r in rows if r["target_achieved"]]
        values = sorted(r["final_A_d"] for r in achieved)
        summaries.append(
            {
                "objective": objective,
                "runs": len(rows),
                "target_achieved": len(achieved),
                "final_A_d_min": values[0] if values else None,
                "final_A_d_median": statistics.median(values) if values else None,
                "final_A_d_iqr": (
                    [statistics.quantiles(values, n=4)[0], statistics.quantiles(values, n=4)[2]]
                    if len(values) >= 4
                    else None
                ),
                "mean_elapsed_ms": statistics.fmean(r["elapsed_ms"] for r in rows) if rows else None,
            }
        )
    paired = None
    if set(objectives) == set(OBJECTIVES):
        by_seed = {}
        for r in records:
            by_seed.setdefault(r["run_seed"], {})[r["objective"]] = r
        diffs = []
        for run_seed, pair in sorted(by_seed.items()):
            a, b = pair.get("d_min_a_d"), pair.get("d_min_only")
            if a and b and a["target_achieved"] and b["target_achieved"]:
                diffs.append(a["final_A_d"] - b["final_A_d"])
        wins = sum(1 for x in diffs if x < 0)
        losses = sum(1 for x in diffs if x > 0)
        paired = {
            "paired_runs": len(diffs),
            "a_d_objective_lower_A_d": wins,
            "a_d_objective_higher_A_d": losses,
            "ties": len(diffs) - wins - losses,
            "median_diff_dmin_a_d_minus_dmin_only": statistics.median(diffs) if diffs else None,
            "sign_test_exact_p": hybrid_ga_sign_test(wins, losses),
        }

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Matched-seed GA objective ablation: d_min-only vs d_min + A_d selection.",
        "args": vars(args),
        "instance_config": cfg,
        "seed_meta": {**seed_meta, "seed_d_min": seed_diag.get("minimum_distance"), "seed_A_d": seed_diag.get("A_d")},
        "summaries": summaries,
        "paired": paired,
    }
    (out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


def hybrid_ga_sign_test(wins: int, losses: int) -> float:
    import math

    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, i) for i in range(0, min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


if __name__ == "__main__":
    main()
