#!/usr/bin/env python3
"""Decision-policy replay ablation for the Reflector's threshold policy.

Re-runs the deterministic [22,11,7] GA proxy (matched seeds, per-generation
logging) and replays alternative repair-trigger policies on the identical
trajectories. For each (policy, seed) the candidate at the trigger generation is
then actually repaired with all-parity SAT (30 s), so end-to-end costs are
measured, not estimated.

This is a decision-level replay: it compares WHEN each policy fires the first
repair and what that costs. It cannot emulate post-intervention trajectory
changes (e.g., mutation boosts after an exploration action); the manuscript
states this limitation explicitly.

Policies:
- reflector_threshold: production priority chain (improvement-rate exploitation,
  diversity-collapse exploration preempts repair, stagnation >= N_stag -> repair).
- reflector_no_diversity_guard: same, but the diversity branch never preempts.
- scalar_dmin_stag: stagnation counted on best d_min only (no fitness/A_d signal).
- fixed_25 / fixed_50: unconditional repair at a fixed generation.
- random_uniform: seeded uniform trigger in [1, horizon] (negative control).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.baselines.repair_policy_eval import RepairPolicyConfig, repair_with_columns  # noqa: E402
from solevolve.contracts import HybridGAPolicy  # noqa: E402
from solevolve.graph import REFLECTOR_THRESHOLDS  # noqa: E402
from solevolve.hybrid_ga import (  # noqa: E402
    TARGET_D,
    TARGET_K,
    TARGET_N,
    _diagnostics,
    _diversity,
    _load_hybrid_artifact,
    _matrix_from_repair_payload,
    _mutate_matrix,
    _score,
    initialize_population,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="Matched GA seeds.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=150, help="GA generations per trajectory.")
    parser.add_argument("--repair-timeout-sec", type=float, default=30.0)
    parser.add_argument(
        "--artifact-path",
        default=str(REPO_ROOT / "artifacts" / "reviewer_core_claims.json"),
    )
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def ga_trajectory(
    seed_matrix: list[list[int]],
    *,
    run_seed: int,
    population_size: int,
    horizon: int,
) -> dict:
    """Faithful replica of hybrid_ga.evolve_population's loop with per-generation logging."""
    policy = HybridGAPolicy(enabled=True, seed=run_seed, population=population_size, generations=horizon, repair_interval=0)
    init = initialize_population(seed_matrix=seed_matrix, policy=policy, n=TARGET_N, k=TARGET_K)
    population = init["population"]
    rng = np.random.default_rng(policy.seed)

    started = time.perf_counter()
    per_gen = []
    best_matrices = []
    for generation in range(0, horizon + 1):
        scored = sorted(
            (
                (_score(_diagnostics(matrix, n=TARGET_N, k=TARGET_K, d=TARGET_D, expected_a_d=None)), matrix)
                for matrix in population
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_matrix = scored[0]
        best_diag = _diagnostics(best_matrix, n=TARGET_N, k=TARGET_K, d=TARGET_D, expected_a_d=None)
        per_gen.append(
            {
                "generation": generation,
                "best_fitness": best_score,
                "best_d_min": int(best_diag.get("minimum_distance") or 0),
                "best_A_d": best_diag.get("A_d"),
                "diversity": _diversity(population, k=TARGET_K),
            }
        )
        best_matrices.append(best_matrix)
        if generation == horizon:
            break
        elites = [matrix for _, matrix in scored[: max(1, min(4, len(scored)))]]
        next_population = list(elites)
        while len(next_population) < policy.population:
            parent_a = elites[int(rng.integers(0, len(elites)))]
            parent_b = elites[int(rng.integers(0, len(elites)))]
            cut = int(rng.integers(TARGET_K + 1, TARGET_N))
            child = [row[:cut] + other_row[cut:] for row, other_row in zip(parent_a, parent_b, strict=True)]
            next_population.append(_mutate_matrix(child, rng, k=TARGET_K))
        population = next_population
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {"per_gen": per_gen, "best_matrices": best_matrices, "ga_elapsed_ms": elapsed_ms}


def replay_triggers(per_gen: list[dict], *, horizon: int, run_seed: int) -> dict[str, dict]:
    tau_rho = REFLECTOR_THRESHOLDS["tau_rho"]
    tau_d = REFLECTOR_THRESHOLDS["tau_D"]
    n_stag = REFLECTOR_THRESHOLDS["N_stag"]

    def improvement_rate(t: int) -> float:
        prev, cur = per_gen[t - 1]["best_fitness"], per_gen[t]["best_fitness"]
        return (cur - prev) / max(1.0, abs(prev))

    triggers: dict[str, dict] = {}

    for name, diversity_guard in (("reflector_threshold", True), ("reflector_no_diversity_guard", False)):
        stagnation = 0
        diversity_preemptions = 0
        trigger = None
        for t in range(1, horizon + 1):
            if improvement_rate(t) > tau_rho:
                stagnation = 0
                continue
            stagnation += 1
            if diversity_guard and per_gen[t]["diversity"] < tau_d:
                diversity_preemptions += 1
                continue
            if stagnation >= n_stag:
                trigger = t
                break
        triggers[name] = {"trigger_generation": trigger, "diversity_preemptions": diversity_preemptions}

    stagnation = 0
    trigger = None
    for t in range(1, horizon + 1):
        if per_gen[t]["best_d_min"] > per_gen[t - 1]["best_d_min"]:
            stagnation = 0
            continue
        stagnation += 1
        if stagnation >= n_stag:
            trigger = t
            break
    triggers["scalar_dmin_stag"] = {"trigger_generation": trigger}

    triggers["fixed_25"] = {"trigger_generation": 25 if horizon >= 25 else None}
    triggers["fixed_50"] = {"trigger_generation": 50 if horizon >= 50 else None}
    rng = np.random.default_rng(90_000 + run_seed)
    triggers["random_uniform"] = {"trigger_generation": int(rng.integers(1, horizon + 1))}
    return triggers


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload, resolved, error = _load_hybrid_artifact(
        artifact_dir=Path(args.artifact_path).parent,
        artifact_path=Path(args.artifact_path),
    )
    if error or payload is None:
        raise SystemExit(error or f"could not load hybrid artifact from {resolved}")
    seed_matrix = _matrix_from_repair_payload(payload.get("repair_replay") or {})

    repair_config = RepairPolicyConfig(timeout_sec=args.repair_timeout_sec)
    all_parity = list(range(TARGET_K, TARGET_N))

    records = []
    repair_cache: dict[tuple[int, int], dict] = {}
    for run_idx in range(args.runs):
        run_seed = args.seed_start + run_idx
        traj = ga_trajectory(
            seed_matrix,
            run_seed=run_seed,
            population_size=args.population,
            horizon=args.horizon,
        )
        gen_ms = traj["ga_elapsed_ms"] / max(1, args.horizon)
        tau_d = REFLECTOR_THRESHOLDS["tau_D"]
        collapse_gen = next(
            (g["generation"] for g in traj["per_gen"] if g["diversity"] < tau_d),
            None,
        )
        triggers = replay_triggers(traj["per_gen"], horizon=args.horizon, run_seed=run_seed)
        for policy_name, info in triggers.items():
            trigger = info.get("trigger_generation")
            row = {
                "run_seed": run_seed,
                "policy": policy_name,
                "trigger_generation": trigger,
                "diversity_preemptions": info.get("diversity_preemptions"),
                "diversity_collapse_generation": collapse_gen,
                "initial_best_d_min": traj["per_gen"][0]["best_d_min"],
                "ga_ms_per_generation": round(gen_ms, 3),
            }
            if trigger is not None:
                key = (run_seed, trigger)
                if key not in repair_cache:
                    repair_cache[key] = repair_with_columns(
                        traj["best_matrices"][trigger],
                        all_parity,
                        config=repair_config,
                        cadical_path=args.cadical_path,
                        workdir=out_dir / "scratch",
                        case_id=f"seed{run_seed}_gen{trigger}",
                    )
                repair = repair_cache[key]
                row.update(
                    {
                        "candidate_d_min_at_trigger": traj["per_gen"][trigger]["best_d_min"],
                        "repair_success": bool(repair.get("target_achieved")),
                        "repair_elapsed_ms": repair.get("elapsed_ms"),
                        "end_to_end_ms": round(trigger * gen_ms + float(repair.get("elapsed_ms") or 0.0), 3),
                    }
                )
            records.append(row)
        print(
            f"[reflector-replay] seed {run_seed}: "
            + ", ".join(f"{k}={v.get('trigger_generation')}" for k, v in triggers.items()),
            file=sys.stderr,
        )

    summaries = []
    for policy_name in sorted({row["policy"] for row in records}):
        rows = [row for row in records if row["policy"] == policy_name]
        triggered = [row for row in rows if row["trigger_generation"] is not None]
        succeeded = [row for row in triggered if row.get("repair_success")]
        e2e = sorted(row["end_to_end_ms"] for row in succeeded)
        summaries.append(
            {
                "policy": policy_name,
                "runs": len(rows),
                "triggered": len(triggered),
                "repair_success": len(succeeded),
                "median_trigger_generation": statistics.median(r["trigger_generation"] for r in triggered) if triggered else None,
                "median_end_to_end_ms": statistics.median(e2e) if e2e else None,
                "median_repair_ms": statistics.median(r["repair_elapsed_ms"] for r in succeeded) if succeeded else None,
            }
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Decision-level replay of repair-trigger policies on matched deterministic GA trajectories; trigger candidates verified by actual all-parity SAT repair.",
        "args": vars(args),
        "thresholds": dict(REFLECTOR_THRESHOLDS),
        "summaries": summaries,
    }
    (out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
