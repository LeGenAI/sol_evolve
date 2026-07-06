#!/usr/bin/env python3
"""Regenerate matched-seed Hybrid SAT-GA repair ablation logs.

The output is deliberately raw-first: each seed/variant record is written to
JSONL before aggregate summaries are computed.  The design separates the GA
trajectory from the SAT-repair policy, so each repair variant is evaluated on
the same candidate produced by the same random seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.contracts import HybridGAPolicy  # noqa: E402
from solevolve.hybrid_ga import (  # noqa: E402
    HYBRID_CLAIM_ID,
    _candidate_id,
    _load_hybrid_artifact,
    _matrix_from_repair_payload,
    attempt_sat_repair,
    evolve_population,
    final_verify_hybrid_ga,
    initialize_population,
)


VARIANT_REPAIR_STRATEGIES: dict[str, str | None] = {
    "ga_only": None,
    "random_mask_repair": "random_support_size_mask",
    "all_parity_repair": "all_parity_columns",
    "support_repair_only": "low_weight_support_mask_only",
    "support_repair_fallback": "low_weight_support_mask_with_fallback",
    "support_repair": "low_weight_support_mask_with_fallback",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200, help="Number of matched seeds per variant.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value.")
    parser.add_argument(
        "--variants",
        default="ga_only,random_mask_repair,all_parity_repair,support_repair_only,support_repair_fallback",
        help=(
            "Comma-separated variants: ga_only, random_mask_repair, all_parity_repair, "
            "support_repair_only, support_repair_fallback."
        ),
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--repair-interval", type=int, default=50)
    parser.add_argument("--repair-candidates", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--target-d", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N matched seeds.")
    parser.add_argument(
        "--keep-cnf",
        action="store_true",
        help="Keep per-repair CNF scratch files. By default they are removed after each repair record.",
    )
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument(
        "--artifact-path",
        default=str(REPO_ROOT / "artifacts" / "reviewer_core_claims.json"),
        help="Reviewer core-claims JSON containing the archived pre-repair seed.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "raw_regenerated_hybrid_repair_matched"),
        help="Directory for raw JSONL/CSV/summary outputs.",
    )
    return parser.parse_args()


def load_pre_repair_seed(artifact_path: Path) -> list[list[int]]:
    payload, resolved, error = _load_hybrid_artifact(
        artifact_dir=artifact_path.parent,
        artifact_path=artifact_path,
    )
    if error or payload is None:
        raise RuntimeError(error or f"Could not load {HYBRID_CLAIM_ID} from {resolved}")
    repair_replay = payload.get("repair_replay")
    if not isinstance(repair_replay, dict):
        raise RuntimeError("reviewer_core_claims.json is missing repair_replay for Hybrid SAT-GA")
    return _matrix_from_repair_payload(repair_replay)


def json_ready(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return json_ready(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (centre - spread) / denom, (centre + spread) / denom


def exact_mcnemar_p(discordant_success: int, discordant_failure: int) -> float:
    trials = discordant_success + discordant_failure
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, idx) for idx in range(0, min(discordant_success, discordant_failure) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def base_policy(args: argparse.Namespace, *, seed: int, repair_strategy: str = "none") -> HybridGAPolicy:
    return HybridGAPolicy(
        enabled=True,
        mode="replay",
        seed=seed,
        population=args.population,
        generations=args.generations,
        repair_interval=args.repair_interval,
        timeout_sec=args.timeout_sec,
        target_distance=args.target_d,
        solver_preference="cadical",
        repair_strategy=repair_strategy,
    )


def below_target_count(diagnostics: dict[str, Any], target_d: int) -> int:
    distribution = diagnostics.get("weight_distribution") or {}
    total = 0
    for weight, count in distribution.items():
        weight_int = int(weight)
        if 0 < weight_int < target_d:
            total += int(count)
    return total


def run_base_evolution(args: argparse.Namespace, seed_matrix: list[list[int]], seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    policy = base_policy(args, seed=seed, repair_strategy="none")
    init = initialize_population(seed_matrix=seed_matrix, policy=policy)
    evolution = evolve_population(
        population=init["population"],
        policy=policy,
        target_d=args.target_d,
        expected_a_d=None,
    )
    best_matrix = evolution["best_matrix"]
    final = final_verify_hybrid_ga(
        matrix=best_matrix,
        target_d=args.target_d,
        expected_a_d=None,
    )
    repair_candidates = list(evolution.get("repair_candidates") or [])
    if not repair_candidates:
        repair_candidates = [
            {
                "generation": args.generations,
                "candidate_id": _candidate_id(best_matrix),
                "matrix": best_matrix,
                "summary": evolution["generation_summaries"][-1],
            }
        ]
    return {
        "policy": policy,
        "seed_status": init["seed_status"],
        "initial_diversity": init["diversity"],
        "generation_summaries": evolution["generation_summaries"],
        "repair_candidates": repair_candidates,
        "best_matrix": best_matrix,
        "final_diagnostics": final,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def ga_only_record(args: argparse.Namespace, base: dict[str, Any], seed: int, run_idx: int) -> dict[str, Any]:
    final = base["final_diagnostics"]
    return {
        "variant": "ga_only",
        "seed": seed,
        "run_index": run_idx,
        "config": base["policy"].model_dump(mode="python"),
        "seed_status": base["seed_status"],
        "initial_diversity": base["initial_diversity"],
        "generation_summaries": base["generation_summaries"],
        "repair_events": [],
        "final_diagnostics": final,
        "target_achieved": bool(final.get("target_achieved")),
        "below_target_count": below_target_count(final, args.target_d),
        "elapsed_ms": base["elapsed_ms"],
    }


def repair_record(
    args: argparse.Namespace,
    *,
    base: dict[str, Any],
    seed: int,
    run_idx: int,
    variant: str,
    repair_strategy: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    policy = base_policy(args, seed=seed, repair_strategy=repair_strategy)
    best_matrix = base["best_matrix"]
    final = base["final_diagnostics"]
    repair_events = []

    if not final.get("target_achieved") and args.repair_interval:
        for candidate in base["repair_candidates"][: args.repair_candidates]:
            matrix = candidate.get("matrix") if isinstance(candidate.get("matrix"), list) else best_matrix
            scratch_dir = Path(args.out_dir) / "scratch" / variant / f"seed_{seed}"
            event = attempt_sat_repair(
                candidate=matrix,
                generation=int(candidate.get("generation") or args.generations),
                policy=policy,
                cadical_path=args.cadical_path,
                artifact_dir=scratch_dir,
                claim_id=HYBRID_CLAIM_ID,
                target_d=args.target_d,
                expected_a_d=None,
            )
            if not args.keep_cnf:
                shutil.rmtree(scratch_dir, ignore_errors=True)
            repair_events.append(event)
            if event.post_repair_matrix is not None:
                best_matrix = event.post_repair_matrix
                final = final_verify_hybrid_ga(matrix=best_matrix, target_d=args.target_d, expected_a_d=None)
                if final.get("target_achieved"):
                    break
            if event.solver_status == "TIMEOUT":
                break

    return {
        "variant": variant,
        "seed": seed,
        "run_index": run_idx,
        "config": policy.model_dump(mode="python"),
        "seed_status": base["seed_status"],
        "initial_diversity": base["initial_diversity"],
        "generation_summaries": base["generation_summaries"],
        "repair_events": [json_ready(event) for event in repair_events],
        "final_diagnostics": final,
        "target_achieved": bool(final.get("target_achieved")),
        "below_target_count": below_target_count(final, args.target_d),
        "elapsed_ms": round(base["elapsed_ms"] + (time.perf_counter() - started) * 1000, 3),
    }


def summarize(records: list[dict[str, Any]], target_d: int) -> list[dict[str, Any]]:
    rows = []
    variants = sorted({record["variant"] for record in records})
    for variant in variants:
        subset = [record for record in records if record["variant"] == variant]
        successes = sum(1 for record in subset if record.get("target_achieved"))
        low, high = wilson_interval(successes, len(subset))
        statuses = Counter(
            event.get("solver_status")
            for record in subset
            for event in (record.get("repair_events") or [])
            if isinstance(event, dict)
        )
        final_distances = [
            int((record.get("final_diagnostics") or {}).get("minimum_distance") or 0)
            for record in subset
        ]
        below_counts = [int(record.get("below_target_count") or 0) for record in subset]
        elapsed = [float(record.get("elapsed_ms") or 0.0) for record in subset]
        rows.append(
            {
                "variant": variant,
                "successes": successes,
                "total": len(subset),
                "rate": successes / len(subset) if subset else 0.0,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "repair_attempts": sum(statuses.values()),
                "repair_sat": statuses.get("SAT", 0),
                "repair_timeout": statuses.get("TIMEOUT", 0),
                "repair_unsat": statuses.get("UNSAT", 0),
                "mean_final_d_min": statistics.fmean(final_distances) if final_distances else 0.0,
                "mean_below_target_count": statistics.fmean(below_counts) if below_counts else 0.0,
                "median_below_target_count": statistics.median(below_counts) if below_counts else 0.0,
                "mean_elapsed_ms": statistics.fmean(elapsed) if elapsed else 0.0,
                "target_d": target_d,
            }
        )
    return rows


def paired_comparisons(records: list[dict[str, Any]], reference: str = "ga_only") -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, bool]] = defaultdict(dict)
    for record in records:
        by_seed[int(record["seed"])][record["variant"]] = bool(record.get("target_achieved"))

    variants = sorted({record["variant"] for record in records if record["variant"] != reference})
    rows = []
    for variant in variants:
        paired = [outcomes for outcomes in by_seed.values() if reference in outcomes and variant in outcomes]
        ref_successes = sum(1 for outcomes in paired if outcomes[reference])
        var_successes = sum(1 for outcomes in paired if outcomes[variant])
        variant_only = sum(1 for outcomes in paired if outcomes[variant] and not outcomes[reference])
        reference_only = sum(1 for outcomes in paired if outcomes[reference] and not outcomes[variant])
        rows.append(
            {
                "reference": reference,
                "variant": variant,
                "paired_total": len(paired),
                "reference_successes": ref_successes,
                "variant_successes": var_successes,
                "risk_difference": (var_successes - ref_successes) / len(paired) if paired else 0.0,
                "variant_only_successes": variant_only,
                "reference_only_successes": reference_only,
                "mcnemar_exact_p": exact_mcnemar_p(variant_only, reference_only),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unsupported = sorted(set(variants) - set(VARIANT_REPAIR_STRATEGIES))
    if unsupported:
        raise SystemExit(f"Unsupported variants: {', '.join(unsupported)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_runs.jsonl"
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    paired_csv_path = out_dir / "paired_comparisons.csv"
    manifest_path = out_dir / "manifest.json"

    seed_matrix = load_pre_repair_seed(Path(args.artifact_path))
    records: list[dict[str, Any]] = []

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for run_idx in range(args.runs):
            seed = args.seed_start + run_idx
            base = run_base_evolution(args, seed_matrix, seed)
            for variant in variants:
                repair_strategy = VARIANT_REPAIR_STRATEGIES[variant]
                if repair_strategy is None:
                    record = ga_only_record(args, base, seed, run_idx)
                else:
                    record = repair_record(
                        args,
                        base=base,
                        seed=seed,
                        run_idx=run_idx,
                        variant=variant,
                        repair_strategy=repair_strategy,
                    )
                record["created_at"] = datetime.now(timezone.utc).isoformat()
                records.append(record)
                raw_file.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
                raw_file.flush()
            if args.progress_every and (run_idx + 1) % args.progress_every == 0:
                print(
                    f"[matched-ablation] completed {run_idx + 1}/{args.runs} seeds "
                    f"({len(records)} records)",
                    file=sys.stderr,
                    flush=True,
                )

    rows = summarize(records, args.target_d)
    paired_rows = paired_comparisons(records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_id": HYBRID_CLAIM_ID,
        "description": "Matched-seed raw Hybrid SAT-GA ablation records from the current public implementation.",
        "design": {
            "unit": "matched random seed",
            "primary_endpoint": f"verified minimum distance at least {args.target_d}",
            "secondary_endpoints": ["below-target codeword count", "final minimum distance", "repair solver status"],
            "paired_test": "exact McNemar test against ga_only",
            "variants": variants,
        },
        "important_limit": (
            "These records regenerate the current coordinator implementation. They do not by themselves "
            "authenticate the older 05-14 compact 1/50, 2/50, 5/50 summary unless run with the matching "
            "historical configuration and archived per-run logs."
        ),
        "args": vars(args),
        "raw_jsonl": str(raw_path),
        "summary_csv": str(csv_path),
        "paired_comparisons_csv": str(paired_csv_path),
        "rows": rows,
        "paired_comparisons": paired_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_fields = [
        "variant",
        "successes",
        "total",
        "rate",
        "wilson_95_low",
        "wilson_95_high",
        "repair_attempts",
        "repair_sat",
        "repair_timeout",
        "repair_unsat",
        "mean_final_d_min",
        "mean_below_target_count",
        "median_below_target_count",
        "mean_elapsed_ms",
        "target_d",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(rows)

    paired_fields = [
        "reference",
        "variant",
        "paired_total",
        "reference_successes",
        "variant_successes",
        "risk_difference",
        "variant_only_successes",
        "reference_only_successes",
        "mcnemar_exact_p",
    ]
    with paired_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_rows)

    manifest = {
        "created_at": summary["created_at"],
        "files": {
            "raw_runs_jsonl": str(raw_path),
            "summary_json": str(summary_path),
            "summary_csv": str(csv_path),
            "paired_comparisons_csv": str(paired_csv_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
