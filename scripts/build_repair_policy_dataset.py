#!/usr/bin/env python3
"""Build a frozen candidate bank for verifier-facing repair-policy baselines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.baselines.repair_policy_eval import candidate_record, json_ready, write_jsonl  # noqa: E402
from solevolve.contracts import HybridGAPolicy  # noqa: E402
from solevolve.hybrid_ga import (  # noqa: E402
    HYBRID_CLAIM_ID,
    TARGET_D,
    TARGET_K,
    TARGET_N,
    _load_hybrid_artifact,
    _matrix_from_repair_payload,
    evolve_population,
    generate_frontier_seed_bank,
    initialize_population,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=60, help="Number of GA seeds to sample.")
    parser.add_argument("--source-mode", choices=["ga", "perturb", "frontier", "mixed", "corrupt"], default="ga")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--repair-interval", type=int, default=50)
    parser.add_argument("--candidates-per-seed", type=int, default=1)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--validation-frac", type=float, default=0.2)
    parser.add_argument("--include-successful", action="store_true", help="Keep candidates that already meet target distance.")
    parser.add_argument("--min-d", type=int, default=0, help="Minimum candidate d_min to keep.")
    parser.add_argument("--max-d", type=int, default=None, help="Maximum candidate d_min to keep; defaults to target_d - 1.")
    parser.add_argument("--perturb-rate", type=float, default=0.02, help="Parity-bit flip rate for source-mode=perturb|mixed.")
    parser.add_argument(
        "--perturb-rates",
        default=None,
        help="Optional comma-separated parity-bit flip rates. When set, rates are cycled by seed.",
    )
    parser.add_argument(
        "--corrupt-column-counts",
        default="1,2,3,4",
        help="Comma-separated parity-column counts to corrupt for source-mode=corrupt.",
    )
    parser.add_argument(
        "--corrupt-bits-per-column",
        type=int,
        default=1,
        help="Number of row bits flipped in each selected parity column for source-mode=corrupt.",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--frontier-seed-timeout-sec", type=int, default=10)
    parser.add_argument(
        "--artifact-path",
        default=str(REPO_ROOT / "artifacts" / "reviewer_core_claims.json"),
        help="Reviewer core-claims JSON containing the archived pre-repair seed.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset"),
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


def load_final_witness(artifact_path: Path) -> list[list[int]]:
    payload, resolved, error = _load_hybrid_artifact(
        artifact_dir=artifact_path.parent,
        artifact_path=artifact_path,
    )
    if error or payload is None:
        raise RuntimeError(error or f"Could not load {HYBRID_CLAIM_ID} from {resolved}")
    matrix = payload.get("generator_matrix")
    if not isinstance(matrix, list):
        raise RuntimeError(f"reviewer_core_claims.json is missing generator_matrix for {HYBRID_CLAIM_ID}")
    return [[int(value) & 1 for value in row] for row in matrix]


def policy_for_seed(args: argparse.Namespace, seed: int) -> HybridGAPolicy:
    return HybridGAPolicy(
        enabled=True,
        mode="replay",
        seed=seed,
        population=args.population,
        generations=args.generations,
        repair_interval=args.repair_interval,
        timeout_sec=1,
        target_distance=args.target_d,
        solver_preference="cadical",
        repair_strategy="none",
    )


def split_for_index(index: int, total: int, train_frac: float, validation_frac: float) -> str:
    train_cut = int(total * train_frac)
    validation_cut = train_cut + int(total * validation_frac)
    if index < train_cut:
        return "train"
    if index < validation_cut:
        return "validation"
    return "test"


def numeric_summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    sorted_values = sorted(float(value) for value in values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[mid]
    else:
        median = 0.5 * (sorted_values[mid - 1] + sorted_values[mid])
    return {
        "min": sorted_values[0],
        "median": median,
        "mean": sum(sorted_values) / len(sorted_values),
        "max": sorted_values[-1],
    }


def perturb_matrix(matrix: list[list[int]], *, rng: np.random.Generator, rate: float, k: int) -> list[list[int]]:
    arr = np.asarray(matrix, dtype=np.uint8).copy()
    if rate > 0:
        mask = rng.random(arr[:, k:].shape) < rate
        arr[:, k:] ^= mask.astype(np.uint8)
    return arr.astype(int).tolist()


def corrupt_witness(
    matrix: list[list[int]],
    *,
    rng: np.random.Generator,
    k: int,
    n: int,
    column_count: int,
    bits_per_column: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.uint8).copy()
    parity_columns = np.arange(k, n)
    column_count = max(1, min(int(column_count), len(parity_columns)))
    bits_per_column = max(1, min(int(bits_per_column), arr.shape[0]))
    selected_columns = sorted(int(col) for col in rng.choice(parity_columns, size=column_count, replace=False))
    flips: list[dict[str, Any]] = []
    for col in selected_columns:
        selected_rows = sorted(int(row) for row in rng.choice(np.arange(arr.shape[0]), size=bits_per_column, replace=False))
        for row in selected_rows:
            arr[row, col] ^= 1
        flips.append({"column": col, "rows": selected_rows})
    return arr.astype(int).tolist(), {"corrupted_columns": selected_columns, "flips": flips}


def keep_record(record: dict[str, Any], *, args: argparse.Namespace) -> bool:
    max_d = args.max_d if args.max_d is not None else args.target_d - 1
    d_min = int(record.get("d_min") or 0)
    if not args.include_successful and d_min >= args.target_d:
        return False
    return int(args.min_d) <= d_min <= int(max_d)


def add_record(
    records: list[dict[str, Any]],
    seen: set[str],
    matrix: list[list[int]],
    *,
    args: argparse.Namespace,
    source: str,
    seed: int,
    metadata: dict[str, Any],
) -> None:
    record = candidate_record(
        matrix,
        source=source,
        seed=seed,
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        metadata=metadata,
    )
    if record["candidate_id"] in seen:
        return
    if not keep_record(record, args=args):
        return
    seen.add(record["candidate_id"])
    records.append(record)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "candidates.jsonl"
    manifest_path = out_dir / "manifest.json"

    artifact_path = Path(args.artifact_path)
    seed_matrix = load_pre_repair_seed(artifact_path)
    final_witness_matrix = load_final_witness(artifact_path) if args.source_mode == "corrupt" else None
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    perturb_rates = (
        [float(item.strip()) for item in args.perturb_rates.split(",") if item.strip()]
        if args.perturb_rates
        else [float(args.perturb_rate)]
    )
    corrupt_column_counts = [int(item.strip()) for item in args.corrupt_column_counts.split(",") if item.strip()]

    if args.source_mode == "frontier":
        policy = policy_for_seed(args, args.seed_start).model_copy(
            update={
                "frontier_seed_count": args.runs,
                "frontier_seed_timeout_sec": args.frontier_seed_timeout_sec,
                "frontier_distance": max(1, args.target_d - 1),
            }
        )
        result = generate_frontier_seed_bank(
            policy=policy,
            artifact_dir=out_dir / "frontier_seed_bank",
            cadical_path=args.cadical_path,
            n=args.n,
            k=args.k,
            frontier_d=max(1, args.target_d - 1),
        )
        for seed_idx, item in enumerate(result.get("seeds") or []):
            matrix = item.get("matrix")
            if isinstance(matrix, list):
                add_record(
                    records,
                    seen,
                    matrix,
                    args=args,
                    source="frontier_seed_bank_sat",
                    seed=args.seed_start + seed_idx,
                    metadata={
                        "run_index": seed_idx,
                        "frontier_seed_index": seed_idx,
                        "frontier_status": result.get("status"),
                        "frontier_seed_timeout_sec": args.frontier_seed_timeout_sec,
                    },
                )

    loop_runs = 0 if args.source_mode == "frontier" else args.runs
    for run_idx in range(loop_runs):
        seed = args.seed_start + run_idx
        if args.source_mode == "corrupt":
            if final_witness_matrix is None:
                raise RuntimeError("source-mode=corrupt requires a final witness matrix")
            rng = np.random.default_rng(seed)
            column_count = corrupt_column_counts[run_idx % len(corrupt_column_counts)]
            matrix, corruption = corrupt_witness(
                final_witness_matrix,
                rng=rng,
                k=args.k,
                n=args.n,
                column_count=column_count,
                bits_per_column=args.corrupt_bits_per_column,
            )
            add_record(
                records,
                seen,
                matrix,
                args=args,
                source="corrupted_verified_witness",
                seed=seed,
                metadata={
                    "run_index": run_idx,
                    "source_matrix": "verified_final_witness",
                    "corrupt_column_count": column_count,
                    "corrupt_bits_per_column": args.corrupt_bits_per_column,
                    **corruption,
                },
            )
        if args.source_mode == "mixed":
            policy = policy_for_seed(args, seed).model_copy(
                update={
                    "frontier_seed_count": 1,
                    "frontier_seed_timeout_sec": args.frontier_seed_timeout_sec,
                    "frontier_distance": max(1, args.target_d - 1),
                }
            )
            result = generate_frontier_seed_bank(
                policy=policy,
                artifact_dir=out_dir / "frontier_tmp" / f"seed_{seed}",
                cadical_path=args.cadical_path,
                n=args.n,
                k=args.k,
                frontier_d=max(1, args.target_d - 1),
            )
            for seed_idx, item in enumerate(result.get("seeds") or []):
                matrix = item.get("matrix")
                if isinstance(matrix, list):
                    add_record(
                        records,
                        seen,
                        matrix,
                        args=args,
                        source="frontier_seed_bank_sat",
                        seed=seed,
                        metadata={
                            "run_index": run_idx,
                            "frontier_seed_index": seed_idx,
                            "frontier_status": result.get("status"),
                            "frontier_seed_timeout_sec": args.frontier_seed_timeout_sec,
                        },
                    )
        if args.source_mode in {"perturb", "mixed"}:
            rng = np.random.default_rng(seed)
            perturb_rate = perturb_rates[run_idx % len(perturb_rates)]
            matrix = perturb_matrix(seed_matrix, rng=rng, rate=perturb_rate, k=args.k)
            add_record(
                records,
                seen,
                matrix,
                args=args,
                source="perturbed_archived_frontier_seed",
                seed=seed,
                metadata={
                    "run_index": run_idx,
                    "perturb_rate": perturb_rate,
                    "source_matrix": "archived_pre_repair_seed",
                },
            )
        if args.source_mode in {"ga", "mixed"}:
            policy = policy_for_seed(args, seed)
            init = initialize_population(seed_matrix=seed_matrix, policy=policy, n=args.n, k=args.k)
            evolved = evolve_population(
                population=init["population"],
                policy=policy,
                n=args.n,
                k=args.k,
                target_d=args.target_d,
                expected_a_d=None,
            )
            candidates = list(evolved.get("repair_candidates") or [])
            if not candidates:
                candidates = [
                    {
                        "generation": args.generations,
                        "candidate_id": None,
                        "matrix": evolved["best_matrix"],
                        "summary": (evolved.get("generation_summaries") or [{}])[-1],
                    }
                ]
            for candidate_idx, candidate in enumerate(candidates[: args.candidates_per_seed]):
                matrix = candidate.get("matrix")
                if not isinstance(matrix, list):
                    matrix = evolved["best_matrix"]
                add_record(
                    records,
                    seen,
                    matrix,
                    args=args,
                    source="hybrid_ga_replay_candidate_bank",
                    seed=seed,
                    metadata={
                        "run_index": run_idx,
                        "candidate_index": candidate_idx,
                        "generation": candidate.get("generation"),
                        "seed_status": init.get("seed_status"),
                        "initial_diversity": init.get("diversity"),
                        "generation_summary": candidate.get("summary"),
                    },
                )
        if args.progress_every and (run_idx + 1) % args.progress_every == 0:
            print(f"[dataset] processed {run_idx + 1}/{args.runs} seeds, unique={len(records)}", file=sys.stderr)

    total = len(records)
    for idx, record in enumerate(records):
        record["split"] = split_for_index(idx, total, args.train_frac, args.validation_frac)

    write_jsonl(raw_path, records)
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    d_min_counts = Counter(str(record.get("d_min")) for record in records)
    source_counts = Counter(str(record.get("source")) for record in records)
    d_min_by_split: dict[str, dict[str, int]] = {}
    for split_name in sorted(split_counts):
        d_min_by_split[split_name] = dict(
            sorted(
                Counter(str(record.get("d_min")) for record in records if record.get("split") == split_name).items(),
                key=lambda item: item[0],
            )
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Frozen candidate bank for EoH/FunSearch/SolEvolve repair-policy comparison.",
        "claim_id": HYBRID_CLAIM_ID,
        "args": vars(args),
        "candidate_jsonl": str(raw_path),
        "total_candidates": total,
        "split_counts": split_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "d_min_counts": dict(sorted(d_min_counts.items(), key=lambda item: item[0])),
        "d_min_by_split": d_min_by_split,
        "low_weight_count_summary": numeric_summary([int(record.get("low_weight_count") or 0) for record in records]),
        "below_target_count_summary": numeric_summary([int(record.get("below_target_count") or 0) for record in records]),
        "target": {"n": args.n, "k": args.k, "d": args.target_d},
        "important_limits": [
            "Generated policies may train only on split=train and tune on split=validation.",
            "Final paper comparisons must be reported on split=test.",
            "This dataset stores candidate matrices because the deterministic verifier is the evaluation authority.",
        ],
    }
    manifest_path.write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
