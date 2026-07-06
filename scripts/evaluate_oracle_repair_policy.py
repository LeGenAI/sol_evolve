#!/usr/bin/env python3
"""Evaluate an oracle repair policy that knows each candidate's corrupted parity columns.

The oracle establishes the achievable ceiling for the bounded (B=4) repair-policy
interface: it scores the columns recorded in `metadata.corrupted_columns` as 1.0 and
all other parity columns as 0.0, then routes selection and repair through the same
`select_top_budget` / `repair_with_columns` pathway used by every other policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.baselines.repair_policy_eval import (  # noqa: E402
    RepairPolicyConfig,
    column_feature_table,
    json_ready,
    load_jsonl,
    repair_with_columns,
    select_top_budget,
    summarize_records,
    write_jsonl,
)
from solevolve.hybrid_ga import TARGET_D, TARGET_K, TARGET_N  # noqa: E402

POLICY_NAME = "oracle_corrupted_columns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset_corrupt_main" / "candidates.jsonl"),
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--mutable-budget", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "oracle_corrupt_main_test_budget4_t10"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(args.dataset)
    if args.split != "all":
        candidates = [row for row in candidates if row.get("split") == args.split]
    if args.max_cases is not None:
        candidates = candidates[: args.max_cases]

    config = RepairPolicyConfig(
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        mutable_budget=args.mutable_budget,
        timeout_sec=args.timeout_sec,
    )

    records = []
    for idx, candidate in enumerate(candidates):
        corrupted = sorted(int(col) for col in (candidate.get("metadata") or {}).get("corrupted_columns", []))
        try:
            if not corrupted:
                raise ValueError("candidate has no metadata.corrupted_columns; oracle undefined")
            features, _global_features, parity_columns, feature_meta = column_feature_table(
                candidate["matrix"],
                n=config.n,
                k=config.k,
                target_d=config.target_d,
            )
            del features
            scores = np.array([1.0 if col in corrupted else 0.0 for col in parity_columns], dtype=float)
            columns = select_top_budget(scores, parity_columns, config.mutable_budget)
            repair = repair_with_columns(
                candidate["matrix"],
                columns,
                config=config,
                cadical_path=args.cadical_path,
                workdir=out_dir / "scratch" / POLICY_NAME,
                case_id=f"{POLICY_NAME}_{idx:05d}_{candidate['candidate_id'][:10]}",
            )
            error = None
        except Exception as exc:
            feature_meta = {}
            repair = {
                "solver_status": "ERROR",
                "target_achieved": False,
                "elapsed_ms": 0.0,
                "mutable_columns": [],
                "post_diagnostics": None,
            }
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "policy": POLICY_NAME,
                "case_index": idx,
                "candidate_id": candidate.get("candidate_id"),
                "seed": candidate.get("seed"),
                "split": candidate.get("split"),
                "pre_d_min": candidate.get("d_min"),
                "pre_below_target_count": candidate.get("below_target_count"),
                "pre_low_weight_count": candidate.get("low_weight_count"),
                "oracle_corrupted_columns": corrupted,
                "feature_meta": feature_meta,
                "repair": repair,
                "target_achieved": bool(repair.get("target_achieved")),
                "solver_status": repair.get("solver_status"),
                "elapsed_ms": repair.get("elapsed_ms"),
                "mutable_column_count": repair.get("mutable_column_count", len(repair.get("mutable_columns") or [])),
                "error": error,
            }
        )
        print(
            f"[oracle] case {idx + 1}/{len(candidates)} "
            f"{candidate['candidate_id'][:10]} corrupted={corrupted} "
            f"status={repair.get('solver_status')} success={repair.get('target_achieved')}",
            file=sys.stderr,
        )

    raw_path = out_dir / "raw_policy_records.jsonl"
    summary_path = out_dir / "summary.json"
    write_jsonl(raw_path, records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Oracle ceiling for the bounded repair-policy interface: mutable set seeded from metadata.corrupted_columns, padded to the budget via the shared top-B selection pathway.",
        "args": vars(args),
        "config": config.__dict__,
        "candidate_count": len(candidates),
        "summary": summarize_records(records),
        "raw_records": str(raw_path),
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
