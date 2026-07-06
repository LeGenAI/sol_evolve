#!/usr/bin/env python3
"""Evaluate fixed repair-policy baselines on a frozen candidate bank."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.baselines.repair_policy_eval import (  # noqa: E402
    RepairPolicyConfig,
    RepairPolicyEvaluator,
    json_ready,
    load_jsonl,
    paired_comparisons,
    write_jsonl,
    write_summary_csv,
)
from solevolve.hybrid_ga import TARGET_D, TARGET_K, TARGET_N  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset" / "candidates.jsonl"),
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--mutable-budget", type=int, default=4)
    parser.add_argument("--timeout-sec", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-cnf", action="store_true")
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument(
        "--policies",
        default="random,low_weight_support,all_parity",
        help="Comma-separated policies: random, low_weight_support, all_parity.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_baseline_pilot"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.dataset)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if args.max_cases is not None:
        records = records[: args.max_cases]
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]

    config = RepairPolicyConfig(
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        mutable_budget=args.mutable_budget,
        timeout_sec=args.timeout_sec,
        keep_cnf=args.keep_cnf,
    )
    evaluator = RepairPolicyEvaluator(
        records,
        config=config,
        cadical_path=args.cadical_path,
        artifact_dir=out_dir,
    )

    all_records = []
    summaries = []
    by_policy = {}
    for policy_name in policies:
        result = evaluator.evaluate_policy(policy_name=policy_name, seed=args.seed)
        by_policy[policy_name] = result["records"]
        for record in result["records"]:
            all_records.append(record)
        summaries.append({"policy": policy_name, **result["summary"]})
        print(f"[baseline] {policy_name}: {result['summary']}", file=sys.stderr)

    paired = paired_comparisons(by_policy, reference="random") if "random" in by_policy else []
    raw_path = out_dir / "raw_policy_records.jsonl"
    summary_path = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    paired_csv = out_dir / "paired_comparisons.json"
    write_jsonl(raw_path, all_records)
    write_summary_csv(summary_csv, summaries)
    paired_csv.write_text(json.dumps(json_ready(paired), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Fixed-policy pilot for the verifier-facing repair-policy baseline interface.",
        "args": vars(args),
        "config": config.__dict__,
        "dataset_size_after_split": len(records),
        "raw_records": str(raw_path),
        "summary_csv": str(summary_csv),
        "summaries": summaries,
        "paired_comparisons": paired,
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

