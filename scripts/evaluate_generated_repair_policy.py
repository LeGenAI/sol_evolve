#!/usr/bin/env python3
"""Evaluate an EoH/FunSearch-generated repair-policy function on a frozen split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.baselines.repair_policy_eval import (  # noqa: E402
    RepairPolicyConfig,
    RepairPolicyEvaluator,
    json_ready,
    load_jsonl,
    write_jsonl,
)
from solevolve.hybrid_ga import TARGET_D, TARGET_K, TARGET_N  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset_stratified_pilot" / "candidates.jsonl"),
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--policy-json", required=True, help="EoH/FunSearch JSON containing a `code` field.")
    parser.add_argument("--policy-name", default="generated_policy")
    parser.add_argument("--mutable-budget", type=int, default=6)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "generated_repair_policy_eval"),
    )
    return parser.parse_args()


def load_policy(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{path} does not contain a non-empty `code` field")
    namespace: dict[str, Any] = {"np": np, "__builtins__": __builtins__}
    exec(compile(code, str(path), "exec"), namespace)
    func = namespace.get("score_columns")
    if not callable(func):
        raise ValueError("generated code did not define callable score_columns")
    return payload, func


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(args.dataset)
    if args.split != "all":
        candidates = [row for row in candidates if row.get("split") == args.split]
    if args.max_cases is not None:
        candidates = candidates[: args.max_cases]
    policy_payload, policy_func = load_policy(args.policy_json)
    config = RepairPolicyConfig(
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        mutable_budget=args.mutable_budget,
        timeout_sec=args.timeout_sec,
        expected_a_d=None,
    )
    evaluator = RepairPolicyEvaluator(
        candidates,
        config=config,
        cadical_path=args.cadical_path,
        artifact_dir=out_dir,
    )
    result = evaluator.evaluate_policy(policy_name=args.policy_name, policy=policy_func, seed=2026)
    raw_path = out_dir / "raw_policy_records.jsonl"
    summary_path = out_dir / "summary.json"
    write_jsonl(raw_path, result["records"])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Held-out evaluation for generated repair-policy code.",
        "args": vars(args),
        "policy_algorithm": policy_payload.get("algorithm"),
        "policy_objective": policy_payload.get("objective"),
        "config": config.__dict__,
        "candidate_count": len(candidates),
        "summary": result["summary"],
        "raw_records": str(raw_path),
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

