#!/usr/bin/env python3
"""Evaluate natural-size (unbounded) repair-mask policies on a frozen candidate bank.

Unlike the top-B scoring interface, these policies use variable-size mutable sets,
matching how the SolEvolve pipeline actually deploys the low-weight-support mask:

- support_natural: all parity columns touching any below-target codeword.
- random_matched: uniformly random parity subset of the same size as support_natural.
- support_fallback: support_natural first, then all-parity repair if that fails
  (the pipeline's default two-phase plan).
- all_parity: every parity column (control endpoint).
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
    json_ready,
    load_jsonl,
    low_weight_codewords,
    repair_with_columns,
    summarize_records,
    write_jsonl,
)
from solevolve.hybrid_ga import TARGET_D, TARGET_K, TARGET_N  # noqa: E402

POLICIES = ("support_natural", "random_matched", "support_fallback", "all_parity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset_corrupt_main" / "candidates.jsonl"),
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def natural_mask(matrix: list[list[int]], *, n: int, k: int, target_d: int) -> list[int]:
    lows = low_weight_codewords(matrix, n=n, k=k, target_d=target_d)
    columns: set[int] = set()
    for item in lows:
        columns.update(int(col) for col in item["parity_support"])
    return sorted(columns) or list(range(k, n))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_jsonl(args.dataset)
    if args.split != "all":
        candidates = [row for row in candidates if row.get("split") == args.split]
    if args.max_cases is not None:
        candidates = candidates[: args.max_cases]
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise SystemExit(f"unknown policies: {sorted(unknown)}")

    config = RepairPolicyConfig(
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        mutable_budget=args.n - args.k,
        timeout_sec=args.timeout_sec,
    )
    all_parity = list(range(args.k, args.n))

    all_records = []
    summaries = []
    for policy_name in policies:
        records = []
        for idx, candidate in enumerate(candidates):
            case_seed = args.seed + idx * 9973
            try:
                base_mask = natural_mask(candidate["matrix"], n=args.n, k=args.k, target_d=args.target_d)
                if policy_name in {"support_natural", "support_fallback"}:
                    columns = base_mask
                elif policy_name == "random_matched":
                    rng = np.random.default_rng(case_seed)
                    columns = sorted(int(c) for c in rng.choice(all_parity, size=len(base_mask), replace=False))
                elif policy_name == "all_parity":
                    columns = all_parity
                repair = repair_with_columns(
                    candidate["matrix"],
                    columns,
                    config=config,
                    cadical_path=args.cadical_path,
                    workdir=out_dir / "scratch" / policy_name,
                    case_id=f"{policy_name}_{idx:05d}_{candidate['candidate_id'][:10]}",
                )
                fallback_repair = None
                if policy_name == "support_fallback" and not repair.get("target_achieved"):
                    fallback_repair = repair_with_columns(
                        candidate["matrix"],
                        all_parity,
                        config=config,
                        cadical_path=args.cadical_path,
                        workdir=out_dir / "scratch" / policy_name,
                        case_id=f"{policy_name}_fb_{idx:05d}_{candidate['candidate_id'][:10]}",
                    )
                error = None
            except Exception as exc:
                repair = {
                    "solver_status": "ERROR",
                    "target_achieved": False,
                    "elapsed_ms": 0.0,
                    "mutable_columns": [],
                    "post_diagnostics": None,
                }
                fallback_repair = None
                error = f"{type(exc).__name__}: {exc}"

            final = fallback_repair if (fallback_repair and fallback_repair.get("target_achieved")) else repair
            elapsed_total = float(repair.get("elapsed_ms") or 0.0) + float((fallback_repair or {}).get("elapsed_ms") or 0.0)
            records.append(
                {
                    "policy": policy_name,
                    "case_index": idx,
                    "candidate_id": candidate.get("candidate_id"),
                    "seed": candidate.get("seed"),
                    "split": candidate.get("split"),
                    "pre_d_min": candidate.get("d_min"),
                    "pre_below_target_count": candidate.get("below_target_count"),
                    "pre_low_weight_count": candidate.get("low_weight_count"),
                    "repair": repair,
                    "fallback_repair": fallback_repair,
                    "used_fallback": bool(fallback_repair is not None),
                    "target_achieved": bool(final.get("target_achieved")),
                    "solver_status": final.get("solver_status"),
                    "elapsed_ms": elapsed_total,
                    "mutable_column_count": final.get("mutable_column_count", len(final.get("mutable_columns") or [])),
                    "error": error,
                }
            )
            print(
                f"[natural] {policy_name} case {idx + 1}/{len(candidates)} "
                f"cols={records[-1]['mutable_column_count']} status={final.get('solver_status')} "
                f"success={final.get('target_achieved')}",
                file=sys.stderr,
            )
        all_records.extend(records)
        summaries.append({"policy": policy_name, **summarize_records(records)})

    raw_path = out_dir / "raw_policy_records.jsonl"
    write_jsonl(raw_path, all_records)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Natural-size repair-mask policies (unbounded support mask, matched random, fallback plan, all-parity).",
        "args": vars(args),
        "config": config.__dict__,
        "candidate_count": len(candidates),
        "summaries": summaries,
        "raw_records": str(raw_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready({"summaries": summaries}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
