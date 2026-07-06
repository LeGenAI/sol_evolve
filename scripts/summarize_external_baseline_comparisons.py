#!/usr/bin/env python3
"""Aggregate held-out repair-policy runs into paired statistics for the manuscript.

Pairs per-case outcomes across policy runs by candidate_id, then reports per-policy
Wilson intervals plus exact McNemar tests with Holm correction over all pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from solevolve.baselines.repair_policy_eval import (  # noqa: E402
    exact_mcnemar_p,
    json_ready,
    load_jsonl,
    wilson_interval,
)

EXTERNAL_DIR = REPO_ROOT / "artifacts" / "external_baselines"

DEFAULT_RUNS = [
    ("random", EXTERNAL_DIR / "corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", "random"),
    ("low_weight_support", EXTERNAL_DIR / "corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", "low_weight_support"),
    ("funsearch_s12", EXTERNAL_DIR / "funsearch_corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", None),
    ("funsearch_s20", EXTERNAL_DIR / "funsearch_s20_corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", None),
    ("eoh_s20", EXTERNAL_DIR / "eoh_corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", None),
    ("oracle", EXTERNAL_DIR / "oracle_corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", None),
    ("all_parity", EXTERNAL_DIR / "corrupt_main_test_budget4_t10" / "raw_policy_records.jsonl", "all_parity"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(EXTERNAL_DIR / "corrupt_main_test_paired_stats"),
    )
    return parser.parse_args()


def load_outcomes(path: Path, policy_filter: str | None) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for record in load_jsonl(path):
        if policy_filter is not None and record.get("policy") != policy_filter:
            continue
        outcomes[str(record["candidate_id"])] = bool(record.get("target_achieved"))
    return outcomes


def holm_correction(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["mcnemar_exact_p"])
    m = len(ordered)
    running_max = 0.0
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * row["mcnemar_exact_p"])
        running_max = max(running_max, adjusted)
        row["mcnemar_holm_adjusted_p"] = running_max
        row["significant_at_0_05_holm"] = running_max < 0.05


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies: dict[str, dict[str, bool]] = {}
    for label, path, policy_filter in DEFAULT_RUNS:
        if not path.exists():
            print(f"[stats] skipping {label}: {path} not found", file=sys.stderr)
            continue
        policies[label] = load_outcomes(path, policy_filter)

    common_ids = None
    for outcomes in policies.values():
        ids = set(outcomes)
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids or [])
    if not common_ids:
        raise SystemExit("no common candidate_ids across runs")

    summaries = []
    for label, outcomes in policies.items():
        successes = sum(1 for cid in common_ids if outcomes[cid])
        low, high = wilson_interval(successes, len(common_ids))
        summaries.append(
            {
                "policy": label,
                "successes": successes,
                "total": len(common_ids),
                "success_rate": successes / len(common_ids),
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )

    pairwise = []
    for label_a, label_b in combinations(policies, 2):
        a_outcomes = policies[label_a]
        b_outcomes = policies[label_b]
        a_only = sum(1 for cid in common_ids if a_outcomes[cid] and not b_outcomes[cid])
        b_only = sum(1 for cid in common_ids if b_outcomes[cid] and not a_outcomes[cid])
        a_total = sum(1 for cid in common_ids if a_outcomes[cid])
        b_total = sum(1 for cid in common_ids if b_outcomes[cid])
        pairwise.append(
            {
                "policy_a": label_a,
                "policy_b": label_b,
                "paired_total": len(common_ids),
                "a_successes": a_total,
                "b_successes": b_total,
                "a_only_successes": a_only,
                "b_only_successes": b_only,
                "risk_difference_a_minus_b": (a_total - b_total) / len(common_ids),
                "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
            }
        )
    holm_correction(pairwise)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Paired held-out statistics across bounded repair policies, the oracle ceiling, and the full-repair control.",
        "paired_candidate_count": len(common_ids),
        "runs": {label: str(path) for label, path, _ in DEFAULT_RUNS if label in policies},
        "policy_summaries": summaries,
        "pairwise_mcnemar_holm": sorted(pairwise, key=lambda row: row["mcnemar_exact_p"]),
    }
    out_path = out_dir / "paired_stats.json"
    out_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
