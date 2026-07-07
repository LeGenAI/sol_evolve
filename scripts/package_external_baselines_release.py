#!/usr/bin/env python3
"""Assemble the reviewer-facing release bundle for the common-interface baseline study.

Copies the compact, deterministic inputs and outputs of the FunSearch/EoH
repair-policy comparison into `artifacts/external_baselines/release/` and writes
SHA-256 checksums. Large regenerable artifacts (raw per-case records, CNF scratch)
stay out of the bundle; the docs describe how to regenerate them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = REPO_ROOT / "artifacts" / "external_baselines"
RELEASE_DIR = EXTERNAL_DIR / "release"

# (source relative to EXTERNAL_DIR, destination relative to RELEASE_DIR)
BUNDLE = [
    # Benchmark dataset (deterministic rebuild: build_repair_policy_dataset.py, seeds in manifest)
    ("repair_policy_dataset_corrupt_main/manifest.json", "dataset/manifest.json"),
    ("repair_policy_dataset_corrupt_main/candidates.jsonl", "dataset/candidates.jsonl"),
    # Evolution run manifests (config, sample budgets, LLM pin) and frozen best policies
    ("eoh_repair_policy_corrupt_main/run_manifest.json", "runs/eoh_run_manifest.json"),
    ("eoh_repair_policy_corrupt_main/results/samples/samples_best.json", "policies/eoh_samples_best.json"),
    ("funsearch_repair_policy_corrupt_main/run_manifest.json", "runs/funsearch_s12_run_manifest.json"),
    ("funsearch_repair_policy_corrupt_main/samples_best.json", "policies/funsearch_s12_samples_best.json"),
    ("funsearch_repair_policy_corrupt_main_s20/run_manifest.json", "runs/funsearch_s20_run_manifest.json"),
    ("funsearch_repair_policy_corrupt_main_s20/samples_best.json", "policies/funsearch_s20_samples_best.json"),
    # Held-out test summaries
    ("corrupt_main_test_budget4_t10/summary.json", "results/fixed_policies_test_summary.json"),
    ("corrupt_main_test_budget4_t10/paired_comparisons.json", "results/fixed_policies_paired_comparisons.json"),
    ("eoh_corrupt_main_test_budget4_t10/summary.json", "results/eoh_test_summary.json"),
    ("funsearch_corrupt_main_test_budget4_t10/summary.json", "results/funsearch_s12_test_summary.json"),
    ("funsearch_s20_corrupt_main_test_budget4_t10/summary.json", "results/funsearch_s20_test_summary.json"),
    ("oracle_corrupt_main_test_budget4_t10/summary.json", "results/oracle_test_summary.json"),
    # Aggregated paired statistics (Wilson + exact McNemar + Holm)
    ("corrupt_main_test_paired_stats/paired_stats.json", "results/paired_stats.json"),
    # Constrained repair-localization sweeps (budget axis at T=10s, timeout axis for oracle/all-parity)
    ("corrupt_main_test_budget2_t10/summary.json", "results/localization/fixed_policies_b2_t10.json"),
    ("corrupt_main_test_budget6_t10/summary.json", "results/localization/fixed_policies_b6_t10.json"),
    ("corrupt_main_test_budget8_t10/summary.json", "results/localization/fixed_policies_b8_t10.json"),
    ("oracle_corrupt_main_test_budget2_t10/summary.json", "results/localization/oracle_b2_t10.json"),
    ("oracle_corrupt_main_test_budget6_t10/summary.json", "results/localization/oracle_b6_t10.json"),
    ("oracle_corrupt_main_test_budget8_t10/summary.json", "results/localization/oracle_b8_t10.json"),
    ("corrupt_main_test_allparity_t05/summary.json", "results/localization/all_parity_t05.json"),
    ("corrupt_main_test_allparity_t2/summary.json", "results/localization/all_parity_t2.json"),
    ("corrupt_main_test_allparity_t5/summary.json", "results/localization/all_parity_t5.json"),
    ("corrupt_main_test_allparity_t30/summary.json", "results/localization/all_parity_t30.json"),
    ("oracle_corrupt_main_test_budget4_t05/summary.json", "results/localization/oracle_b4_t05.json"),
    ("oracle_corrupt_main_test_budget4_t2/summary.json", "results/localization/oracle_b4_t2.json"),
    ("oracle_corrupt_main_test_budget4_t5/summary.json", "results/localization/oracle_b4_t5.json"),
    ("oracle_corrupt_main_test_budget4_t30/summary.json", "results/localization/oracle_b4_t30.json"),
    # A_d objective ablation (matched-seed GA, deterministic)
    ("ad_objective_43_10_16_n50_p100_g100/summary.json", "results/ad_objective/43_10_16_summary.json"),
    ("ad_objective_43_10_16_n50_p100_g100/records.jsonl", "results/ad_objective/43_10_16_records.jsonl"),
    ("ad_objective_22_11_7_n50_p100_g100/summary.json", "results/ad_objective/22_11_7_summary.json"),
    ("ad_objective_22_11_7_n50_p100_g100/records.jsonl", "results/ad_objective/22_11_7_records.jsonl"),
    # Encoding-choice ablation
    ("encoding_ablation_main/summary.json", "results/encoding/summary.json"),
    # Monte Carlo power analysis quoted in the manuscript (seeded, deterministic)
    ("power_analysis_mcnemar/summary.json", "results/power_analysis_mcnemar.json"),
    # Binary SO embeddings of the SO table (reconstruction + independent verification)
    ("so_f2_reconstruction/summary.json", "results/so_f2/summary.json"),
    # Canonical [52,26] d=8 witness: SAT + verifier-feedback (CEGAR) construction
    ("so_52_26_d8_sat_cegar/summary.json", "results/so_f2/so_52_26_d8_sat_cegar_summary.json"),
    ("so_52_26_d8_sat_cegar/so_52_26_d8_matrix.json", "results/so_f2/so_52_26_d8_matrix.json"),
    # Auxiliary independent check: Gram-preserving annealing also reaches d=8
    ("so_52_26_d8_search/summary.json", "results/so_f2/so_52_26_d8_annealing_summary.json"),
    # Lucas partition feasibility incl. the UNSAT rows
    ("lucas_partition_feasibility/summary.json", "results/lucas_partition_feasibility.json"),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    copied = []
    missing = []
    for src_rel, dst_rel in BUNDLE:
        src = EXTERNAL_DIR / src_rel
        dst = RELEASE_DIR / dst_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(
            {
                "path": dst_rel,
                "source": f"artifacts/external_baselines/{src_rel}",
                "sha256": sha256_of(dst),
                "bytes": dst.stat().st_size,
            }
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Reviewer-facing bundle for the common-interface FunSearch/EoH repair-policy benchmark.",
        "reproduction_doc": "docs/external_baselines_reproduction.md",
        "files": copied,
        "missing_at_packaging_time": missing,
    }
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASE_DIR / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[release] copied {len(copied)} files to {RELEASE_DIR}")
    for item in missing:
        print(f"[release] WARNING missing: {item}", file=sys.stderr)


if __name__ == "__main__":
    main()
