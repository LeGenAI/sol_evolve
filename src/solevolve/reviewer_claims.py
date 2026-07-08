from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .binary_repro import reproduce_binary_ad_43_10_16
from .cegar_eager_repro import CEGAR_EAGER_CLAIM_IDS, reproduce_cegar_eager_claim
from .claim_registry import expand_claim_ids as expand_so_claim_ids
from .claim_registry import get_claim
from .contracts import HybridGAPolicy, LucasTarget, PaperTarget
from .hybrid_ga import HYBRID_CLAIM_ID, run_hybrid_ga
from .lucas_repro import LUCAS_CLAIMS, reproduce_lucas_claim
from .so_claim_repro import reproduce_paper_claim
from .tracing import traceable_run

SO_CLAIM_IDS = ["ternary_bch_d9", "gf4_hermitian_10_5_d6", "gf5_so_12_6_d8"]
LUCAS_CLAIM_IDS = ["lucas_l7_s4", "lucas_l15_s12", "lucas_l15_s11"]
BINARY_CLAIM_IDS = ["binary_ad_43_10_16"]
HYBRID_GA_CLAIM_IDS = [HYBRID_CLAIM_ID]

REVIEWER_CLAIM_GROUPS: dict[str, list[str]] = {
    "all_so_table": SO_CLAIM_IDS,
    "lucas_cubes": LUCAS_CLAIM_IDS,
    "binary_ad_43_10_16": BINARY_CLAIM_IDS,
    HYBRID_CLAIM_ID: [HYBRID_CLAIM_ID],
    "hybrid_sat_ga": [HYBRID_CLAIM_ID],
    "all_hybrid_ga": HYBRID_GA_CLAIM_IDS,
    "cegar_eager_scaling": CEGAR_EAGER_CLAIM_IDS,
    "all_reviewer_core": SO_CLAIM_IDS + LUCAS_CLAIM_IDS + BINARY_CLAIM_IDS,
}


def expand_reviewer_claim_ids(claim_id: str | None) -> list[str]:
    if not claim_id:
        return []
    normalized = claim_id.strip().lower()
    if normalized in REVIEWER_CLAIM_GROUPS:
        return list(REVIEWER_CLAIM_GROUPS[normalized])
    if normalized in LUCAS_CLAIMS:
        return [normalized]
    if normalized in BINARY_CLAIM_IDS:
        return [normalized]
    if normalized in HYBRID_GA_CLAIM_IDS:
        return [normalized]
    if normalized in CEGAR_EAGER_CLAIM_IDS:
        return [normalized]
    return expand_so_claim_ids(normalized)


def claim_family(claim_id: str) -> str:
    normalized = claim_id.strip().lower()
    if normalized in SO_CLAIM_IDS:
        return "self_orthogonal"
    if normalized in LUCAS_CLAIMS:
        return "lucas_cubes"
    if normalized in BINARY_CLAIM_IDS:
        return "binary_codes"
    if normalized in HYBRID_GA_CLAIM_IDS:
        return "hybrid_sat_ga"
    if normalized in CEGAR_EAGER_CLAIM_IDS:
        return "cegar_eager_scaling"
    raise KeyError(f"unknown reviewer claim id: {claim_id}")


def codetables_query_for_claim(claim_id: str) -> dict[str, int] | None:
    family = claim_family(claim_id)
    if family == "self_orthogonal":
        spec = get_claim(claim_id)
        return {"q": spec.field_order, "n": spec.n_prime, "k": spec.k}
    if claim_id == "binary_ad_43_10_16":
        return None
    return None


def paper_target_for_claim(claim_id: str) -> PaperTarget:
    family = claim_family(claim_id)
    if family == "self_orthogonal":
        spec = get_claim(claim_id)
        required_checks = [
            "explicit_matrix_verification",
            "sat_feasibility_minimum_distance_target",
        ]
        if spec.require_optimality_unsat:
            required_checks.append("optimality_no_larger_minimum_distance")
        return PaperTarget(
            claim_id=spec.claim_id,
            q=spec.field_order,
            n=spec.n_prime,
            k=spec.k,
            d=spec.d,
            field=f"GF({spec.field_order})",
            inner_product=spec.inner_product,
            objective=f"Reproduce paper claim {spec.claim_id}: {spec.title}",
            required_checks=required_checks,
            solver_preference=spec.solver_preference,
            source_n=spec.start_n,
            t=spec.t,
        )
    if family == "binary_codes":
        return PaperTarget(
            claim_id="binary_ad_43_10_16",
            q=2,
            n=43,
            k=10,
            d=16,
            field="GF(2)",
            inner_product="dot",
            objective="Verify archived A_d optimization progression for binary [43,10,16] codes.",
            required_checks=[
                "generator_matrix_artifacts",
                "gf2_rank_full",
                "minimum_distance_enumeration",
                "A16_progression_91_86_80",
            ],
            solver_preference="cadical",
        )
    if family == "hybrid_sat_ga":
        return PaperTarget(
            claim_id=HYBRID_CLAIM_ID,
            q=2,
            n=22,
            k=11,
            d=7,
            field="GF(2)",
            inner_product="dot",
            objective="Verify the paper-aligned Hybrid SAT-GA public witness for a binary [22,11,7] code.",
            required_checks=[
                "archived_hybrid_ga_witness",
                "gf2_rank_full",
                "minimum_distance_enumeration",
                "A7_weight_distribution",
                "optional_replay_or_live_hybrid_ga",
            ],
            solver_preference="cadical",
        )
    if family == "cegar_eager_scaling":
        if claim_id == "so_52_26_d8_cegar_eager":
            return PaperTarget(
                claim_id=claim_id,
                q=2,
                n=52,
                k=26,
                d=8,
                field="GF(2)",
                inner_product="dot",
                objective="Compare verifier-feedback CEGAR against eager distance encoding for the binary [52,26] self-orthogonal embedding.",
                required_checks=[
                    "agent_loop_run_claim_proof",
                    "cegar_threshold_sweep",
                    "eager_encoding_baseline",
                    "deterministic_minimum_distance_verification",
                    "artifact_hashes",
                ],
                solver_preference="cadical",
                source_n=31,
                t=21,
            )
        return PaperTarget(
            claim_id=claim_id,
            q=2,
            n=43,
            k=10,
            d=16,
            field="GF(2)",
            inner_product="dot",
            objective="Compare verifier-feedback CEGAR against eager distance encoding for binary [43,10] feasibility, with a capped d=17 frontier stress test.",
            required_checks=[
                "agent_loop_run_claim_proof",
                "d16_cegar_eager_comparison",
                "optional_d17_frontier_stress_test",
                "deterministic_minimum_distance_verification",
                "artifact_hashes",
            ],
            solver_preference="cadical",
        )
    lucas = LUCAS_CLAIMS[claim_id]
    return PaperTarget(
        claim_id=claim_id,
        q=2,
        n=int(lucas["n"]),
        k=int(lucas["expected_center_count"]),
        d=3,
        field="Lucas cube",
        inner_product="dot",
        objective=f"Verify exact perfect partition claim for Lambda_{lucas['n']}(1^{lucas['s']}).",
        required_checks=[
            "public_center_set_artifact",
            "circular_forbidden_vertex_generation",
            "exact_closed_neighborhood_cover",
            "pairwise_center_distance_at_least_3",
            "ball_size_distribution",
        ],
        solver_preference="none",
        lucas=LucasTarget(
            n=int(lucas["n"]),
            s=int(lucas["s"]),
            expected_vertex_count=int(lucas["expected_vertex_count"]),
            expected_center_count=int(lucas["expected_center_count"]),
            expected_ball_size_distribution={
                int(size): int(count)
                for size, count in dict(lucas["expected_ball_size_distribution"]).items()
            },
        ),
    )


def _claim_report_verdict(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "SKIPPED"
    if all(report.get("verdict") == "PASS" for report in reports):
        return "PASS"
    if any(report.get("verdict") == "FAIL" for report in reports):
        return "FAIL"
    return "PARTIAL"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@traceable_run("solevolve.reviewer_claims", run_type="chain")
def reproduce_reviewer_claims(
    *,
    claim_id: str,
    artifact_dir: str | Path = "artifacts",
    cadical_path: str | None = None,
    timeout: int = 300,
    prove_optimality: bool = True,
    hybrid_ga_policy: HybridGAPolicy | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    expanded = expand_reviewer_claim_ids(claim_id)
    reports: list[dict[str, Any]] = []
    for expanded_id in expanded:
        family = claim_family(expanded_id)
        if family == "self_orthogonal":
            reports.append(
                reproduce_paper_claim(
                    claim_id=expanded_id,
                    artifact_dir=artifact_dir,
                    cadical_path=cadical_path,
                    timeout=timeout,
                    prove_optimality=prove_optimality,
                )
            )
        elif family == "lucas_cubes":
            reports.append(reproduce_lucas_claim(claim_id=expanded_id, artifact_dir=artifact_dir))
        elif family == "binary_codes":
            reports.append(reproduce_binary_ad_43_10_16(artifact_dir=artifact_dir))
        elif family == "hybrid_sat_ga":
            policy = hybrid_ga_policy or HybridGAPolicy(
                enabled=True,
                mode="archived",
                seed=0,
                population=100,
                generations=100,
                repair_interval=50,
                timeout_sec=timeout,
                solver_preference="cadical",
            )
            reports.append(
                run_hybrid_ga(
                    policy=policy,
                    artifact_dir=artifact_dir,
                    cadical_path=cadical_path,
                    claim_id=expanded_id,
                )
            )
        elif family == "cegar_eager_scaling":
            reports.append(
                reproduce_cegar_eager_claim(
                    claim_id=expanded_id,
                    artifact_dir=artifact_dir,
                    cadical_path=cadical_path,
                    timeout=timeout,
                )
            )

    artifact_root = Path(artifact_dir) / "paper_claims"
    summary = {
        "result_id": claim_id,
        "verdict": _claim_report_verdict(reports),
        "claim_ids": expanded,
        "reports": reports,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    summary_path = artifact_root / f"{claim_id}_summary.json"
    _write_summary(summary_path, summary)
    summary["report_path"] = str(summary_path)
    return summary
