from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .tracing import file_sha256, traceable_run, update_current_run_context

ROOT_DIR = Path(__file__).resolve().parents[2]
REVIEWER_CORE_ARTIFACT = ROOT_DIR / "artifacts" / "reviewer_core_claims.json"

LUCAS_L7_S4_CENTERS = [
    "0000000",
    "0001011",
    "0010110",
    "0011101",
    "0100111",
    "0101100",
    "0110001",
    "0111010",
    "1000101",
    "1001110",
    "1010011",
    "1011000",
    "1100010",
    "1101001",
    "1110100",
]

LUCAS_CLAIMS: dict[str, dict[str, Any]] = {
    "lucas_l7_s4": {
        "claim_id": "lucas_l7_s4",
        "title": "Perfect partition of Lambda_7(1^4)",
        "n": 7,
        "s": 4,
        "expected_vertex_count": 99,
        "expected_center_count": 15,
        "expected_ball_size_distribution": {6: 7, 7: 7, 8: 1},
        "source": "neurocomputing_revision/manuscript/sections/A3_lucas_cubes.tex",
        "required": True,
    },
    "lucas_l15_s12": {
        "claim_id": "lucas_l15_s12",
        "title": "Perfect partition of Lambda_15(1^12)",
        "n": 15,
        "s": 12,
        "expected_vertex_count": 32707,
        "expected_center_count": 2047,
        "expected_ball_size_distribution": {14: 11, 15: 23, 16: 2013},
        "source": "artifacts/reviewer_core_claims.json from archived 2026-05-06 SAT experiment",
        "required": True,
    },
    "lucas_l15_s11": {
        "claim_id": "lucas_l15_s11",
        "title": "Perfect partition of Lambda_15(1^11)",
        "n": 15,
        "s": 11,
        "expected_vertex_count": 32647,
        "expected_center_count": 2047,
        "expected_ball_size_distribution": {14: 16, 15: 73, 16: 1958},
        "source": "artifacts/reviewer_core_claims.json from archived 2026-05-06 SAT experiment",
        "required": True,
    },
}


def _contains_circular_run(bits: str, run_length: int) -> bool:
    if run_length <= 0:
        return True
    if not bits:
        return False
    circular = bits + bits[: max(run_length - 1, 0)]
    return "1" * run_length in circular


def lucas_vertices(n: int, s: int) -> list[str]:
    """Generate vertices of Lambda_n(1^s) as fixed-width bitstrings."""
    return [
        format(value, f"0{n}b")
        for value in range(1 << n)
        if not _contains_circular_run(format(value, f"0{n}b"), s)
    ]


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def _closed_neighborhood(center: str, vertex_set: set[str]) -> list[str]:
    neighbors = [center] if center in vertex_set else []
    for index, bit in enumerate(center):
        flipped = center[:index] + ("0" if bit == "1" else "1") + center[index + 1 :]
        if flipped in vertex_set:
            neighbors.append(flipped)
    return neighbors


def _artifact_path(artifact_dir: str | Path, artifact_name: str) -> Path:
    return Path(artifact_dir) / "lucas_cubes" / artifact_name


def _load_core_lucas_centers(claim_id: str, artifact_dir: str | Path) -> tuple[list[str] | None, Path | None, str | None]:
    candidates = [
        Path(artifact_dir) / "reviewer_core_claims.json",
        REVIEWER_CORE_ARTIFACT,
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, path, f"Could not parse reviewer core artifact: {exc}"
        claim_payload = ((payload.get("lucas_cubes") or {}).get(claim_id) or {})
        centers = claim_payload.get("centers")
        if centers is None:
            return None, path, f"Reviewer core artifact has no public center set for {claim_id}."
        if not isinstance(centers, list) or not all(isinstance(item, str) for item in centers):
            return None, path, "Lucas center artifact must contain a string list at key 'centers'."
        return centers, path, None
    return None, None, None


def _load_center_artifact(spec: dict[str, Any], artifact_dir: str | Path) -> tuple[list[str] | None, Path | None, str | None]:
    if "centers" in spec:
        return list(spec["centers"]), None, None
    core_centers, core_path, core_error = _load_core_lucas_centers(str(spec["claim_id"]), artifact_dir)
    if core_centers is not None or core_error is not None:
        return core_centers, core_path, core_error
    artifact_name = spec.get("artifact_name")
    if not artifact_name:
        return None, None, "Lucas claim has no bundled centers and no artifact_name."
    path = _artifact_path(artifact_dir, artifact_name)
    if not path.exists():
        bundled = _artifact_path(ROOT_DIR / "artifacts", artifact_name)
        if bundled.exists():
            path = bundled
    if not path.exists():
        return None, path, f"Required Lucas center artifact is missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, path, f"Could not parse Lucas center artifact: {exc}"
    centers = payload.get("centers")
    if not isinstance(centers, list) or not all(isinstance(item, str) for item in centers):
        return None, path, "Lucas center artifact must contain a string list at key 'centers'."
    return centers, path, None


def verify_lucas_partition(
    *,
    n: int,
    s: int,
    centers: list[str],
    expected_vertex_count: int,
    expected_center_count: int,
    expected_ball_size_distribution: dict[int, int],
) -> dict[str, Any]:
    started = time.perf_counter()
    vertices = lucas_vertices(n, s)
    vertex_set = set(vertices)
    center_set = set(centers)
    invalid_centers = sorted(center for center in centers if center not in vertex_set)
    duplicate_count = len(centers) - len(center_set)

    coverage = {vertex: 0 for vertex in vertices}
    ball_sizes: list[int] = []
    for center in centers:
        neighborhood = _closed_neighborhood(center, vertex_set)
        ball_sizes.append(len(neighborhood))
        for vertex in neighborhood:
            coverage[vertex] += 1

    uncovered = [vertex for vertex, count in coverage.items() if count == 0]
    overcovered = [vertex for vertex, count in coverage.items() if count > 1]
    min_pairwise_distance = None
    for index, left in enumerate(centers):
        for right in centers[index + 1 :]:
            distance = hamming_distance(left, right)
            min_pairwise_distance = distance if min_pairwise_distance is None else min(min_pairwise_distance, distance)

    actual_distribution = dict(sorted(Counter(ball_sizes).items()))
    expected_distribution = dict(sorted((int(key), int(value)) for key, value in expected_ball_size_distribution.items()))
    checks = {
        "vertex_count_ok": len(vertices) == expected_vertex_count,
        "center_count_ok": len(centers) == expected_center_count,
        "centers_unique": duplicate_count == 0,
        "centers_valid": not invalid_centers,
        "minimum_pairwise_distance_ok": min_pairwise_distance is not None and min_pairwise_distance >= 3,
        "exact_cover": not uncovered and not overcovered and all(count == 1 for count in coverage.values()),
        "ball_size_distribution_ok": actual_distribution == expected_distribution,
    }
    return {
        "n": n,
        "s": s,
        "vertex_count": len(vertices),
        "expected_vertex_count": expected_vertex_count,
        "center_count": len(centers),
        "expected_center_count": expected_center_count,
        "duplicate_center_count": duplicate_count,
        "invalid_center_count": len(invalid_centers),
        "invalid_centers_preview": invalid_centers[:10],
        "minimum_pairwise_distance": min_pairwise_distance,
        "uncovered_count": len(uncovered),
        "overcovered_count": len(overcovered),
        "uncovered_preview": uncovered[:10],
        "overcovered_preview": overcovered[:10],
        "ball_size_distribution": actual_distribution,
        "expected_ball_size_distribution": expected_distribution,
        "checks": checks,
        "verification_ok": all(checks.values()),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _write_lucas_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reproduce_lucas_claim_impl(
    *,
    claim_id: str,
    artifact_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    started = time.perf_counter()
    if claim_id not in LUCAS_CLAIMS:
        raise KeyError(f"unknown Lucas claim id: {claim_id}")
    spec = LUCAS_CLAIMS[claim_id]
    output_root = Path(artifact_dir) / "lucas_cubes" / claim_id
    output_root.mkdir(parents=True, exist_ok=True)
    centers, center_path, load_error = _load_center_artifact(spec, artifact_dir)

    if load_error is not None or centers is None:
        report = {
            "result_id": claim_id,
            "claim_family": "lucas_cubes",
            "title": spec["title"],
            "verdict": "INSUFFICIENT_ARTIFACT",
            "parameters": {"n": spec["n"], "s": spec["s"]},
            "expected_vertex_count": spec["expected_vertex_count"],
            "expected_center_count": spec["expected_center_count"],
            "expected_ball_size_distribution": spec["expected_ball_size_distribution"],
            "missing_obligations": ["public_center_set_artifact"],
            "artifact_path": str(center_path) if center_path else None,
            "error": load_error,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path = output_root / "reproduction_report.json"
        _write_lucas_report(report_path, report)
        report["report_path"] = str(report_path)
        update_current_run_context(
            metadata={
                "claim_id": claim_id,
                "claim_family": "lucas_cubes",
                "verdict": report["verdict"],
                "missing_obligations": report["missing_obligations"],
                "elapsed_ms": report["elapsed_ms"],
            },
            tags=["paper-claim", claim_id, "lucas"],
        )
        return report

    verification = verify_lucas_partition(
        n=int(spec["n"]),
        s=int(spec["s"]),
        centers=centers,
        expected_vertex_count=int(spec["expected_vertex_count"]),
        expected_center_count=int(spec["expected_center_count"]),
        expected_ball_size_distribution=dict(spec["expected_ball_size_distribution"]),
    )
    artifacts = []
    if center_path:
        artifacts.append({"path": str(center_path), "sha256": file_sha256(center_path)})

    verdict = "PASS" if verification["verification_ok"] else "FAIL"
    missing = [] if verdict == "PASS" else [name for name, ok in verification["checks"].items() if not ok]
    report = {
        "result_id": claim_id,
        "claim_family": "lucas_cubes",
        "title": spec["title"],
        "verdict": verdict,
        "parameters": {"n": spec["n"], "s": spec["s"]},
        "expected_vertex_count": spec["expected_vertex_count"],
        "expected_center_count": spec["expected_center_count"],
        "expected_ball_size_distribution": spec["expected_ball_size_distribution"],
        "source": spec.get("source"),
        "source_artifacts": artifacts,
        "diagnostics": verification,
        "missing_obligations": missing,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    report_path = output_root / "reproduction_report.json"
    _write_lucas_report(report_path, report)
    report["report_path"] = str(report_path)
    update_current_run_context(
        metadata={
            "claim_id": claim_id,
            "claim_family": "lucas_cubes",
            "verdict": verdict,
            "n": spec["n"],
            "s": spec["s"],
            "vertex_count": verification["vertex_count"],
            "center_count": verification["center_count"],
            "minimum_pairwise_distance": verification["minimum_pairwise_distance"],
            "exact_cover": verification["checks"]["exact_cover"],
            "ball_size_distribution": verification["ball_size_distribution"],
            "elapsed_ms": report["elapsed_ms"],
        },
        tags=["paper-claim", claim_id, "lucas"],
    )
    return report


def reproduce_lucas_claim(
    *,
    claim_id: str,
    artifact_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    @traceable_run(f"solevolve.lucas_verify:{claim_id}", run_type="chain")
    def _traced() -> dict[str, Any]:
        return _reproduce_lucas_claim_impl(claim_id=claim_id, artifact_dir=artifact_dir)

    return _traced()


@traceable_run("solevolve.lucas_claims", run_type="chain")
def reproduce_lucas_claims(
    *,
    claim_ids: list[str],
    artifact_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    started = time.perf_counter()
    reports = [reproduce_lucas_claim(claim_id=claim_id, artifact_dir=artifact_dir) for claim_id in claim_ids]
    verdict = "PASS" if all(report["verdict"] == "PASS" for report in reports) else "FAIL" if any(report["verdict"] == "FAIL" for report in reports) else "PARTIAL"
    return {
        "result_id": "lucas_cubes",
        "claim_family": "lucas_cubes",
        "verdict": verdict,
        "claim_ids": claim_ids,
        "reports": reports,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
