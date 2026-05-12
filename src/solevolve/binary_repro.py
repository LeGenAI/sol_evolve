from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .tracing import file_sha256, traceable_run, update_current_run_context

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BINARY_ARTIFACT = ROOT_DIR / "artifacts" / "reviewer_core_claims.json"


def gf2_rank(matrix: list[list[int]]) -> int:
    """Compute row rank over GF(2) using integer bitsets."""
    if not matrix:
        return 0
    width = len(matrix[0])
    rows = []
    for row in matrix:
        value = 0
        for bit in row:
            value = (value << 1) | (int(bit) & 1)
        rows.append(value)

    rank = 0
    for col in range(width):
        mask = 1 << (width - 1 - col)
        pivot = next((idx for idx in range(rank, len(rows)) if rows[idx] & mask), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for idx in range(len(rows)):
            if idx != rank and rows[idx] & mask:
                rows[idx] ^= rows[rank]
        rank += 1
    return rank


def _row_to_int(row: list[int]) -> int:
    value = 0
    for bit in row:
        value = (value << 1) | (int(bit) & 1)
    return value


def _is_binary_matrix(matrix: list[list[int]]) -> bool:
    return bool(matrix) and all(row and all(value in {0, 1} for value in row) for row in matrix)


def _is_standard_form(matrix: list[list[int]]) -> bool:
    if not matrix:
        return False
    k = len(matrix)
    if any(len(row) < k for row in matrix):
        return False
    for row_idx, row in enumerate(matrix):
        for col_idx in range(k):
            expected = 1 if row_idx == col_idx else 0
            if row[col_idx] != expected:
                return False
    return True


def verify_binary_linear_code(
    *,
    matrix: list[list[int]],
    n: int,
    k: int,
    d: int,
    expected_a_d: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    row_lengths = [len(row) for row in matrix]
    shape_ok = len(matrix) == k and all(length == n for length in row_lengths)
    binary_ok = _is_binary_matrix(matrix)
    rank = gf2_rank(matrix) if shape_ok and binary_ok else 0
    row_ints = [_row_to_int(row) for row in matrix] if shape_ok and binary_ok else []

    weights: list[int] = []
    distribution: Counter[int] = Counter({0: 1})
    minimum_distance: int | None = None
    if row_ints:
        for message in range(1, 1 << k):
            codeword = 0
            for bit_idx, row_value in enumerate(row_ints):
                if message & (1 << bit_idx):
                    codeword ^= row_value
            weight = codeword.bit_count()
            weights.append(weight)
            distribution[weight] += 1
        minimum_distance = min(weights) if weights else None

    actual_distribution = dict(sorted(distribution.items()))
    actual_a_d = actual_distribution.get(d, 0)
    checks = {
        "shape_ok": shape_ok,
        "binary_entries": binary_ok,
        "rank_ok": rank == k,
        "standard_form": _is_standard_form(matrix) if shape_ok and binary_ok else False,
        "minimum_distance_ok": minimum_distance == d,
        "a_d_ok": expected_a_d is None or actual_a_d == expected_a_d,
    }
    return {
        "n": n,
        "k": k,
        "d": d,
        "row_count": len(matrix),
        "row_lengths": row_lengths,
        "rank": rank,
        "minimum_distance": minimum_distance,
        "weight_distribution": actual_distribution,
        "A_d": actual_a_d,
        "expected_A_d": expected_a_d,
        "checks": checks,
        "verification_ok": all(checks.values()),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _load_binary_artifact(path: str | Path | None = None) -> tuple[dict[str, Any] | None, Path, str | None]:
    artifact_path = Path(path) if path is not None else DEFAULT_BINARY_ARTIFACT
    if not artifact_path.exists():
        return None, artifact_path, f"Required binary-code artifact is missing: {artifact_path}"
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, artifact_path, f"Could not parse binary-code artifact: {exc}"
    if payload.get("claim_id") == "binary_ad_43_10_16":
        return payload, artifact_path, None
    nested = (payload.get("binary_codes") or {}).get("binary_ad_43_10_16")
    if isinstance(nested, dict):
        return nested, artifact_path, None
    return None, artifact_path, "Artifact does not contain binary_ad_43_10_16."


def _write_binary_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reproduce_binary_ad_43_10_16_impl(
    *,
    artifact_dir: str | Path = "artifacts",
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected_artifact = Path(artifact_path) if artifact_path is not None else Path(artifact_dir) / "reviewer_core_claims.json"
    if artifact_path is None and not selected_artifact.exists():
        selected_artifact = DEFAULT_BINARY_ARTIFACT
    payload, resolved_artifact, load_error = _load_binary_artifact(selected_artifact)
    output_root = Path(artifact_dir) / "binary_codes" / "binary_ad_43_10_16"
    output_root.mkdir(parents=True, exist_ok=True)
    if load_error is not None or payload is None:
        report = {
            "result_id": "binary_ad_43_10_16",
            "claim_family": "binary_codes",
            "verdict": "INSUFFICIENT_ARTIFACT",
            "parameters": {"n": 43, "k": 10, "d": 16},
            "artifact_path": str(resolved_artifact),
            "missing_obligations": ["public_generator_matrix_artifact"],
            "error": load_error,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path = output_root / "reproduction_report.json"
        _write_binary_report(report_path, report)
        report["report_path"] = str(report_path)
        return report

    n = int(payload["n"])
    k = int(payload["k"])
    d = int(payload["d"])
    solutions = payload.get("solutions", [])
    solution_reports = []
    for solution in solutions:
        verification = verify_binary_linear_code(
            matrix=solution["generator_matrix"],
            n=n,
            k=k,
            d=d,
            expected_a_d=int(solution["expected_A_d"]),
        )
        solution_reports.append(
            {
                "solution_id": solution.get("solution_id"),
                "expected_A_d": solution.get("expected_A_d"),
                "source_file": solution.get("source_file"),
                "source_sha256": solution.get("source_sha256"),
                "discovery_time_sec": solution.get("discovery_time_sec"),
                "diagnostics": verification,
            }
        )

    actual_progression = [int(item["diagnostics"]["A_d"]) for item in solution_reports]
    expected_progression = [int(item["expected_A_d"]) for item in solution_reports]
    progression_strictly_decreasing = all(
        left > right for left, right in zip(actual_progression, actual_progression[1:], strict=False)
    )
    all_verified = all(item["diagnostics"]["verification_ok"] for item in solution_reports)
    progression_ok = actual_progression == expected_progression == [91, 86, 80] and progression_strictly_decreasing
    bklc_a_d = int(payload.get("bklc_reference", {}).get("A_d", 0) or 0)
    best_a_d = min(actual_progression) if actual_progression else None
    reduction_fraction = ((bklc_a_d - best_a_d) / bklc_a_d) if bklc_a_d and best_a_d is not None else None
    checks = {
        "artifact_schema_ok": n == 43 and k == 10 and d == 16 and len(solution_reports) == 3,
        "all_solutions_verified": all_verified,
        "a_d_progression_ok": progression_ok,
        "bklc_reference_present": bklc_a_d == 225,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "result_id": "binary_ad_43_10_16",
        "claim_family": "binary_codes",
        "title": "A_d optimization for binary [43,10,16] codes",
        "verdict": verdict,
        "parameters": {"n": n, "k": k, "d": d},
        "artifact": {
            "path": str(resolved_artifact),
            "sha256": file_sha256(resolved_artifact),
        },
        "bklc_reference": payload.get("bklc_reference"),
        "A_d_progression": actual_progression,
        "expected_A_d_progression": expected_progression,
        "best_A_d": best_a_d,
        "bklc_A_d": bklc_a_d,
        "reduction_fraction": reduction_fraction,
        "reduction_percent": round(reduction_fraction * 100, 3) if reduction_fraction is not None else None,
        "solution_reports": solution_reports,
        "checks": checks,
        "missing_obligations": [] if verdict == "PASS" else [name for name, ok in checks.items() if not ok],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    report_path = output_root / "reproduction_report.json"
    _write_binary_report(report_path, report)
    report["report_path"] = str(report_path)
    update_current_run_context(
        metadata={
            "claim_id": "binary_ad_43_10_16",
            "claim_family": "binary_codes",
            "verdict": verdict,
            "n": n,
            "k": k,
            "d": d,
            "A_d_progression": actual_progression,
            "best_A_d": best_a_d,
            "bklc_A_d": bklc_a_d,
            "reduction_percent": report["reduction_percent"],
            "elapsed_ms": report["elapsed_ms"],
        },
        tags=["paper-claim", "binary_ad_43_10_16", "binary-code"],
    )
    return report


def reproduce_binary_ad_43_10_16(
    *,
    artifact_dir: str | Path = "artifacts",
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    @traceable_run("solevolve.binary_verify:binary_ad_43_10_16", run_type="chain")
    def _traced() -> dict[str, Any]:
        return _reproduce_binary_ad_43_10_16_impl(artifact_dir=artifact_dir, artifact_path=artifact_path)

    return _traced()
