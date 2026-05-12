from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .binary_repro import verify_binary_linear_code
from .contracts import HybridGAPolicy, HybridGARepairEvent, HybridGAReport
from .engines.cnf_encoder import CNFEncoder
from .engines.code_generator import CodeGenerator
from .engines.result_parser import ResultParser
from .engines.sat_solver_interface import SATSolver
from .so_claim_repro import resolve_cadical
from .tracing import file_sha256, traceable_run, update_current_run_context

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REVIEWER_ARTIFACT = ROOT_DIR / "artifacts" / "reviewer_core_claims.json"
HYBRID_CLAIM_ID = "binary_22_11_7_hybrid_ga"
HYBRID_CLAIM_IDS = {HYBRID_CLAIM_ID}
TARGET_N = 22
TARGET_K = 11
TARGET_D = 7
TARGET_A_D = 176


def _target_config(claim_id: str, policy: HybridGAPolicy) -> dict[str, Any]:
    if claim_id == HYBRID_CLAIM_ID:
        target_d = int(policy.target_distance or TARGET_D)
        frontier_d = int(policy.frontier_distance or max(1, target_d - 1))
        return {
            "claim_id": HYBRID_CLAIM_ID,
            "n": TARGET_N,
            "k": TARGET_K,
            "target_d": target_d,
            "frontier_d": frontier_d,
            "expected_a_d": TARGET_A_D if target_d == TARGET_D else None,
            "experimental": False,
        }
    raise KeyError(f"unsupported Hybrid SAT-GA claim id: {claim_id}")


def _policy_config(policy: HybridGAPolicy) -> dict[str, Any]:
    return {
        "enabled": policy.enabled,
        "mode": policy.mode,
        "seed": policy.seed,
        "population": policy.population,
        "generations": policy.generations,
        "repair_interval": policy.repair_interval,
        "timeout_sec": policy.timeout_sec,
        "solver_preference": policy.solver_preference,
        "target_distance": policy.target_distance,
        "frontier_distance": policy.frontier_distance,
        "frontier_seed_count": policy.frontier_seed_count,
        "frontier_target_timeout_sec": policy.frontier_target_timeout_sec,
        "frontier_seed_timeout_sec": policy.frontier_seed_timeout_sec,
        "repair_strategy": policy.repair_strategy,
    }


def _artifact_path(artifact_dir: str | Path, artifact_path: str | Path | None = None) -> Path:
    if artifact_path is not None:
        return Path(artifact_path)
    candidate = Path(artifact_dir) / "reviewer_core_claims.json"
    return candidate if candidate.exists() else DEFAULT_REVIEWER_ARTIFACT


def _load_hybrid_artifact(
    *,
    artifact_dir: str | Path,
    artifact_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path, str | None]:
    path = _artifact_path(artifact_dir, artifact_path)
    if not path.exists():
        return None, path, f"Required reviewer artifact is missing: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, path, f"Could not parse reviewer artifact: {exc}"
    if payload.get("claim_id") == HYBRID_CLAIM_ID:
        return payload, path, None
    nested = (payload.get("binary_codes") or {}).get(HYBRID_CLAIM_ID)
    if isinstance(nested, dict):
        return nested, path, None
    return None, path, f"Artifact does not contain {HYBRID_CLAIM_ID}."


def _write_report(path: Path, report: HybridGAReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _matrix_from_payload(payload: dict[str, Any]) -> list[list[int]]:
    matrix = payload.get("generator_matrix")
    if not isinstance(matrix, list):
        raise ValueError("hybrid artifact is missing generator_matrix")
    return [[int(value) & 1 for value in row] for row in matrix]


def _matrix_from_repair_payload(repair_payload: dict[str, Any]) -> list[list[int]]:
    matrix = repair_payload.get("pre_repair_generator_matrix")
    if not isinstance(matrix, list):
        raise ValueError("hybrid repair replay artifact is missing pre_repair_generator_matrix")
    return [[int(value) & 1 for value in row] for row in matrix]


def _diagnostics(
    matrix: list[list[int]],
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
    d: int = TARGET_D,
    expected_a_d: int | None = TARGET_A_D,
) -> dict[str, Any]:
    diagnostics = verify_binary_linear_code(matrix=matrix, n=n, k=k, d=d, expected_a_d=expected_a_d)
    minimum_distance = int(diagnostics.get("minimum_distance") or 0)
    diagnostics["d_min"] = minimum_distance
    diagnostics["A_min"] = (diagnostics.get("weight_distribution") or {}).get(minimum_distance, 0)
    distance_at_least_target = minimum_distance >= d
    expected_a_ok = expected_a_d is None or diagnostics.get("A_d") == expected_a_d
    diagnostics["checks"] = {
        **diagnostics.get("checks", {}),
        "minimum_distance_at_least_target": distance_at_least_target,
        "expected_A_d_ok": expected_a_ok,
    }
    diagnostics["target_achieved"] = bool(
        diagnostics.get("checks", {}).get("shape_ok")
        and diagnostics.get("checks", {}).get("binary_entries")
        and diagnostics.get("checks", {}).get("rank_ok")
        and distance_at_least_target
        and expected_a_ok
    )
    return diagnostics


def _low_weight_summary(
    matrix: list[list[int]],
    target_d: int = TARGET_D,
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
    limit: int = 5,
) -> dict[str, Any]:
    arr = np.asarray(matrix, dtype=np.uint8)
    supports: list[list[int]] = []
    count = 0
    for message in range(1, 1 << k):
        codeword = np.zeros(n, dtype=np.uint8)
        for bit_idx in range(k):
            if message & (1 << bit_idx):
                codeword ^= arr[bit_idx]
        weight = int(codeword.sum())
        if weight < target_d:
            count += 1
            if len(supports) < limit:
                supports.append([idx for idx, value in enumerate(codeword.tolist()) if value])
    return {"target_d": target_d, "count": count, "sample_supports": supports}


def _mutable_columns_from_low_weight_supports(
    matrix: list[list[int]],
    target_d: int = TARGET_D,
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
) -> list[int]:
    arr = np.asarray(matrix, dtype=np.uint8)
    columns: set[int] = set()
    for message in range(1, 1 << k):
        codeword = np.zeros(n, dtype=np.uint8)
        for bit_idx in range(k):
            if message & (1 << bit_idx):
                codeword ^= arr[bit_idx]
        if int(codeword.sum()) < target_d:
            columns.update(idx for idx, value in enumerate(codeword.tolist()) if value and idx >= k)
    return sorted(columns) or list(range(k, n))


def _diagnostics_for_actual_min(matrix: list[list[int]], *, n: int = TARGET_N, k: int = TARGET_K) -> dict[str, Any]:
    rough = verify_binary_linear_code(matrix=matrix, n=n, k=k, d=1, expected_a_d=None)
    actual_d = int(rough.get("minimum_distance") or 0)
    actual_a = (rough.get("weight_distribution") or {}).get(actual_d, 0)
    return _diagnostics(matrix, n=n, k=k, d=max(1, actual_d), expected_a_d=actual_a)


def _seed_record(
    matrix: list[list[int]],
    *,
    source: str,
    solution_number: int | None = None,
    n: int = TARGET_N,
    k: int = TARGET_K,
) -> dict[str, Any]:
    diagnostics = _diagnostics_for_actual_min(matrix, n=n, k=k)
    minimum_distance = int(diagnostics.get("minimum_distance") or 0)
    return {
        "seed_id": _candidate_id(matrix),
        "solution_number": solution_number,
        "source": source,
        "matrix": matrix,
        "matrix_sha256": _candidate_id(matrix),
        "rank": diagnostics.get("rank"),
        "d_min": minimum_distance,
        "A_min": (diagnostics.get("weight_distribution") or {}).get(minimum_distance, 0),
        "weight_distribution": diagnostics.get("weight_distribution"),
        "diagnostics": diagnostics,
    }


def _compact_seed_bank(
    seeds: list[dict[str, Any]],
    *,
    artifact_path: Path | None = None,
    k: int = TARGET_K,
) -> dict[str, Any]:
    matrices = [seed["matrix"] for seed in seeds if isinstance(seed.get("matrix"), list)]
    return {
        "seed_count": len(seeds),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "diversity": _diversity(matrices, k=k) if matrices else 0.0,
        "seeds": [
            {key: value for key, value in seed.items() if key not in {"matrix", "diagnostics"}}
            for seed in seeds
        ],
    }


def _write_seed_bank(
    path: Path,
    *,
    seeds: list[dict[str, Any]],
    policy: HybridGAPolicy,
    solver_runs: list[dict[str, Any]],
    claim_id: str = HYBRID_CLAIM_ID,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "claim_id": claim_id,
        "schema_version": 1,
        "config": _policy_config(policy),
        "seed_count": len(seeds),
        "diversity": _diversity([seed["matrix"] for seed in seeds]) if seeds else 0.0,
        "solver_runs": solver_runs,
        "seeds": seeds,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score(diagnostics: dict[str, Any]) -> float:
    d_min = diagnostics.get("minimum_distance") or 0
    a_d = diagnostics.get("A_d")
    if a_d is None:
        a_d = 10**9
    return float(d_min * 10_000 - a_d)


def _diversity(population: list[list[list[int]]], *, k: int = TARGET_K) -> float:
    if len(population) < 2:
        return 0.0
    arrays = [np.asarray(matrix, dtype=np.uint8)[:, k:] for matrix in population]
    total = 0
    distance = 0
    for left_idx, left in enumerate(arrays):
        for right in arrays[left_idx + 1 :]:
            distance += int(np.count_nonzero(left != right))
            total += int(left.size)
    return round(distance / total, 6) if total else 0.0


def _mutate_matrix(
    matrix: list[list[int]],
    rng: np.random.Generator,
    rate: float = 0.015,
    *,
    k: int = TARGET_K,
) -> list[list[int]]:
    arr = np.asarray(matrix, dtype=np.uint8).copy()
    parity = arr[:, k:]
    mask = rng.random(parity.shape) < rate
    parity ^= mask.astype(np.uint8)
    arr[:, :k] = np.eye(k, dtype=np.uint8)
    return arr.astype(int).tolist()


def _random_systematic_matrix(
    rng: np.random.Generator,
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
) -> list[list[int]]:
    identity = np.eye(k, dtype=np.uint8)
    parity = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8)
    return np.concatenate([identity, parity], axis=1).astype(int).tolist()


def _candidate_id(matrix: list[list[int]]) -> str:
    body = json.dumps(matrix, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(body).hexdigest()[:16]


def _matrix_shape(matrix: Any) -> dict[str, int] | None:
    if not isinstance(matrix, list) or not matrix:
        return None
    first = matrix[0]
    if not isinstance(first, list):
        return None
    return {"rows": len(matrix), "cols": len(first)}


def _sanitize_hybrid_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(inputs)
    for key in ("seed_matrix", "candidate", "matrix", "post_matrix"):
        if key in sanitized:
            sanitized[f"{key}_shape"] = _matrix_shape(sanitized[key])
            sanitized[f"{key}_sha256"] = _candidate_id(sanitized[key]) if _matrix_shape(sanitized[key]) else None
            sanitized.pop(key, None)
    if "repair_source" in sanitized and isinstance(sanitized["repair_source"], dict):
        repair_source = dict(sanitized["repair_source"])
        repair_source.pop("pre_repair_generator_matrix", None)
        sanitized["repair_source"] = repair_source
    if "population" in sanitized:
        population = sanitized.pop("population")
        sanitized["population_count"] = len(population) if isinstance(population, list) else None
        sanitized["population_matrix_shape"] = _matrix_shape(population[0]) if isinstance(population, list) and population else None
    if "seeds" in sanitized:
        seeds = sanitized.pop("seeds")
        sanitized["seed_count"] = len(seeds) if isinstance(seeds, list) else None
        sanitized["seed_matrix_shape"] = _matrix_shape(seeds[0].get("matrix")) if isinstance(seeds, list) and seeds and isinstance(seeds[0], dict) else None
    return sanitized


def _sanitize_hybrid_outputs(output: Any) -> Any:
    raw_matrix_keys = {
        "candidate",
        "matrix",
        "post_matrix",
        "post_repair_matrix",
        "pre_repair_generator_matrix",
        "seed_matrix",
        "best_matrix",
    }

    def sanitize(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            try:
                return sanitize(value.model_dump(mode="python"))
            except TypeError:
                return sanitize(value.model_dump())
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, dict):
            return value

        sanitized: dict[str, Any] = {}
        if "population" in value:
            population = value.get("population")
            sanitized["population_count"] = len(population) if isinstance(population, list) else None
            sanitized["population_matrix_shape"] = (
                _matrix_shape(population[0]) if isinstance(population, list) and population else None
            )
        if "seeds" in value:
            seeds = value.get("seeds")
            sanitized["seed_count"] = len(seeds) if isinstance(seeds, list) else None
            compact = []
            if isinstance(seeds, list):
                for seed in seeds:
                    compact.append(sanitize(seed) if isinstance(seed, dict) else seed)
            sanitized["seed_summaries"] = compact[:20]

        for key, item in value.items():
            if key in {"population", "seeds"}:
                continue
            if key in raw_matrix_keys:
                shape = _matrix_shape(item)
                sanitized[f"{key}_shape"] = shape
                sanitized[f"{key}_sha256"] = _candidate_id(item) if shape else None
                continue
            sanitized[key] = sanitize(item)
        return sanitized

    return sanitize(output)


@traceable_run("hybrid_ga:archived_verify", run_type="chain")
def verify_archived_hybrid_ga(
    *,
    policy: HybridGAPolicy,
    artifact_dir: str | Path = "artifacts",
    artifact_path: str | Path | None = None,
) -> HybridGAReport:
    started = time.perf_counter()
    payload, resolved_artifact, error = _load_hybrid_artifact(artifact_dir=artifact_dir, artifact_path=artifact_path)
    report_dir = Path(artifact_dir) / "hybrid_ga" / HYBRID_CLAIM_ID
    if error is not None or payload is None:
        report = HybridGAReport(
            claim_id=HYBRID_CLAIM_ID,
            mode=policy.mode,
            config=_policy_config(policy),
            seed_status="artifact_missing",
            final_diagnostics={"error": error, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
            verdict="INSUFFICIENT_ARTIFACT",
            missing_obligations=["public_binary_22_11_7_generator_matrix"],
            artifact_paths=[],
        )
        _write_report(report_dir / "hybrid_ga_report.json", report)
        return report

    matrix = _matrix_from_payload(payload)
    diagnostics = _diagnostics(matrix)
    expected_distribution = {int(k): int(v) for k, v in (payload.get("expected_weight_distribution") or {}).items()}
    checks = {
        "rank_ok": diagnostics.get("rank") == TARGET_K,
        "minimum_distance_ok": diagnostics.get("minimum_distance") == TARGET_D,
        "a_d_ok": diagnostics.get("A_d") == TARGET_A_D,
        "weight_distribution_ok": diagnostics.get("weight_distribution") == expected_distribution,
        "source_hash_present": bool(payload.get("source_sha256")),
    }
    verdict = "PASS" if all(checks.values()) and diagnostics.get("verification_ok") else "FAIL"
    diagnostics.update(
        {
            "checks": {**diagnostics.get("checks", {}), **checks},
            "artifact": {"path": str(resolved_artifact), "sha256": file_sha256(resolved_artifact)},
            "source_path": payload.get("source_path"),
            "source_sha256": payload.get("source_sha256"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    )
    report_path = report_dir / "hybrid_ga_report.json"
    report = HybridGAReport(
        claim_id=HYBRID_CLAIM_ID,
        mode=policy.mode,
        config=_policy_config(policy),
        seed_status="archived_final_matrix_verified",
        generation_summaries=[
            {
                "generation": 0,
                "best_candidate_id": _candidate_id(matrix),
                "best_d_min": diagnostics.get("minimum_distance"),
                "best_A_d": diagnostics.get("A_d"),
                "best_fitness": _score(diagnostics),
                "diversity": 0.0,
            }
        ],
        final_diagnostics=diagnostics,
        verdict=verdict,
        missing_obligations=[] if verdict == "PASS" else [name for name, ok in checks.items() if not ok],
        artifact_paths=[str(report_path)],
    )
    _write_report(report_path, report)
    return report


@traceable_run(
    "hybrid_ga:init_population",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def initialize_population(
    *,
    seed_matrix: list[list[int]] | None,
    policy: HybridGAPolicy,
    n: int = TARGET_N,
    k: int = TARGET_K,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(policy.seed)
    population: list[list[list[int]]] = []
    if seed_matrix is not None:
        population.append(seed_matrix)
    while len(population) < policy.population:
        if seed_matrix is not None and len(population) <= max(2, policy.population // 5):
            population.append(_mutate_matrix(seed_matrix, rng, rate=0.01 + 0.002 * len(population), k=k))
        else:
            population.append(_random_systematic_matrix(rng, n=n, k=k))
    return {
        "population": population,
        "seed_status": "archived_seed_plus_deterministic_population" if seed_matrix is not None else "deterministic_random_population",
        "diversity": _diversity(population, k=k),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _generation_summary(
    population: list[list[list[int]]],
    generation: int,
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
    expected_a_d: int | None = TARGET_A_D,
) -> tuple[dict[str, Any], list[list[int]]]:
    scored = []
    for matrix in population:
        diagnostics = _diagnostics(matrix, n=n, k=k, d=target_d, expected_a_d=expected_a_d)
        scored.append((_score(diagnostics), matrix, diagnostics))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_matrix, best_diagnostics = scored[0]
    summary = {
        "generation": generation,
        "best_candidate_id": _candidate_id(best_matrix),
        "best_d_min": best_diagnostics.get("minimum_distance"),
        "best_A_d": best_diagnostics.get("A_d"),
        "best_A_min": (best_diagnostics.get("weight_distribution") or {}).get(best_diagnostics.get("minimum_distance"), 0),
        "best_fitness": best_score,
        "avg_fitness": round(sum(item[0] for item in scored) / len(scored), 6),
        "diversity": _diversity(population, k=k),
    }
    return summary, best_matrix


@traceable_run(
    "hybrid_ga:live_seed",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def generate_live_seed(
    *,
    policy: HybridGAPolicy,
    artifact_dir: str | Path,
    cadical_path: str | None = None,
    claim_id: str = HYBRID_CLAIM_ID,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = resolve_cadical(cadical_path)
    if not resolved:
        return {
            "status": "UNAVAILABLE",
            "seed_status": "cadical_unavailable",
            "seed_matrix": None,
            "solver_runs": [],
            "missing_obligations": ["cadical_binary_for_live_seed_generation"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    workdir = Path(artifact_dir) / "hybrid_ga" / claim_id / "live_seed"
    generator = CodeGenerator(n=n, k=k, d_min=target_d, systematic=True, working_dir=str(workdir))
    result = generator.solve(
        kissat_path=resolved,
        solver_type="cadical",
        solver_path=resolved,
        timeout=policy.timeout_sec,
        verbose=False,
    )
    matrix = result.get("generator_matrix")
    seed_matrix = matrix.astype(int).tolist() if hasattr(matrix, "astype") else None
    cnf_path = result.get("cnf_file")
    solver_path = result.get("solver_path") or resolved
    solver_run = {
        "name": f"live_seed_generation:d>={target_d}",
        "claim_id": claim_id,
        "obligation": f"live_seed_generation:d>={target_d}",
        "solver": result.get("solver_used") or "cadical",
        "status": result.get("status"),
        "solver_path": solver_path,
        "argv": [solver_path, cnf_path] if cnf_path else [solver_path],
        "command": " ".join([solver_path, cnf_path]) if cnf_path else solver_path,
        "cwd": str(workdir),
        "elapsed_ms": round(float(result.get("solver_time") or 0.0) * 1000, 3),
        "timeout_sec": policy.timeout_sec,
        "cnf_path": cnf_path,
        "cnf_sha256": file_sha256(cnf_path) if cnf_path and Path(cnf_path).exists() else None,
        "return_code": None,
        "memory_mb": None,
    }
    return {
        "status": result.get("status") or "UNKNOWN",
        "seed_status": "cadical_live_seed_sat" if seed_matrix is not None else f"cadical_live_seed_{result.get('status', 'unknown').lower()}",
        "seed_matrix": seed_matrix,
        "solver_runs": [solver_run],
        "missing_obligations": [] if seed_matrix is not None else ["live_seed_generation_sat"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@traceable_run(
    "hybrid_ga:frontier_target_sat",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def run_frontier_target_sat(
    *,
    policy: HybridGAPolicy,
    artifact_dir: str | Path,
    cadical_path: str | None = None,
    claim_id: str = HYBRID_CLAIM_ID,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
    expected_a_d: int | None = TARGET_A_D,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = resolve_cadical(cadical_path)
    if not resolved:
        return {
            "status": "UNAVAILABLE",
            "seed_matrix": None,
            "solver_runs": [],
            "missing_obligations": ["cadical_binary_for_frontier_target_sat"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    workdir = Path(artifact_dir) / "hybrid_ga" / claim_id / "frontier_target"
    generator = CodeGenerator(
        n=n,
        k=k,
        d_min=target_d,
        systematic=True,
        working_dir=str(workdir),
    )
    result = generator.solve(
        solver_type="cadical",
        solver_path=resolved,
        timeout=policy.frontier_target_timeout_sec,
        verbose=False,
    )
    matrix = result.get("generator_matrix")
    seed_matrix = matrix.astype(int).tolist() if hasattr(matrix, "astype") else None
    diagnostics = _diagnostics(seed_matrix, n=n, k=k, d=target_d, expected_a_d=expected_a_d) if seed_matrix is not None else None
    cnf_path = result.get("cnf_file")
    solver_path = result.get("solver_path") or resolved
    solver_run = {
        "name": f"frontier_target_sat:d>={target_d}",
        "claim_id": claim_id,
        "obligation": f"frontier_target_sat:d>={target_d}",
        "solver": result.get("solver_used") or "cadical",
        "status": result.get("status"),
        "solver_path": solver_path,
        "argv": [solver_path, cnf_path] if cnf_path else [solver_path],
        "command": " ".join([solver_path, cnf_path]) if cnf_path else solver_path,
        "cwd": str(workdir),
        "elapsed_ms": round(float(result.get("solver_time") or 0.0) * 1000, 3),
        "timeout_sec": policy.frontier_target_timeout_sec,
        "cnf_path": cnf_path,
        "cnf_sha256": file_sha256(cnf_path) if cnf_path and Path(cnf_path).exists() else None,
        "return_code": None,
        "memory_mb": None,
    }
    return {
        "status": result.get("status") or "UNKNOWN",
        "seed_matrix": seed_matrix,
        "diagnostics": diagnostics,
        "solver_runs": [solver_run],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@traceable_run(
    "hybrid_ga:frontier_seed_bank",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def generate_frontier_seed_bank(
    *,
    policy: HybridGAPolicy,
    artifact_dir: str | Path,
    cadical_path: str | None = None,
    claim_id: str = HYBRID_CLAIM_ID,
    n: int = TARGET_N,
    k: int = TARGET_K,
    frontier_d: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = resolve_cadical(cadical_path)
    report_dir = Path(artifact_dir) / "hybrid_ga" / claim_id
    seed_bank_path = report_dir / "seed_bank.json"
    selected_frontier_d = int(frontier_d or policy.frontier_distance or max(1, policy.target_distance - 1))
    if not resolved:
        return {
            "status": "UNAVAILABLE",
            "seeds": [],
            "seed_bank": _compact_seed_bank([], artifact_path=seed_bank_path, k=k),
            "solver_runs": [],
            "missing_obligations": ["cadical_binary_for_frontier_seed_bank"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    workdir = report_dir / "frontier_seed_bank"
    generator = CodeGenerator(
        n=n,
        k=k,
        d_min=selected_frontier_d,
        systematic=True,
        working_dir=str(workdir),
    )
    results = generator.solve_multiple(
        solver_type="cadical",
        solver_path=resolved,
        num_solutions=policy.frontier_seed_count,
        timeout_per_instance=policy.frontier_seed_timeout_sec,
        verbose=False,
    )
    seeds: list[dict[str, Any]] = []
    solver_runs: list[dict[str, Any]] = []
    for result in results:
        matrix = result.get("generator_matrix")
        if hasattr(matrix, "astype"):
            matrix_list = matrix.astype(int).tolist()
            seeds.append(
                _seed_record(
                    matrix_list,
                    source="frontier_seed_sat",
                    solution_number=result.get("solution_number"),
                    n=n,
                    k=k,
                )
            )
        cnf_path = result.get("cnf_file")
        solver_path = result.get("solver_path") or resolved
        solver_runs.append(
            {
                "name": f"frontier_seed:d>={selected_frontier_d}",
                "claim_id": claim_id,
                "obligation": f"frontier_seed:d>={selected_frontier_d}",
                "solution_number": result.get("solution_number"),
                "solver": result.get("solver_used") or "cadical",
                "status": result.get("status"),
                "solver_path": solver_path,
                "argv": [solver_path, cnf_path] if cnf_path else [solver_path],
                "command": " ".join([solver_path, cnf_path]) if cnf_path else solver_path,
                "cwd": str(workdir),
                "elapsed_ms": round(float(result.get("solver_time") or 0.0) * 1000, 3),
                "timeout_sec": policy.frontier_seed_timeout_sec,
                "cnf_path": cnf_path,
                "cnf_sha256": file_sha256(cnf_path) if cnf_path and Path(cnf_path).exists() else None,
                "return_code": None,
                "memory_mb": None,
            }
        )
    _write_seed_bank(seed_bank_path, seeds=seeds, policy=policy, solver_runs=solver_runs, claim_id=claim_id)
    status = "OK" if len(seeds) >= policy.frontier_seed_count else "PARTIAL" if seeds else "UNAVAILABLE"
    return {
        "status": status,
        "seeds": seeds,
        "seed_bank": _compact_seed_bank(seeds, artifact_path=seed_bank_path, k=k),
        "solver_runs": solver_runs,
        "missing_obligations": [] if seeds else ["frontier_seed_bank_sat"],
        "artifact_path": str(seed_bank_path),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@traceable_run(
    "hybrid_ga:init_from_seed_bank",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def initialize_population_from_seed_bank(
    *,
    seeds: list[dict[str, Any]],
    policy: HybridGAPolicy,
    n: int = TARGET_N,
    k: int = TARGET_K,
) -> dict[str, Any]:
    matrices = [seed["matrix"] for seed in seeds if isinstance(seed.get("matrix"), list)]
    if not matrices:
        return initialize_population(seed_matrix=None, policy=policy, n=n, k=k)
    started = time.perf_counter()
    rng = np.random.default_rng(policy.seed)
    population = list(matrices[: policy.population])
    while len(population) < policy.population:
        base = matrices[int(rng.integers(0, len(matrices)))]
        population.append(_mutate_matrix(base, rng, rate=0.01 + 0.002 * (len(population) % 10), k=k))
    return {
        "population": population,
        "seed_status": "frontier_seed_bank_initialized",
        "diversity": _diversity(population, k=k),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@traceable_run(
    "hybrid_ga:evolve",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def evolve_population(
    *,
    population: list[list[list[int]]],
    policy: HybridGAPolicy,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
    expected_a_d: int | None = TARGET_A_D,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(policy.seed)
    generation_summaries: list[dict[str, Any]] = []
    repair_candidates: list[dict[str, Any]] = []
    summary, best_matrix = _generation_summary(
        population,
        0,
        n=n,
        k=k,
        target_d=target_d,
        expected_a_d=expected_a_d,
    )
    generation_summaries.append(summary)
    if summary.get("best_d_min", 0) >= target_d and (expected_a_d is None or summary.get("best_A_d") == expected_a_d):
        return {
            "best_matrix": best_matrix,
            "generation_summaries": generation_summaries,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    for generation in range(1, policy.generations + 1):
        scored = sorted(
            ((_score(_diagnostics(matrix, n=n, k=k, d=target_d, expected_a_d=expected_a_d)), matrix) for matrix in population),
            key=lambda item: item[0],
            reverse=True,
        )
        elites = [matrix for _, matrix in scored[: max(1, min(4, len(scored)))]]
        next_population = list(elites)
        while len(next_population) < policy.population:
            parent_a = elites[int(rng.integers(0, len(elites)))]
            parent_b = elites[int(rng.integers(0, len(elites)))]
            cut = int(rng.integers(k + 1, n))
            child = [row[:cut] + other_row[cut:] for row, other_row in zip(parent_a, parent_b, strict=True)]
            next_population.append(_mutate_matrix(child, rng, k=k))
        population = next_population
        summary_interval = max(1, min(10, policy.generations))
        repair_due = bool(policy.repair_interval and generation % policy.repair_interval == 0)
        if generation == policy.generations or generation % summary_interval == 0 or repair_due:
            summary, best_matrix = _generation_summary(
                population,
                generation,
                n=n,
                k=k,
                target_d=target_d,
                expected_a_d=expected_a_d,
            )
            generation_summaries.append(summary)
            if repair_due:
                repair_candidates.append(
                    {
                        "generation": generation,
                        "candidate_id": summary.get("best_candidate_id"),
                        "matrix": best_matrix,
                        "summary": summary,
                    }
                )
            if summary.get("best_d_min", 0) >= target_d and (expected_a_d is None or summary.get("best_A_d") == expected_a_d):
                break
    return {
        "best_matrix": best_matrix,
        "generation_summaries": generation_summaries,
        "repair_candidates": repair_candidates,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@traceable_run(
    "hybrid_ga:sat_repair",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def attempt_sat_repair(
    *,
    candidate: list[list[int]],
    generation: int,
    policy: HybridGAPolicy,
    cadical_path: str | None = None,
    artifact_dir: str | Path = "artifacts",
    claim_id: str = HYBRID_CLAIM_ID,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
    expected_a_d: int | None = TARGET_A_D,
    post_matrix: list[list[int]] | None = None,
    repair_source: dict[str, Any] | None = None,
) -> HybridGARepairEvent:
    started = time.perf_counter()
    repair_source = repair_source or {}
    pre_d = int(repair_source.get("pre_repair_expected_d_min") or max(1, target_d - 1))
    pre_a_d = repair_source.get("pre_repair_expected_A_d")
    frontier_diagnostics = _diagnostics(
        candidate,
        n=n,
        k=k,
        d=pre_d,
        expected_a_d=int(pre_a_d) if pre_a_d is not None else None,
    )
    diagnostics = _diagnostics(candidate, n=n, k=k, d=target_d, expected_a_d=expected_a_d)
    diagnostics["frontier_diagnostics"] = {
        key: value
        for key, value in frontier_diagnostics.items()
        if key
        in {
            "d",
            "minimum_distance",
            "d_min",
            "A_d",
            "A_min",
            "expected_A_d",
            "target_achieved",
            "verification_ok",
            "checks",
        }
    }
    diagnostics["low_weight_below_target"] = _low_weight_summary(candidate, target_d, n=n, k=k)
    diagnostics["target_gap"] = max(0, target_d - int(diagnostics.get("minimum_distance") or 0))
    if post_matrix is not None:
        post_diagnostics = _diagnostics(post_matrix, n=n, k=k, d=target_d, expected_a_d=expected_a_d)
        post_diagnostics["repair_source"] = {
            key: value
            for key, value in repair_source.items()
            if key not in {"pre_repair_generator_matrix"}
        }
        return HybridGARepairEvent(
            generation=generation,
            candidate_id=_candidate_id(candidate),
            pre_repair_diagnostics=diagnostics,
            mutable_columns=list(range(k, n)),
            solver_status=str(repair_source.get("repair_status") or "ARCHIVED_REPAIR_REPLAY"),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            post_repair_diagnostics=post_diagnostics,
        )
    resolved = resolve_cadical(cadical_path)
    if not resolved:
        return HybridGARepairEvent(
            generation=generation,
            candidate_id=_candidate_id(candidate),
            pre_repair_diagnostics=diagnostics,
            mutable_columns=list(range(k, n)),
            solver_status="UNAVAILABLE",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            post_repair_diagnostics=None,
        )
    mutable_columns = _mutable_columns_from_low_weight_supports(candidate, target_d, n=n, k=k)
    workdir = Path(artifact_dir) / "hybrid_ga" / claim_id / "sat_repair"
    workdir.mkdir(parents=True, exist_ok=True)
    for attempt_name, columns in (
        ("low_weight_support_mask", mutable_columns),
        ("all_parity_columns", list(range(k, n))),
    ):
        encoder = CNFEncoder(n, k, target_d, systematic=True)
        encoder.encode_all_constraints()
        fixed_columns = [col for col in range(k, n) if col not in set(columns)]
        for row_idx in range(k):
            for col in fixed_columns:
                var = encoder.var_map[("P", row_idx, col - k)]
                encoder.clauses.append([var if int(candidate[row_idx][col]) else -var])
        cnf_path = workdir / f"repair_gen{generation}_{attempt_name}.cnf"
        encoder.export_to_dimacs(str(cnf_path))
        solver = SATSolver(solver_type="cadical", solver_path=resolved)
        result = solver.solve(str(cnf_path), timeout=policy.timeout_sec, verbose=False)
        if result.get("status") == "SAT":
            parser = ResultParser(n, k, target_d, systematic=True)
            repaired = parser.parse_sat_model(result["model"], encoder.var_map)
            post_diagnostics = _diagnostics(
                repaired.astype(int).tolist(),
                n=n,
                k=k,
                d=target_d,
                expected_a_d=expected_a_d,
            )
            post_diagnostics["solver_run"] = {
                "name": f"sat_repair:{attempt_name}:d>={target_d}",
                "attempt": attempt_name,
                "solver": result.get("solver_used") or "cadical",
                "status": result.get("status"),
                "solver_path": result.get("solver_path") or resolved,
                "argv": result.get("command") or [result.get("solver_path") or resolved, str(cnf_path)],
                "command": " ".join(result.get("command") or [result.get("solver_path") or resolved, str(cnf_path)]),
                "cwd": str(workdir),
                "elapsed_ms": round(float(result.get("time") or 0.0) * 1000, 3),
                "timeout_sec": policy.timeout_sec,
                "cnf_path": str(cnf_path),
                "cnf_sha256": file_sha256(cnf_path),
                "return_code": None,
                "memory_mb": None,
            }
            return HybridGARepairEvent(
                generation=generation,
                candidate_id=_candidate_id(candidate),
                pre_repair_diagnostics=diagnostics,
                mutable_columns=columns,
                solver_status="SAT",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                post_repair_diagnostics=post_diagnostics,
                post_repair_matrix=repaired.astype(int).tolist(),
            )
        if result.get("status") == "TIMEOUT":
            break
    return HybridGARepairEvent(
        generation=generation,
        candidate_id=_candidate_id(candidate),
        pre_repair_diagnostics=diagnostics,
        mutable_columns=mutable_columns,
        solver_status="TIMEOUT" if result.get("status") == "TIMEOUT" else "UNSAT",
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        post_repair_diagnostics=None,
    )


@traceable_run(
    "hybrid_ga:final_verify",
    run_type="chain",
    process_inputs=_sanitize_hybrid_inputs,
    process_outputs=_sanitize_hybrid_outputs,
)
def final_verify_hybrid_ga(
    *,
    matrix: list[list[int]],
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
    expected_a_d: int | None = TARGET_A_D,
) -> dict[str, Any]:
    started = time.perf_counter()
    diagnostics = _diagnostics(matrix, n=n, k=k, d=target_d, expected_a_d=expected_a_d)
    diagnostics["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return diagnostics


@traceable_run("hybrid_ga:report", run_type="chain")
def run_hybrid_ga(
    *,
    policy: HybridGAPolicy,
    artifact_dir: str | Path = "artifacts",
    artifact_path: str | Path | None = None,
    cadical_path: str | None = None,
    claim_id: str = HYBRID_CLAIM_ID,
) -> dict[str, Any]:
    started = time.perf_counter()
    if claim_id not in HYBRID_CLAIM_IDS:
        report = HybridGAReport(
            claim_id=claim_id,
            mode=policy.mode,
            config=_policy_config(policy),
            seed_status="unsupported_claim",
            final_diagnostics={},
            verdict="INSUFFICIENT_ARTIFACT",
            missing_obligations=["unsupported_hybrid_ga_claim"],
        )
        return report.model_dump(mode="python")
    target_cfg = _target_config(claim_id, policy)
    n = int(target_cfg["n"])
    k = int(target_cfg["k"])
    target_d = int(target_cfg["target_d"])
    frontier_d = int(target_cfg["frontier_d"])
    expected_a_d = target_cfg.get("expected_a_d")
    report_config = {**_policy_config(policy), "target": target_cfg}
    if not policy.enabled or policy.mode == "off":
        report = HybridGAReport(
            claim_id=claim_id,
            mode="off",
            config=report_config,
            seed_status="disabled",
            final_diagnostics={},
            verdict="SKIPPED",
            missing_obligations=["hybrid_ga_disabled"],
        )
        return report.model_dump(mode="python")

    if policy.mode == "frontier_repair":
        target = run_frontier_target_sat(
            policy=policy,
            artifact_dir=artifact_dir,
            cadical_path=cadical_path,
            claim_id=claim_id,
            n=n,
            k=k,
            target_d=target_d,
            expected_a_d=expected_a_d,
        )
        frontier_solver_runs = list(target.get("solver_runs") or [])
        report_path = Path(artifact_dir) / "hybrid_ga" / claim_id / "frontier_repair_report.json"
        if target.get("seed_matrix") is not None and (target.get("diagnostics") or {}).get("target_achieved"):
            final_diagnostics = final_verify_hybrid_ga(
                matrix=target["seed_matrix"],
                n=n,
                k=k,
                target_d=target_d,
                expected_a_d=expected_a_d,
            )
            report = HybridGAReport(
                claim_id=claim_id,
                mode=policy.mode,
                config=report_config,
                seed_status="direct_target_sat",
                seed_bank=None,
                frontier_solver_runs=frontier_solver_runs,
                generation_summaries=[
                    {
                        "generation": 0,
                        "event": "frontier_target_sat",
                        "best_candidate_id": _candidate_id(target["seed_matrix"]),
                        "best_d_min": final_diagnostics.get("minimum_distance"),
                        "best_A_d": final_diagnostics.get("A_d"),
                        "best_A_min": (final_diagnostics.get("weight_distribution") or {}).get(final_diagnostics.get("minimum_distance"), 0),
                        "best_fitness": _score(final_diagnostics),
                        "diversity": 0.0,
                    }
                ],
                final_diagnostics={**final_diagnostics, "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
                verdict="PASS",
                missing_obligations=[],
                artifact_paths=[str(report_path)],
            )
            _write_report(report_path, report)
            return report.model_dump(mode="python")

        seed_bank_result = generate_frontier_seed_bank(
            policy=policy,
            artifact_dir=artifact_dir,
            cadical_path=cadical_path,
            claim_id=claim_id,
            n=n,
            k=k,
            frontier_d=frontier_d,
        )
        frontier_solver_runs.extend(seed_bank_result.get("solver_runs") or [])
        seeds = seed_bank_result.get("seeds") or []
        if not seeds:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            report = HybridGAReport(
                claim_id=claim_id,
                mode=policy.mode,
                config=report_config,
                seed_status="frontier_seed_bank_unavailable",
                seed_bank=seed_bank_result.get("seed_bank"),
                frontier_solver_runs=frontier_solver_runs,
                final_diagnostics={
                    "elapsed_ms": elapsed_ms,
                    "target_status": target.get("status"),
                    "seed_bank_status": seed_bank_result.get("status"),
                },
                verdict="PARTIAL",
                missing_obligations=seed_bank_result.get("missing_obligations") or ["frontier_seed_bank"],
                artifact_paths=[path for path in [seed_bank_result.get("artifact_path"), str(report_path)] if path],
            )
            _write_report(report_path, report)
            return report.model_dump(mode="python")

        init = initialize_population_from_seed_bank(seeds=seeds, policy=policy, n=n, k=k)
        evolution = evolve_population(
            population=init["population"],
            policy=policy,
            n=n,
            k=k,
            target_d=target_d,
            expected_a_d=expected_a_d,
        )
        best_matrix = evolution["best_matrix"]
        final_diagnostics = final_verify_hybrid_ga(
            matrix=best_matrix,
            n=n,
            k=k,
            target_d=target_d,
            expected_a_d=expected_a_d,
        )
        repair_events: list[HybridGARepairEvent] = []
        if not final_diagnostics.get("target_achieved") and policy.repair_interval:
            candidates = list(evolution.get("repair_candidates") or [])
            if not candidates:
                candidates = [
                    {
                        "generation": int(policy.repair_interval),
                        "candidate_id": _candidate_id(best_matrix),
                        "matrix": best_matrix,
                        "summary": evolution["generation_summaries"][-1] if evolution.get("generation_summaries") else {},
                    }
                ]
            seen_candidate_ids: set[str] = set()
            unique_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                candidate_id = str(candidate.get("candidate_id") or _candidate_id(candidate.get("matrix", best_matrix)))
                if candidate_id in seen_candidate_ids:
                    continue
                seen_candidate_ids.add(candidate_id)
                unique_candidates.append(candidate)
            for candidate in unique_candidates[:3]:
                repair_matrix = candidate.get("matrix") if isinstance(candidate.get("matrix"), list) else best_matrix
                repair_generation = int(candidate.get("generation") or policy.repair_interval)
                repair_event = attempt_sat_repair(
                    candidate=repair_matrix,
                    generation=repair_generation,
                    policy=policy,
                    cadical_path=cadical_path,
                    artifact_dir=artifact_dir,
                    claim_id=claim_id,
                    n=n,
                    k=k,
                    target_d=target_d,
                    expected_a_d=expected_a_d,
                )
                repair_events.append(repair_event)
                if repair_event.post_repair_matrix is not None:
                    best_matrix = repair_event.post_repair_matrix
                    final_diagnostics = final_verify_hybrid_ga(
                        matrix=best_matrix,
                        n=n,
                        k=k,
                        target_d=target_d,
                        expected_a_d=expected_a_d,
                    )
                    evolution["generation_summaries"].append(
                        {
                            "generation": repair_generation,
                            "event": "sat_repair",
                            "best_candidate_id": _candidate_id(best_matrix),
                            "best_d_min": final_diagnostics.get("minimum_distance"),
                            "best_A_d": final_diagnostics.get("A_d"),
                            "best_A_min": (final_diagnostics.get("weight_distribution") or {}).get(final_diagnostics.get("minimum_distance"), 0),
                            "best_fitness": _score(final_diagnostics),
                            "diversity": init["diversity"],
                            "pre_repair_d_min": repair_event.pre_repair_diagnostics.get("minimum_distance"),
                            "pre_repair_A_d": repair_event.pre_repair_diagnostics.get("A_d"),
                            "solver_status": repair_event.solver_status,
                        }
                    )
                    if final_diagnostics.get("target_achieved"):
                        break
                if repair_event.solver_status == "TIMEOUT":
                    break
        verdict = "PASS" if final_diagnostics.get("target_achieved") else "PARTIAL"
        missing = [] if verdict == "PASS" else ["frontier_repair_did_not_reach_expected_target"]
        report = HybridGAReport(
            claim_id=claim_id,
            mode=policy.mode,
            config=report_config,
            seed_status=init["seed_status"],
            seed_bank=seed_bank_result.get("seed_bank"),
            frontier_solver_runs=frontier_solver_runs,
            generation_summaries=evolution["generation_summaries"],
            repair_events=repair_events,
            final_diagnostics={
                **final_diagnostics,
                "initial_diversity": init["diversity"],
                "replay_elapsed_ms": evolution["elapsed_ms"],
                "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "target_sat_status": target.get("status"),
                "seed_bank_status": seed_bank_result.get("status"),
            },
            verdict=verdict,
            missing_obligations=missing,
            artifact_paths=[path for path in [seed_bank_result.get("artifact_path"), str(report_path)] if path],
        )
        _write_report(report_path, report)
        update_current_run_context(
            metadata={
                "claim_id": claim_id,
                "hybrid_ga_mode": policy.mode,
                "hybrid_ga_verdict": verdict,
                "hybrid_ga_seed_count": (report.seed_bank or {}).get("seed_count"),
                "hybrid_ga_generation_count": len(report.generation_summaries),
                "hybrid_ga_repair_count": len(report.repair_events),
                "hybrid_ga_solver_elapsed_ms": sum(run.get("elapsed_ms", 0) for run in frontier_solver_runs) + sum(event.elapsed_ms for event in report.repair_events),
                "elapsed_ms": report.final_diagnostics.get("total_elapsed_ms"),
            },
            tags=["hybrid-ga", claim_id, "frontier-repair"],
        )
        return report.model_dump(mode="python")

    live_seed: dict[str, Any] | None = None
    if policy.mode == "live":
        live_seed = generate_live_seed(policy=policy, artifact_dir=artifact_dir, cadical_path=cadical_path)
        if not live_seed.get("seed_matrix"):
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            report = HybridGAReport(
                claim_id=claim_id,
                mode=policy.mode,
                config=_policy_config(policy),
                seed_status=live_seed.get("seed_status") or "live_seed_unavailable",
                generation_summaries=[],
                solver_runs=live_seed.get("solver_runs") or [],
                final_diagnostics={"elapsed_ms": elapsed_ms, "live_seed_status": live_seed.get("status")},
                verdict="PARTIAL",
                missing_obligations=live_seed.get("missing_obligations") or ["live_seed_generation"],
                artifact_paths=[],
            )
            return report.model_dump(mode="python")

    archived_report = verify_archived_hybrid_ga(policy=policy, artifact_dir=artifact_dir, artifact_path=artifact_path)
    if policy.mode == "archived" or archived_report.verdict != "PASS":
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        payload = archived_report.model_dump(mode="python")
        payload["final_diagnostics"]["total_elapsed_ms"] = elapsed_ms
        update_current_run_context(
            metadata={
                "claim_id": claim_id,
                "hybrid_ga_mode": policy.mode,
                "hybrid_ga_verdict": archived_report.verdict,
                "hybrid_ga_generation_count": len(archived_report.generation_summaries),
                "hybrid_ga_repair_count": len(archived_report.repair_events),
                "hybrid_ga_solver_elapsed_ms": sum(run.get("elapsed_ms", 0) for run in archived_report.solver_runs),
                "elapsed_ms": elapsed_ms,
            },
            tags=["hybrid-ga", claim_id],
        )
        return payload

    payload, _, _ = _load_hybrid_artifact(artifact_dir=artifact_dir, artifact_path=artifact_path)
    repair_replay = (payload or {}).get("repair_replay") if policy.mode == "replay" else None
    if live_seed and live_seed.get("seed_matrix"):
        seed_matrix = live_seed["seed_matrix"]
        seed_status_override = live_seed.get("seed_status")
    elif isinstance(repair_replay, dict) and policy.repair_interval:
        seed_matrix = _matrix_from_repair_payload(repair_replay)
        seed_status_override = "archived_d6_seed_for_sat_repair_replay"
    else:
        seed_matrix = _matrix_from_payload(payload or {})
        seed_status_override = None
    init = initialize_population(seed_matrix=seed_matrix, policy=policy)
    evolution = evolve_population(population=init["population"], policy=policy)
    repair_events: list[HybridGARepairEvent] = []
    best_matrix = evolution["best_matrix"]
    if policy.repair_interval and policy.mode in {"replay", "live"}:
        last_generation = int((evolution["generation_summaries"][-1] or {}).get("generation", 0))
        if isinstance(repair_replay, dict):
            repair_generation = int(repair_replay.get("repair_generation") or max(policy.repair_interval, last_generation))
            repair_event = attempt_sat_repair(
                candidate=seed_matrix,
                generation=repair_generation,
                policy=policy,
                cadical_path=cadical_path,
                post_matrix=_matrix_from_payload(payload or {}),
                repair_source=repair_replay,
            )
            repair_events.append(repair_event)
            if repair_event.post_repair_diagnostics and repair_event.post_repair_diagnostics.get("target_achieved"):
                best_matrix = _matrix_from_payload(payload or {})
                evolution["generation_summaries"].append(
                    {
                        "generation": repair_generation,
                        "event": "sat_repair",
                        "best_candidate_id": _candidate_id(best_matrix),
                        "best_d_min": repair_event.post_repair_diagnostics.get("minimum_distance"),
                        "best_A_d": repair_event.post_repair_diagnostics.get("A_d"),
                        "best_A_min": (
                            repair_event.post_repair_diagnostics.get("weight_distribution") or {}
                        ).get(repair_event.post_repair_diagnostics.get("minimum_distance"), 0),
                        "best_fitness": _score(repair_event.post_repair_diagnostics),
                        "diversity": _diversity(init["population"]),
                        "pre_repair_d_min": repair_event.pre_repair_diagnostics.get("minimum_distance"),
                        "pre_repair_A_d": repair_event.pre_repair_diagnostics.get("A_d"),
                        "solver_status": repair_event.solver_status,
                    }
                )
        elif last_generation and last_generation % policy.repair_interval == 0:
            repair_events.append(
                attempt_sat_repair(
                    candidate=best_matrix,
                    generation=last_generation,
                    policy=policy,
                    cadical_path=cadical_path,
                )
            )
    final_diagnostics = final_verify_hybrid_ga(matrix=best_matrix)
    verdict = "PASS" if final_diagnostics.get("target_achieved") else "PARTIAL"
    missing = [] if verdict == "PASS" else ["hybrid_ga_replay_did_not_reach_archived_target"]
    solver_runs = live_seed.get("solver_runs", []) if live_seed else []
    report_path = Path(artifact_dir) / "hybrid_ga" / claim_id / f"{policy.mode}_report.json"
    report = HybridGAReport(
        claim_id=claim_id,
        mode=policy.mode,
        config=_policy_config(policy),
        seed_status=seed_status_override or (live_seed.get("seed_status") if live_seed else init["seed_status"]),
        generation_summaries=evolution["generation_summaries"],
        repair_events=repair_events,
        solver_runs=solver_runs,
        final_diagnostics={
            **final_diagnostics,
            "initial_diversity": init["diversity"],
            "replay_elapsed_ms": evolution["elapsed_ms"],
            "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        verdict=verdict,
        missing_obligations=missing,
        artifact_paths=[str(report_path)],
    )
    _write_report(report_path, report)
    update_current_run_context(
        metadata={
            "claim_id": claim_id,
            "hybrid_ga_mode": policy.mode,
            "hybrid_ga_verdict": verdict,
            "hybrid_ga_generation_count": len(report.generation_summaries),
            "hybrid_ga_repair_count": len(report.repair_events),
            "hybrid_ga_solver_elapsed_ms": sum(event.elapsed_ms for event in report.repair_events),
            "elapsed_ms": report.final_diagnostics.get("total_elapsed_ms"),
        },
        tags=["hybrid-ga", claim_id],
    )
    return report.model_dump(mode="python")
