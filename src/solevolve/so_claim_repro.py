from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from itertools import combinations
from pathlib import Path
from shutil import which
from typing import Any

import numpy as np

from .claim_registry import ClaimSpec, expand_claim_ids, get_claim
from .finite_fields import (
    canonical_nonzero_messages,
    codeword,
    field_add,
    field_conjugate,
    field_mul,
    gram_matrix,
    minimum_distance,
    rank_field,
    weight_distribution,
)
from .tracing import flush_tracing, traceable_run, update_current_run_context

ROOT_DIR = Path(__file__).resolve().parents[2]


def _child_memory_mb() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except Exception:
        return None
    if usage <= 0:
        return None
    # macOS reports bytes; most Linux builds report KiB. Keep the value approximate.
    return round(usage / (1024 * 1024) if usage > 10_000_000 else usage / 1024, 3)


class _CnfBuilder:
    def __init__(self) -> None:
        self.next_var = 1
        self.clauses: list[list[int]] = []

    def new_var(self) -> int:
        value = self.next_var
        self.next_var += 1
        return value

    def exactly_one(self, variables: list[int]) -> None:
        self.clauses.append(list(variables))
        for i, left in enumerate(variables):
            for right in variables[i + 1 :]:
                self.clauses.append([-left, -right])

    @property
    def var_count(self) -> int:
        return self.next_var - 1


def _user_cadical_candidate() -> Path:
    return (
        Path.home()
        / "Desktop"
        / "CodingTheoryLib"
        / "references"
        / ("Code" + "Evolve")
        / "sat_solvers"
        / "cadical"
    )


def resolve_cadical(path: str | None) -> str | None:
    def first_executable(paths: list[Path]) -> str | None:
        for candidate in paths:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return None

    candidates: list[Path] = []
    if path:
        candidate = Path(path)
        candidates.append(candidate)
        if candidate.is_dir():
            candidates.extend([candidate / "build" / "cadical", candidate / "cadical"])
        return first_executable(candidates)

    env_path = os.getenv("CADICAL_PATH")
    if env_path:
        env_candidate = Path(env_path)
        candidates.append(env_candidate)
        if env_candidate.is_dir():
            candidates.extend([env_candidate / "build" / "cadical", env_candidate / "cadical"])
    repo_candidate = ROOT_DIR / "cadical"
    candidates.extend([repo_candidate, repo_candidate / "build" / "cadical", repo_candidate / "cadical"])
    user_candidate = _user_cadical_candidate()
    candidates.extend([user_candidate, user_candidate / "build" / "cadical", user_candidate / "cadical"])
    found = which("cadical")
    if found:
        candidates.append(Path(found))
    return first_executable(candidates)


def _encode_extension_coordinate_sum(
    cnf: _CnfBuilder,
    value_vars: dict[tuple[int, int, int], int],
    *,
    q: int,
    message: tuple[int, ...],
    col: int,
) -> dict[int, int]:
    state: dict[tuple[int, int], int] = {}
    k = len(message)
    for pos in range(k + 1):
        variables = []
        for residue in range(q):
            var = cnf.new_var()
            state[(pos, residue)] = var
            variables.append(var)
        cnf.exactly_one(variables)

    cnf.clauses.append([state[(0, 0)]])
    for row, coefficient in enumerate(message):
        for current in range(q):
            for value in range(q):
                next_residue = field_add(q, current, field_mul(q, coefficient, value))
                cnf.clauses.append(
                    [
                        -state[(row, current)],
                        -value_vars[(row, col, value)],
                        state[(row + 1, next_residue)],
                    ]
                )
    return {residue: state[(k, residue)] for residue in range(q)}


def _require_at_least_nonzero_coordinates(
    cnf: _CnfBuilder,
    coordinate_residue_vars: list[dict[int, int]],
    *,
    q: int,
    required: int,
) -> int:
    if required <= 0:
        return 0
    coordinate_count = len(coordinate_residue_vars)
    if required > coordinate_count:
        cnf.clauses.append([])
        return 1

    before = len(cnf.clauses)
    subset_size = coordinate_count - required + 1
    for subset in combinations(range(coordinate_count), subset_size):
        clause: list[int] = []
        for col in subset:
            clause.extend(coordinate_residue_vars[col][residue] for residue in range(1, q))
        cnf.clauses.append(clause)
    return len(cnf.clauses) - before


def _message_fixed_weight(message: tuple[int, ...], generator: np.ndarray, q: int) -> int:
    return int(np.count_nonzero(codeword(message, generator, q)))


def _add_minimum_distance_constraints(
    cnf: _CnfBuilder,
    value_vars: dict[tuple[int, int, int], int],
    spec: ClaimSpec,
    *,
    target: int,
) -> dict[str, Any]:
    q = spec.field_order
    representatives = canonical_nonzero_messages(spec.k, q)
    fixed_weight_distribution: dict[int, int] = {}
    encoded_by_fixed_weight: dict[int, int] = {}
    cardinality_clauses = 0
    impossible_classes = 0
    encoded_classes = 0
    skipped_classes = 0
    start_var_count = cnf.var_count
    start_clause_count = len(cnf.clauses)

    for message in representatives:
        fixed_weight = _message_fixed_weight(message, spec.base_matrix, q)
        fixed_weight_distribution[fixed_weight] = fixed_weight_distribution.get(fixed_weight, 0) + 1
        required = target - fixed_weight
        if required <= 0:
            skipped_classes += 1
            continue
        encoded_classes += 1
        encoded_by_fixed_weight[fixed_weight] = encoded_by_fixed_weight.get(fixed_weight, 0) + 1
        if required > spec.t:
            impossible_classes += 1
            cnf.clauses.append([])
            continue
        coordinate_residue_vars = [
            _encode_extension_coordinate_sum(cnf, value_vars, q=q, message=message, col=col)
            for col in range(spec.t)
        ]
        cardinality_clauses += _require_at_least_nonzero_coordinates(
            cnf,
            coordinate_residue_vars,
            q=q,
            required=required,
        )

    return {
        "minimum_distance_target": target,
        "message_scalar_classes": len(representatives),
        "message_representatives_used": True,
        "distance_encoded_classes": encoded_classes,
        "distance_skipped_classes": skipped_classes,
        "distance_impossible_classes": impossible_classes,
        "base_code_fixed_weight_distribution_by_class": dict(sorted(fixed_weight_distribution.items())),
        "distance_encoded_classes_by_fixed_weight": dict(sorted(encoded_by_fixed_weight.items())),
        "distance_auxiliary_variables": cnf.var_count - start_var_count,
        "distance_added_clauses": len(cnf.clauses) - start_clause_count,
        "distance_cardinality_clauses": cardinality_clauses,
    }


def _add_self_orthogonality_constraints(
    cnf: _CnfBuilder,
    value_vars: dict[tuple[int, int, int], int],
    spec: ClaimSpec,
) -> dict[str, Any]:
    q = spec.field_order
    base_gram = gram_matrix(spec.base_matrix, q, hermitian=spec.hermitian)
    start_var_count = cnf.var_count
    start_clause_count = len(cnf.clauses)

    for i in range(spec.k):
        row_range = range(spec.k) if spec.hermitian else range(i, spec.k)
        for j in row_range:
            state: dict[tuple[int, int], int] = {}
            for pos in range(spec.t + 1):
                variables = []
                for residue in range(q):
                    var = cnf.new_var()
                    state[(pos, residue)] = var
                    variables.append(var)
                cnf.exactly_one(variables)
            cnf.clauses.append([state[(0, int(base_gram[i, j]))]])

            for col in range(spec.t):
                product_vars = []
                product: dict[int, int] = {}
                for residue in range(q):
                    var = cnf.new_var()
                    product[residue] = var
                    product_vars.append(var)
                cnf.exactly_one(product_vars)

                for a in range(q):
                    for b in range(q):
                        rhs = field_conjugate(q, b) if spec.hermitian else b
                        cnf.clauses.append(
                            [
                                -value_vars[(i, col, a)],
                                -value_vars[(j, col, b)],
                                product[field_mul(q, a, rhs)],
                            ]
                        )
                for current in range(q):
                    for prod in range(q):
                        cnf.clauses.append(
                            [
                                -state[(col, current)],
                                -product[prod],
                                state[(col + 1, field_add(q, current, prod))],
                            ]
                        )
            cnf.clauses.append([state[(spec.t, 0)]])

    return {
        "base_gram_rank": rank_field(base_gram, q),
        "self_orthogonality_auxiliary_variables": cnf.var_count - start_var_count,
        "self_orthogonality_added_clauses": len(cnf.clauses) - start_clause_count,
    }


def build_so_extension_cnf(path: Path, spec: ClaimSpec, *, min_distance: int | None) -> dict[str, Any]:
    q = spec.field_order
    cnf = _CnfBuilder()
    x: dict[tuple[int, int, int], int] = {}
    for row in range(spec.k):
        for col in range(spec.t):
            variables = []
            for value in range(q):
                var = cnf.new_var()
                x[(row, col, value)] = var
                variables.append(var)
            cnf.exactly_one(variables)

    so_info = _add_self_orthogonality_constraints(cnf, x, spec)
    distance_info: dict[str, Any] = {
        "minimum_distance_target": min_distance,
        "distance_constraints": "not_encoded" if min_distance is None else "encoded",
    }
    if min_distance is not None:
        distance_info.update(_add_minimum_distance_constraints(cnf, x, spec, target=min_distance))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"p cnf {cnf.var_count} {len(cnf.clauses)}\n")
        for clause in cnf.clauses:
            handle.write(" ".join(str(literal) for literal in clause) + " 0\n")

    return {
        "claim_id": spec.claim_id,
        "cnf_path": str(path),
        "field_order": q,
        "variables": cnf.var_count,
        "clauses": len(cnf.clauses),
        "rows": spec.k,
        "source_length": spec.start_n,
        "added_columns": spec.t,
        "inner_product": spec.inner_product,
        **so_info,
        **distance_info,
        "_value_vars": {f"{row},{col},{value}": var for (row, col, value), var in x.items()},
    }


def _parse_cadical_status(output: str, returncode: int) -> str:
    if "s SATISFIABLE" in output or returncode == 10:
        return "SAT"
    if "s UNSATISFIABLE" in output or returncode == 20:
        return "UNSAT"
    return "UNKNOWN"


def _parse_true_vars(output: str) -> list[int]:
    true_vars: list[int] = []
    for line in output.splitlines():
        if not line.startswith("v "):
            continue
        for token in line[2:].split():
            literal = int(token)
            if literal == 0:
                break
            if literal > 0:
                true_vars.append(literal)
    return true_vars


@traceable_run("solevolve.so_claim_sat", run_type="tool")
def run_so_claim_sat(cnf_path: str, cadical_path: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    cnf = Path(cnf_path)
    solver = Path(cadical_path)
    cwd = str(Path.cwd())
    argv = [str(solver), str(cnf)]
    command = " ".join(argv)
    cnf_resolved = str(cnf.resolve()) if cnf.exists() else str(cnf)
    solver_resolved = str(solver.resolve()) if solver.exists() else str(solver)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        true_vars = _parse_true_vars(output)
        payload = {
            "status": _parse_cadical_status(output, proc.returncode),
            "returncode": proc.returncode,
            "solver": "cadical",
            "solver_path": str(solver),
            "solver_resolved_path": solver_resolved,
            "cnf_path": str(cnf),
            "cnf_resolved_path": cnf_resolved,
            "cwd": cwd,
            "argv": argv,
            "command": command,
            "timeout": timeout,
            "solver_elapsed_ms": elapsed_ms,
            "memory_mb": _child_memory_mb(),
            "model_true_var_count": len(true_vars),
            "model_true_vars": true_vars,
            "stdout_preview": proc.stdout[:1200],
            "stderr_preview": proc.stderr[:1200],
            "output_preview": output[:1200],
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        payload = {
            "status": "TIMEOUT",
            "solver": "cadical",
            "solver_path": str(solver),
            "solver_resolved_path": solver_resolved,
            "cnf_path": str(cnf),
            "cnf_resolved_path": cnf_resolved,
            "cwd": cwd,
            "argv": argv,
            "command": command,
            "solver_elapsed_ms": elapsed_ms,
            "timeout": timeout,
            "output_preview": str(exc)[:1200],
            "memory_mb": _child_memory_mb(),
        }
    update_current_run_context(
        metadata={
            "solver_actual_run": True,
            "solver_backend": "cadical",
            "solver_path": str(solver),
            "solver_resolved_path": solver_resolved,
            "solver_command": command,
            "solver_cwd": cwd,
            "cnf_path": str(cnf),
            "cnf_resolved_path": cnf_resolved,
            "solver_timeout_s": timeout,
            "solver_returncode": payload.get("returncode"),
            "solver_elapsed_ms": payload.get("solver_elapsed_ms"),
            "solver_memory_mb": payload.get("memory_mb"),
            "solver_status": payload.get("status"),
            "model_true_var_count": payload.get("model_true_var_count"),
        }
    )
    return payload


def _decode_extension_from_model(value_vars: dict[str, int], true_vars: list[int], *, rows: int, cols: int, q: int) -> np.ndarray:
    true_set = set(true_vars)
    extension = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            for value in range(q):
                if value_vars[f"{row},{col},{value}"] in true_set:
                    extension[row, col] = value
                    break
    return extension


def _drop_solver_model(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    compact.pop("model_true_vars", None)
    return compact


def verify_claim_matrix(spec: ClaimSpec) -> dict[str, Any]:
    extended = spec.extended_matrix
    base_gram = gram_matrix(spec.base_matrix, spec.field_order, hermitian=spec.hermitian)
    extended_gram = gram_matrix(extended, spec.field_order, hermitian=spec.hermitian)
    distribution = weight_distribution(extended, spec.field_order)
    d_min = minimum_distance(extended, spec.field_order)
    return {
        "claim_id": spec.claim_id,
        "field_order": spec.field_order,
        "inner_product": spec.inner_product,
        "source_parameters": [spec.start_n, spec.k],
        "extended_parameters": [spec.n_prime, spec.k],
        "target_minimum_distance": spec.d,
        "gram_rank": rank_field(base_gram, spec.field_order),
        "witness_t": spec.t,
        "self_orthogonal": bool(np.all(extended_gram == 0)),
        "minimum_distance": d_min,
        "meets_minimum_distance_target": d_min is not None and d_min >= spec.d,
        "weight_distribution": distribution,
        "matches_expected_weight_distribution": distribution == spec.expected_weight_distribution,
        "matrix_ok": bool(np.all(extended_gram == 0))
        and d_min is not None
        and d_min >= spec.d
        and distribution == spec.expected_weight_distribution,
    }


def _decode_sat_witness(
    *,
    spec: ClaimSpec,
    solver_payload: dict[str, Any],
    value_vars: dict[str, int],
) -> dict[str, Any] | None:
    if solver_payload.get("status") != "SAT" or not solver_payload.get("model_true_vars"):
        return None
    extension = _decode_extension_from_model(
        value_vars,
        solver_payload["model_true_vars"],
        rows=spec.k,
        cols=spec.t,
        q=spec.field_order,
    )
    extended = np.hstack([spec.base_matrix, extension])
    witness_spec = spec.model_copy(update={"matrix": extended.tolist()})
    payload = verify_claim_matrix(witness_spec)
    payload["s_matrix"] = extension.tolist()
    return payload


@traceable_run("solevolve.reproduce_paper_claim", run_type="chain")
def reproduce_paper_claim(
    *,
    claim_id: str,
    artifact_dir: str | Path = "artifacts",
    cadical_path: str | None = None,
    timeout: int = 300,
    prove_optimality: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    spec = get_claim(claim_id)
    artifact_root = Path(artifact_dir) / "paper_claims" / spec.claim_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    matrix_verification = verify_claim_matrix(spec)
    selected_solver = resolve_cadical(cadical_path)

    cnf_info = build_so_extension_cnf(artifact_root / f"{spec.claim_id}_d{spec.d}.cnf", spec, min_distance=spec.d)
    value_vars = cnf_info.pop("_value_vars")
    if selected_solver:
        solver_payload = run_so_claim_sat(cnf_info["cnf_path"], selected_solver, timeout)
    else:
        solver_payload = {
            "status": "SKIPPED",
            "solver": "cadical",
            "solver_path": None,
            "reason": "No executable CaDiCaL binary found.",
        }
    sat_witness = _decode_sat_witness(spec=spec, solver_payload=solver_payload, value_vars=value_vars)
    solver = _drop_solver_model(solver_payload)

    optimality_cnf_info = None
    optimality_solver = None
    optimality_witness = None
    if prove_optimality and selected_solver:
        optimality_target = spec.d + 1
        optimality_cnf_info = build_so_extension_cnf(
            artifact_root / f"{spec.claim_id}_d{optimality_target}.cnf",
            spec,
            min_distance=optimality_target,
        )
        optimality_value_vars = optimality_cnf_info.pop("_value_vars")
        optimality_solver_payload = run_so_claim_sat(optimality_cnf_info["cnf_path"], selected_solver, timeout)
        optimality_witness = _decode_sat_witness(
            spec=spec,
            solver_payload=optimality_solver_payload,
            value_vars=optimality_value_vars,
        )
        optimality_solver = _drop_solver_model(optimality_solver_payload)

    matrix_ok = matrix_verification["matrix_ok"]
    target_sat_ok = (
        solver.get("status") == "SAT"
        and sat_witness is not None
        and sat_witness.get("self_orthogonal") is True
        and (sat_witness.get("minimum_distance") or 0) >= spec.d
    )
    if spec.require_optimality_unsat:
        optimality_ok = bool(optimality_solver and optimality_solver.get("status") == "UNSAT")
    else:
        optimality_ok = True
    contradiction = solver.get("status") == "UNSAT" or (
        spec.require_optimality_unsat and optimality_solver is not None and optimality_solver.get("status") == "SAT"
    )

    if matrix_ok and target_sat_ok and optimality_ok:
        verdict = "PASS"
    elif contradiction:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"

    if optimality_solver is None:
        optional_status = "SKIPPED"
    else:
        optional_status = str(optimality_solver.get("status") or "UNKNOWN")

    if optional_status == "UNSAT":
        optional_obligation_status = "PASS"
    elif spec.require_optimality_unsat:
        optional_obligation_status = optional_status
    else:
        optional_obligation_status = "PARTIAL"

    proof_obligations = [
        {
            "name": "explicit_matrix_verification",
            "status": "PASS" if matrix_ok else "FAIL",
            "method": f"GF({spec.field_order}) matrix verifier",
            "minimum_distance": matrix_verification["minimum_distance"],
            "self_orthogonal": matrix_verification["self_orthogonal"],
            "matches_expected_weight_distribution": matrix_verification["matches_expected_weight_distribution"],
        },
        {
            "name": "sat_feasibility_minimum_distance_target",
            "status": "PASS" if target_sat_ok else solver.get("status", "UNKNOWN"),
            "method": "SAT(CaDiCaL)",
            "target_minimum_distance": spec.d,
            "solver_status": solver.get("status"),
            "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
            "minimum_distance": sat_witness.get("minimum_distance") if sat_witness else None,
        },
    ]
    if prove_optimality:
        proof_obligations.append(
            {
                "name": "optional_no_larger_minimum_distance" if not spec.require_optimality_unsat else "optimality_no_larger_minimum_distance",
                "required": spec.require_optimality_unsat,
                "status": optional_obligation_status,
                "method": "SAT(CaDiCaL)",
                "target_minimum_distance": spec.d + 1,
                "solver_status": optional_status,
                "solver_elapsed_ms": (optimality_solver or {}).get("solver_elapsed_ms"),
                "summary": f"UNSAT for d>={spec.d + 1} certifies no stronger fixed-base embedding at t={spec.t}.",
            }
        )

    report = {
        "result_id": spec.claim_id,
        "title": spec.title,
        "verdict": verdict,
        "field_order": spec.field_order,
        "inner_product": spec.inner_product,
        "source_parameters": [spec.start_n, spec.k],
        "extended_parameters": [spec.n_prime, spec.k, matrix_verification["minimum_distance"]],
        "target_minimum_distance": spec.d,
        "grassl_bound": spec.grassl_bound,
        "cadical_preferred": True,
        "prove_optimality": prove_optimality,
        "optimality_required": spec.require_optimality_unsat,
        "proof_obligations": proof_obligations,
        "matrix_verification": matrix_verification,
        "cnf": cnf_info,
        "solver": solver,
        "sat_witness": sat_witness,
        "optimality_cnf": optimality_cnf_info,
        "optimality_solver": optimality_solver,
        "optimality_witness": optimality_witness,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    report_path = artifact_root / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    update_current_run_context(
        metadata={
            "run_kind": "paper_claim_repro",
            "result_id": spec.claim_id,
            "verdict": verdict,
            "field_order": spec.field_order,
            "inner_product": spec.inner_product,
            "solver_actual_run": selected_solver is not None,
            "solver_backend": "cadical",
            "solver_path": solver.get("solver_path"),
            "solver_resolved_path": solver.get("solver_resolved_path"),
            "solver_command": solver.get("command"),
            "solver_cwd": solver.get("cwd"),
            "solver_returncode": solver.get("returncode"),
            "solver_status": solver.get("status"),
            "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
            "target_minimum_distance": spec.d,
            "minimum_distance": matrix_verification.get("minimum_distance"),
            "optimality_target": spec.d + 1 if prove_optimality else None,
            "optimality_required": spec.require_optimality_unsat,
            "optimality_solver_status": (optimality_solver or {}).get("status"),
            "optimality_solver_elapsed_ms": (optimality_solver or {}).get("solver_elapsed_ms"),
            "cnf_path": cnf_info.get("cnf_path"),
            "cnf_variables": cnf_info.get("variables"),
            "cnf_clauses": cnf_info.get("clauses"),
            "optimality_cnf_path": (optimality_cnf_info or {}).get("cnf_path"),
            "optimality_cnf_variables": (optimality_cnf_info or {}).get("variables"),
            "optimality_cnf_clauses": (optimality_cnf_info or {}).get("clauses"),
            "elapsed_ms": report["elapsed_ms"],
        },
        tags=["paper-claim", spec.claim_id, "cadical"],
    )
    return report


@traceable_run("solevolve.reproduce_paper_claims", run_type="chain")
def reproduce_paper_claims(
    *,
    claim_id: str,
    artifact_dir: str | Path = "artifacts",
    cadical_path: str | None = None,
    timeout: int = 300,
    prove_optimality: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    reports = [
        reproduce_paper_claim(
            claim_id=expanded,
            artifact_dir=artifact_dir,
            cadical_path=cadical_path,
            timeout=timeout,
            prove_optimality=prove_optimality,
        )
        for expanded in expand_claim_ids(claim_id)
    ]
    if not reports:
        return {
            "result_id": claim_id,
            "verdict": "SKIPPED",
            "reports": [],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    verdict = "PASS" if all(report.get("verdict") == "PASS" for report in reports) else "FAIL" if any(report.get("verdict") == "FAIL" for report in reports) else "PARTIAL"
    artifact_root = Path(artifact_dir) / "paper_claims"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "result_id": claim_id,
        "verdict": verdict,
        "claim_ids": [report["result_id"] for report in reports],
        "reports": reports,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    summary_path = artifact_root / f"{claim_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["report_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce paper-aligned self-orthogonal embedding claims.")
    parser.add_argument("--paper-claim-id", default=os.getenv("SOLEVOLVE_PAPER_CLAIM_ID", "ternary_bch_d9"))
    parser.add_argument("--artifact-dir", default=os.getenv("SOLEVOLVE_ARTIFACT_DIR", "artifacts"))
    parser.add_argument("--cadical-path", default=os.getenv("CADICAL_PATH"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("SOLEVOLVE_PAPER_CLAIM_TIMEOUT", "300")))
    parser.add_argument("--skip-optimality-check", action="store_true")
    args = parser.parse_args()
    try:
        report = reproduce_paper_claims(
            claim_id=args.paper_claim_id,
            artifact_dir=args.artifact_dir,
            cadical_path=args.cadical_path,
            timeout=args.timeout,
            prove_optimality=not args.skip_optimality_check,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        flush_tracing()


if __name__ == "__main__":
    main()
