from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .claim_registry import TERNARY_BCH_S, ternary_bch_base_matrix
from .so_claim_repro import resolve_cadical
from .tracing import flush_tracing, traceable_run, update_current_run_context


ROOT_DIR = Path(__file__).resolve().parents[2]


def _bch_generator_matrix() -> np.ndarray:
    """Return the standard BCH(13,7,5) generator matrix over GF(3)."""
    return np.array(ternary_bch_base_matrix(), dtype=int) % 3


def _summary_witness_s() -> np.ndarray:
    return np.array(TERNARY_BCH_S, dtype=int) % 3


def _rank_mod_p(matrix: np.ndarray, p: int) -> int:
    work = np.array(matrix, dtype=int, copy=True) % p
    rows, cols = work.shape
    rank = 0
    pivot_col = 0
    for row in range(rows):
        while pivot_col < cols and not np.any(work[row:, pivot_col] % p):
            pivot_col += 1
        if pivot_col >= cols:
            break
        pivot_candidates = np.where(work[row:, pivot_col] % p != 0)[0]
        pivot_row = row + int(pivot_candidates[0])
        if pivot_row != row:
            work[[row, pivot_row]] = work[[pivot_row, row]]
        inverse = pow(int(work[row, pivot_col]), -1, p)
        work[row] = (work[row] * inverse) % p
        for other in range(rows):
            if other != row and work[other, pivot_col] % p:
                work[other] = (work[other] - work[other, pivot_col] * work[row]) % p
        rank += 1
        pivot_col += 1
    return rank


def _weight_distribution(generator: np.ndarray, p: int) -> dict[int, int]:
    k, _ = generator.shape
    distribution: dict[int, int] = {}
    for index in range(p**k):
        value = index
        coeffs = []
        for _ in range(k):
            coeffs.append(value % p)
            value //= p
        message = np.array(coeffs, dtype=int)
        codeword = (message @ generator) % p
        weight = int(np.count_nonzero(codeword))
        distribution[weight] = distribution.get(weight, 0) + 1
    return dict(sorted(distribution.items()))


def _nonzero_message_representatives(k: int, p: int) -> list[tuple[int, ...]]:
    """Return one representative for each nonzero scalar class over GF(p)."""
    representatives: list[tuple[int, ...]] = []
    for index in range(1, p**k):
        value = index
        coeffs = []
        for _ in range(k):
            coeffs.append(value % p)
            value //= p
        first_nonzero = next((coeff for coeff in coeffs if coeff), 0)
        if first_nonzero == 1:
            representatives.append(tuple(int(coeff) for coeff in coeffs))
    return representatives


def _message_fixed_weight(message: tuple[int, ...], generator: np.ndarray, p: int) -> int:
    codeword = (np.array(message, dtype=int) @ generator) % p
    return int(np.count_nonzero(codeword))


def _verify_embedding_s(s: np.ndarray) -> dict[str, Any]:
    g = _bch_generator_matrix()
    g_ext = np.hstack([g, s]) % 3
    gram = (g @ g.T) % 3
    gram_ext = (g_ext @ g_ext.T) % 3
    weight_distribution = _weight_distribution(g_ext, 3)
    positive_weights = [weight for weight in weight_distribution if weight > 0]
    return {
        "generator": "BCH(13,7,5), g(x)=x^6+2x^5+2x^4+2x^2+1 over GF(3)",
        "gram_rank": _rank_mod_p(gram, 3),
        "lower_bound_t": _rank_mod_p(gram, 3),
        "witness_t": int(s.shape[1]),
        "extended_length": int(g_ext.shape[1]),
        "self_orthogonal": bool(np.all(gram_ext == 0)),
        "minimum_distance": min(positive_weights) if positive_weights else None,
        "weight_distribution": weight_distribution,
        "matches_summary_weight_distribution": weight_distribution == {0: 1, 9: 166, 12: 972, 15: 954, 18: 94},
    }


def _verify_witness() -> dict[str, Any]:
    return _verify_embedding_s(_summary_witness_s())


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


def _encode_extension_coordinate_sum(
    cnf: _CnfBuilder,
    value_vars: dict[tuple[int, int, int], int],
    *,
    message: tuple[int, ...],
    col: int,
) -> dict[int, int]:
    state: dict[tuple[int, int], int] = {}
    k = len(message)
    for pos in range(k + 1):
        variables = []
        for residue in range(3):
            var = cnf.new_var()
            state[(pos, residue)] = var
            variables.append(var)
        cnf.exactly_one(variables)

    cnf.clauses.append([state[(0, 0)]])
    for row, coefficient in enumerate(message):
        for current in range(3):
            for value in range(3):
                next_residue = (current + coefficient * value) % 3
                cnf.clauses.append(
                    [
                        -state[(row, current)],
                        -value_vars[(row, col, value)],
                        state[(row + 1, next_residue)],
                    ]
                )
    return {residue: state[(k, residue)] for residue in range(3)}


def _require_at_least_nonzero_coordinates(
    cnf: _CnfBuilder,
    coordinate_residue_vars: list[dict[int, int]],
    *,
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
            clause.extend(
                [
                    coordinate_residue_vars[col][1],
                    coordinate_residue_vars[col][2],
                ]
            )
        cnf.clauses.append(clause)
    return len(cnf.clauses) - before


def _add_minimum_distance_constraints(
    cnf: _CnfBuilder,
    value_vars: dict[tuple[int, int, int], int],
    generator: np.ndarray,
    *,
    t: int,
    target: int,
) -> dict[str, Any]:
    k = int(generator.shape[0])
    representatives = _nonzero_message_representatives(k, 3)
    fixed_weight_distribution: dict[int, int] = {}
    encoded_by_fixed_weight: dict[int, int] = {}
    cardinality_clauses = 0
    impossible_classes = 0
    encoded_classes = 0
    skipped_classes = 0
    start_var_count = cnf.var_count
    start_clause_count = len(cnf.clauses)

    for message in representatives:
        fixed_weight = _message_fixed_weight(message, generator, 3)
        fixed_weight_distribution[fixed_weight] = fixed_weight_distribution.get(fixed_weight, 0) + 1
        required = target - fixed_weight
        if required <= 0:
            skipped_classes += 1
            continue
        encoded_classes += 1
        encoded_by_fixed_weight[fixed_weight] = encoded_by_fixed_weight.get(fixed_weight, 0) + 1
        if required > t:
            impossible_classes += 1
            cnf.clauses.append([])
            continue
        coordinate_residue_vars = [
            _encode_extension_coordinate_sum(
                cnf,
                value_vars,
                message=message,
                col=col,
            )
            for col in range(t)
        ]
        cardinality_clauses += _require_at_least_nonzero_coordinates(
            cnf,
            coordinate_residue_vars,
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


def _build_ternary_so_cnf(path: Path, *, t: int = 7, min_distance: int | None = None) -> dict[str, Any]:
    g = _bch_generator_matrix()
    k = g.shape[0]
    cnf = _CnfBuilder()

    x: dict[tuple[int, int, int], int] = {}
    for row in range(k):
        for col in range(t):
            variables = []
            for value in range(3):
                var = cnf.new_var()
                x[(row, col, value)] = var
                variables.append(var)
            cnf.exactly_one(variables)

    gram = (g @ g.T) % 3
    for i in range(k):
        for j in range(i, k):
            state: dict[tuple[int, int], int] = {}
            for pos in range(t + 1):
                variables = []
                for residue in range(3):
                    var = cnf.new_var()
                    state[(pos, residue)] = var
                    variables.append(var)
                cnf.exactly_one(variables)

            cnf.clauses.append([state[(0, int(gram[i, j]))]])

            for col in range(t):
                product_vars = []
                product: dict[int, int] = {}
                for residue in range(3):
                    var = cnf.new_var()
                    product[residue] = var
                    product_vars.append(var)
                cnf.exactly_one(product_vars)

                for a in range(3):
                    for b in range(3):
                        cnf.clauses.append([-x[(i, col, a)], -x[(j, col, b)], product[(a * b) % 3]])

                for current in range(3):
                    for prod in range(3):
                        cnf.clauses.append(
                            [
                                -state[(col, current)],
                                -product[prod],
                                state[(col + 1, (current + prod) % 3)],
                            ]
                        )

            cnf.clauses.append([state[(t, 0)]])

    distance_info: dict[str, Any] = {
        "minimum_distance_target": min_distance,
        "distance_constraints": "not_encoded" if min_distance is None else "encoded",
    }
    if min_distance is not None:
        if min_distance <= 0:
            raise ValueError("min_distance must be positive")
        distance_info.update(
            _add_minimum_distance_constraints(
                cnf,
                x,
                g,
                t=t,
                target=min_distance,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"p cnf {cnf.var_count} {len(cnf.clauses)}\n")
        for clause in cnf.clauses:
            handle.write(" ".join(str(literal) for literal in clause) + " 0\n")

    return {
        "cnf_path": str(path),
        "variables": cnf.var_count,
        "clauses": len(cnf.clauses),
        "rows": k,
        "added_columns": t,
        **distance_info,
        "_value_vars": {f"{row},{col},{value}": var for (row, col, value), var in x.items()},
    }


def _resolve_cadical(path: str | None) -> str | None:
    return resolve_cadical(path)


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


def _decode_s_from_model(value_vars: dict[str, int], true_vars: list[int], *, rows: int, cols: int) -> np.ndarray:
    true_set = set(true_vars)
    s = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        for col in range(cols):
            for value in range(3):
                if value_vars[f"{row},{col},{value}"] in true_set:
                    s[row, col] = value
                    break
    return s


def _drop_solver_model(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    compact.pop("model_true_vars", None)
    return compact


def _decode_sat_witness(
    *,
    solver_payload: dict[str, Any],
    value_vars: dict[str, int],
    cnf_info: dict[str, Any],
) -> dict[str, Any] | None:
    if solver_payload.get("status") != "SAT" or not solver_payload.get("model_true_vars"):
        return None
    generated_s = _decode_s_from_model(
        value_vars,
        solver_payload["model_true_vars"],
        rows=int(cnf_info["rows"]),
        cols=int(cnf_info["added_columns"]),
    )
    witness = _verify_embedding_s(generated_s)
    witness["s_matrix"] = generated_s.tolist()
    target = cnf_info.get("minimum_distance_target")
    if target is not None:
        witness["meets_minimum_distance_target"] = (
            witness["minimum_distance"] is not None and int(witness["minimum_distance"]) >= int(target)
        )
    return witness


@traceable_run("solevolve.ternary_bch_sat", run_type="tool")
def run_ternary_bch_sat(cnf_path: str, cadical_path: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    cnf = Path(cnf_path)
    solver = Path(cadical_path)
    cwd = str(Path.cwd())
    argv = [str(solver), str(cnf)]
    command = " ".join(argv)
    cnf_resolved = str(cnf.resolve()) if cnf.exists() else str(cnf)
    solver_resolved = str(solver.resolve()) if solver.exists() else str(solver)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
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
            "solver_status": payload.get("status"),
            "model_true_var_count": payload.get("model_true_var_count"),
        }
    )
    return payload


@traceable_run("solevolve.reproduce_ternary_bch", run_type="chain")
def reproduce_ternary_bch(
    *,
    artifact_dir: str | Path = "artifacts",
    cadical_path: str | None = None,
    timeout: int = 120,
    min_distance_target: int | None = 9,
    prove_optimality: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact_root = Path(artifact_dir) / "ternary_bch_repro"
    target_suffix = "so" if min_distance_target is None else f"d{min_distance_target}"
    cnf_info = _build_ternary_so_cnf(
        artifact_root / f"ternary_bch_t7_{target_suffix}.cnf",
        t=7,
        min_distance=min_distance_target,
    )
    value_vars = cnf_info.pop("_value_vars")
    summary_witness = _verify_witness()
    selected_solver = _resolve_cadical(cadical_path)
    if selected_solver:
        solver = run_ternary_bch_sat(cnf_info["cnf_path"], selected_solver, timeout)
    else:
        solver = {
            "status": "SKIPPED",
            "solver": "cadical",
            "solver_path": None,
            "reason": "No executable CaDiCaL binary found.",
        }
    sat_witness = _decode_sat_witness(
        solver_payload=solver,
        value_vars=value_vars,
        cnf_info=cnf_info,
    )
    solver = _drop_solver_model(solver)

    optimality_cnf_info = None
    optimality_solver = None
    optimality_witness = None
    if prove_optimality and min_distance_target is not None and selected_solver:
        optimality_target = min_distance_target + 1
        optimality_cnf_info = _build_ternary_so_cnf(
            artifact_root / f"ternary_bch_t7_d{optimality_target}.cnf",
            t=7,
            min_distance=optimality_target,
        )
        optimality_value_vars = optimality_cnf_info.pop("_value_vars")
        optimality_solver_payload = run_ternary_bch_sat(
            optimality_cnf_info["cnf_path"],
            selected_solver,
            timeout,
        )
        optimality_witness = _decode_sat_witness(
            solver_payload=optimality_solver_payload,
            value_vars=optimality_value_vars,
            cnf_info=optimality_cnf_info,
        )
        optimality_solver = _drop_solver_model(optimality_solver_payload)

    lower_bound_ok = summary_witness["gram_rank"] == 7
    target_ok = (
        sat_witness is not None
        and sat_witness["self_orthogonal"]
        and sat_witness["gram_rank"] == 7
        and sat_witness["witness_t"] == 7
        and (
            min_distance_target is None
            or (
                sat_witness["minimum_distance"] is not None
                and int(sat_witness["minimum_distance"]) >= min_distance_target
            )
        )
    )
    optimality_ok = (
        not prove_optimality
        or min_distance_target is None
        or (optimality_solver is not None and optimality_solver.get("status") == "UNSAT")
    )
    contradiction = (
        selected_solver is not None
        and (
            solver.get("status") == "UNSAT"
            or (optimality_solver is not None and optimality_solver.get("status") == "SAT")
        )
    )

    if lower_bound_ok and target_ok and optimality_ok:
        verdict = "PASS"
    elif contradiction:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"

    proof_obligations = [
        {
            "name": "shortest_self_orthogonal_extension_lower_bound",
            "status": "PASS" if lower_bound_ok else "FAIL",
            "method": "rank_GF3(GG^T)",
            "summary": "Any self-orthogonal extension needs at least rank(GG^T)=7 added coordinates.",
            "rank": summary_witness["gram_rank"],
        },
        {
            "name": "existence_self_orthogonal_min_distance_target",
            "status": "PASS" if target_ok else solver.get("status", "UNKNOWN"),
            "method": "SAT(CaDiCaL)",
            "target_minimum_distance": min_distance_target,
            "solver_status": solver.get("status"),
            "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
            "minimum_distance": sat_witness.get("minimum_distance") if sat_witness else None,
        },
    ]
    if min_distance_target is not None and prove_optimality:
        proof_obligations.append(
            {
                "name": "optimality_no_larger_minimum_distance",
                "status": "PASS" if optimality_ok else (optimality_solver or {}).get("status", "UNKNOWN"),
                "method": "SAT(CaDiCaL)",
                "target_minimum_distance": min_distance_target + 1,
                "solver_status": (optimality_solver or {}).get("status"),
                "solver_elapsed_ms": (optimality_solver or {}).get("solver_elapsed_ms"),
                "summary": f"UNSAT for d>={min_distance_target + 1} proves d={min_distance_target} is optimal at t=7.",
            }
        )

    report = {
        "result_id": "ternary_bch_13_7_5_gf3_shortest_so_embedding",
        "claim": (
            "BCH [13,7,5]_3 admits a shortest self-orthogonal embedding with t=7, "
            "n'=20, and target minimum distance d=9; d=9 is optimal for t=7 when "
            "the d>=10 SAT obligation is UNSAT."
        ),
        "scope_note": (
            "This command SAT-encodes self-orthogonality and, when requested, all "
            "nonzero GF(3)^7 message weight constraints for the target distance. "
            "The bundled summary witness is checked separately because raw paper "
            "matrix artifacts are not part of the public archive."
        ),
        "verdict": verdict,
        "cadical_preferred": True,
        "min_distance_target": min_distance_target,
        "prove_optimality": prove_optimality,
        "proof_obligations": proof_obligations,
        "cnf": cnf_info,
        "solver": solver,
        "sat_witness": sat_witness,
        "optimality_cnf": optimality_cnf_info,
        "optimality_solver": optimality_solver,
        "optimality_witness": optimality_witness,
        "summary_witness": summary_witness,
        "summary_witness_status": "OK" if summary_witness["self_orthogonal"] else "MISMATCH",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    report_path = artifact_root / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    update_current_run_context(
        metadata={
            "run_kind": "paper_result_repro",
            "result_id": report["result_id"],
            "verdict": verdict,
            "solver_actual_run": selected_solver is not None,
            "solver_backend": "cadical",
            "solver_path": solver.get("solver_path"),
            "solver_resolved_path": solver.get("solver_resolved_path"),
            "solver_command": solver.get("command"),
            "solver_cwd": solver.get("cwd"),
            "solver_returncode": solver.get("returncode"),
            "solver_status": solver.get("status"),
            "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
            "min_distance_target": min_distance_target,
            "minimum_distance": sat_witness.get("minimum_distance") if sat_witness else None,
            "optimality_target": min_distance_target + 1 if min_distance_target is not None and prove_optimality else None,
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
        tags=["paper-result", "ternary-bch", "cadical"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the ternary BCH self-orthogonal d=9 SAT result.")
    parser.add_argument("--artifact-dir", default=os.getenv("SOLEVOLVE_ARTIFACT_DIR", "artifacts"))
    parser.add_argument("--cadical-path", default=os.getenv("CADICAL_PATH"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--min-distance-target",
        type=int,
        default=9,
        help="Minimum distance target to SAT-encode; use 9 for the paper table d=9 claim.",
    )
    parser.add_argument(
        "--skip-optimality-check",
        action="store_true",
        help="Skip the d>=target+1 UNSAT check.",
    )
    args = parser.parse_args()
    try:
        report = reproduce_ternary_bch(
            artifact_dir=args.artifact_dir,
            cadical_path=args.cadical_path,
            timeout=args.timeout,
            min_distance_target=args.min_distance_target,
            prove_optimality=not args.skip_optimality_check,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        flush_tracing()


if __name__ == "__main__":
    main()
