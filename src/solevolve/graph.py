from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from .agents import EvolverAgent, GeneratorAgent, ReflectorAgent, VerifierAgent
from .contracts import (
    CodetablesLookupResult,
    CoordinatorDecision,
    EvidenceItem,
    EvolverAction,
    PaperTarget,
    PaperWorkflowInput,
    PaperWorkflowOutput,
    ReflectorAction,
    ReflectorMetrics,
    VerifierDiagnostics,
    default_coordinator_decision,
    fallback_coordinator_decision,
)
from .reviewer_claims import (
    codetables_query_for_claim,
    expand_reviewer_claim_ids,
    paper_target_for_claim,
    reproduce_reviewer_claims,
)
from .tracing import file_sha256, traceable_run, write_artifact_summary
from .tools.repro_tools import artifact_check, codetables_lookup, solver_check

ROOT_DIR = Path(__file__).resolve().parents[2]


def _paper_target_from_claim(claim_id: str) -> PaperTarget:
    return paper_target_for_claim(claim_id)


def _load_target_json(target_json: str) -> dict[str, Any]:
    try:
        candidate = Path(target_json)
        if len(target_json) < 4096 and candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    except OSError:
        pass
    return json.loads(target_json)


def _default_codetables_query(target: PaperTarget | None) -> dict[str, int] | None:
    if target is None:
        return None
    return {"q": target.q, "n": target.n, "k": target.k}


def _build_paper_workflow_input_impl(
    *,
    goal: str,
    paper_claim_id: str | None = None,
    target_json: str | None = None,
    solver_preference: str = "cadical",
    solver_budget_sec: int = 300,
) -> PaperWorkflowInput:
    """Resolve CLI/free-form inputs into the paper-facing workflow input contract."""
    if target_json:
        payload = _load_target_json(target_json)
        if "target" in payload or "targets" in payload or "human_goal" in payload:
            merged = {
                "human_goal": goal,
                "solver_budget_sec": solver_budget_sec,
                **payload,
            }
            workflow = PaperWorkflowInput.model_validate(merged)
            if workflow.target is None and len(workflow.targets) == 1:
                workflow = workflow.model_copy(update={"target": workflow.targets[0]})
            return workflow
        target = PaperTarget.model_validate(payload)
        targets = [target]
        return PaperWorkflowInput(
            human_goal=goal,
            target=target,
            targets=targets,
            codetables_query=_default_codetables_query(target),
            solver_budget_sec=solver_budget_sec,
        )

    claim_ids = expand_reviewer_claim_ids(paper_claim_id) if paper_claim_id else []
    targets = [_paper_target_from_claim(claim) for claim in claim_ids]
    primary = targets[0] if len(targets) == 1 else None
    if not targets:
        primary = PaperTarget(
            claim_id=None,
            q=2,
            n=22,
            k=11,
            d=7,
            field="GF(2)",
            inner_product="dot",
            objective="Outline the paper SolEvolve SAT/GA workflow without executing a registered claim.",
            required_checks=["artifact_manifest", "solver_check"],
            solver_preference=solver_preference,
            source_n=None,
            t=None,
        )
        targets = [primary]
    return PaperWorkflowInput(
        human_goal=goal,
        target=primary,
        targets=targets,
        codetables_query=_default_codetables_query(primary),
        solver_budget_sec=solver_budget_sec,
        population_state=None,
        verifier_state=None,
    )


@traceable_run("solevolve.paper_input", run_type="chain")
def build_paper_workflow_input(
    *,
    goal: str,
    paper_claim_id: str | None = None,
    target_json: str | None = None,
    solver_preference: str = "cadical",
    solver_budget_sec: int = 300,
) -> PaperWorkflowInput:
    return _build_paper_workflow_input_impl(
        goal=goal,
        paper_claim_id=paper_claim_id,
        target_json=target_json,
        solver_preference=solver_preference,
        solver_budget_sec=solver_budget_sec,
    )


class AgentState(TypedDict, total=False):
    goal: str
    messages: List[BaseMessage]
    turn: int
    evidence: list[EvidenceItem]
    paper_input: PaperWorkflowInput
    verifier_diagnostics: list[VerifierDiagnostics]
    reflector_action: ReflectorAction
    paper_output: PaperWorkflowOutput
    next_instruction: str | None
    coordinator_decision: CoordinatorDecision
    artifact_path: str
    run_summary: dict[str, Any]


def _repo_manifest_path(artifact_dir: str | Path | None) -> Path:
    if artifact_dir is not None:
        candidate = Path(artifact_dir) / "manifest.json"
        if candidate.exists():
            return candidate
    return ROOT_DIR / "artifacts" / "manifest.json"


def _ternary_bch_report_path(artifact_dir: str | Path | None) -> Path:
    base = Path(artifact_dir) if artifact_dir is not None else ROOT_DIR / "artifacts"
    return base / "ternary_bch_repro" / "reproduction_report.json"


def _loads_tool_json(payload: str) -> dict[str, Any]:
    try:
        loaded = json.loads(payload)
    except Exception as exc:
        return {"status": "ERROR", "error": f"invalid tool JSON: {exc}", "raw": payload}
    return loaded if isinstance(loaded, dict) else {"status": "ERROR", "error": "tool JSON was not an object"}


def _safe_file_hash(path: Any) -> str | None:
    if not path:
        return None
    try:
        candidate = Path(str(path))
        if candidate.is_file():
            return file_sha256(candidate)
    except Exception:
        return None
    return None


def _evidence_from_payload(
    *,
    name: str,
    command_or_function: str,
    payload: dict[str, Any],
    artifacts: list[str],
) -> EvidenceItem:
    status = str(payload.get("status") or "UNKNOWN")
    error = payload.get("error") or payload.get("reason")
    selected_path = payload.get("selected_resolved_path") or payload.get("selected_path") or payload.get("path")
    summary_bits = [
        f"status={status}",
        f"preferred_solver={payload.get('preferred_solver')}" if payload.get("preferred_solver") else "",
        f"selected_solver={payload.get('selected_solver') or payload.get('solver')}" if (payload.get("selected_solver") or payload.get("solver")) else "",
        f"selected_path={selected_path}" if selected_path else "",
        f"cwd={payload.get('cwd')}" if payload.get("cwd") else "",
        f"command={payload.get('command')}" if payload.get("command") else "",
        f"elapsed_ms={payload.get('elapsed_ms')}" if payload.get("elapsed_ms") is not None else "",
        f"version_elapsed_ms={payload.get('version_elapsed_ms')}" if payload.get("version_elapsed_ms") is not None else "",
        f"manifest={payload.get('manifest')}" if payload.get("manifest") else "",
        f"bundled_count={payload.get('bundled_count')}" if payload.get("bundled_count") is not None else "",
        f"fetch_status={payload.get('fetch_status')}" if payload.get("fetch_status") else "",
        f"bounds={payload.get('lower_bound')}..{payload.get('upper_bound')}"
        if payload.get("lower_bound") is not None or payload.get("upper_bound") is not None
        else "",
        f"construction_d={payload.get('construction_d')}" if payload.get("construction_d") is not None else "",
    ]
    details = {
        key: payload[key]
        for key in (
            "argv",
            "bundled_count",
            "checked",
            "command",
            "cwd",
            "elapsed_ms",
            "manifest",
            "preferred_solver",
            "returncode",
            "runtime_output_dir",
            "selected_path",
            "selected_resolved_path",
            "selected_solver",
            "selected_source",
            "solver",
            "status",
            "version",
            "version_elapsed_ms",
            "q",
            "n",
            "k",
            "url",
            "fetch_status",
            "lower_bound",
            "upper_bound",
            "construction_d",
            "content_sha256",
            "fetched_at",
            "cache_path",
        )
        if key in payload
    }
    try:
        return EvidenceItem(
            name=name,
            status=status,
            command_or_function=command_or_function,
            summary="; ".join(bit for bit in summary_bits if bit),
            artifacts=artifacts,
            error=str(error) if error else None,
            details=details,
        )
    except ValidationError:
        return EvidenceItem(
            name=name,
            status="UNKNOWN",
            command_or_function=command_or_function,
            summary=f"status={status}",
            artifacts=artifacts,
            error=str(error) if error else f"unexpected evidence status: {status}",
            details=details,
        )


def _ternary_bch_report_evidence(artifact_dir: str | Path | None) -> EvidenceItem | None:
    report_path = _ternary_bch_report_path(artifact_dir)
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return EvidenceItem(
            name="ternary_bch_d9_report",
            status="ERROR",
            command_or_function=f"read_json({report_path})",
            summary="Failed to read ternary BCH d=9 reproduction report.",
            artifacts=[str(report_path)],
            error=str(exc),
        )

    return _ternary_bch_payload_evidence(payload, report_path=report_path)


def _ternary_bch_payload_evidence(payload: dict[str, Any], *, report_path: Path) -> EvidenceItem:
    verdict = str(payload.get("verdict") or "UNKNOWN")
    status = "OK" if verdict == "PASS" else "ERROR" if verdict == "FAIL" else "UNKNOWN"
    cnf = payload.get("cnf") if isinstance(payload.get("cnf"), dict) else {}
    solver = payload.get("solver") if isinstance(payload.get("solver"), dict) else {}
    optimality_cnf = payload.get("optimality_cnf") if isinstance(payload.get("optimality_cnf"), dict) else {}
    optimality_solver = payload.get("optimality_solver") if isinstance(payload.get("optimality_solver"), dict) else {}
    witness = payload.get("sat_witness") if isinstance(payload.get("sat_witness"), dict) else {}
    proof_obligations = payload.get("proof_obligations") if isinstance(payload.get("proof_obligations"), list) else []
    details = {
        "verdict": verdict,
        "result_id": payload.get("result_id"),
        "min_distance_target": payload.get("min_distance_target"),
        "proof_obligations": proof_obligations,
        "cnf_path": cnf.get("cnf_path"),
        "cnf_sha256": _safe_file_hash(cnf.get("cnf_path")),
        "cnf_variables": cnf.get("variables"),
        "cnf_clauses": cnf.get("clauses"),
        "distance_encoded_classes": cnf.get("distance_encoded_classes"),
        "solver_status": solver.get("status"),
        "solver_command": solver.get("command"),
        "solver_path": solver.get("solver_path"),
        "solver_resolved_path": solver.get("solver_resolved_path"),
        "solver_argv": solver.get("argv"),
        "solver_cwd": solver.get("cwd"),
        "solver_returncode": solver.get("returncode"),
        "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
        "solver_timeout_sec": solver.get("timeout"),
        "solver_memory_mb": solver.get("memory_mb"),
        "optimality_cnf_path": optimality_cnf.get("cnf_path"),
        "optimality_cnf_sha256": _safe_file_hash(optimality_cnf.get("cnf_path")),
        "optimality_cnf_variables": optimality_cnf.get("variables"),
        "optimality_cnf_clauses": optimality_cnf.get("clauses"),
        "optimality_solver_status": optimality_solver.get("status"),
        "optimality_solver_command": optimality_solver.get("command"),
        "optimality_solver_path": optimality_solver.get("solver_path"),
        "optimality_solver_resolved_path": optimality_solver.get("solver_resolved_path"),
        "optimality_solver_argv": optimality_solver.get("argv"),
        "optimality_solver_cwd": optimality_solver.get("cwd"),
        "optimality_solver_returncode": optimality_solver.get("returncode"),
        "optimality_solver_elapsed_ms": optimality_solver.get("solver_elapsed_ms"),
        "optimality_solver_timeout_sec": optimality_solver.get("timeout"),
        "optimality_solver_memory_mb": optimality_solver.get("memory_mb"),
        "minimum_distance": witness.get("minimum_distance"),
        "matches_summary_weight_distribution": witness.get("matches_summary_weight_distribution"),
        "weight_distribution": witness.get("weight_distribution"),
    }
    summary_bits = [
        f"verdict={verdict}",
        f"d_target={payload.get('min_distance_target')}",
        f"d9_solver={solver.get('status')} in {solver.get('solver_elapsed_ms')}ms",
        f"d10_solver={optimality_solver.get('status')} in {optimality_solver.get('solver_elapsed_ms')}ms"
        if optimality_solver
        else "",
        f"minimum_distance={witness.get('minimum_distance')}" if witness else "",
        "weight_distribution_matches_table=true"
        if witness.get("matches_summary_weight_distribution") is True
        else "",
    ]
    return EvidenceItem(
        name="ternary_bch_d9_report",
        status=status,
        command_or_function=f"read_json({report_path})",
        summary="; ".join(bit for bit in summary_bits if bit),
        artifacts=[
            str(path)
            for path in (
                report_path,
                cnf.get("cnf_path"),
                optimality_cnf.get("cnf_path"),
            )
            if path
        ],
        error=None if status == "OK" else payload.get("scope_note"),
        details={key: value for key, value in details.items() if value is not None},
    )


def _paper_claim_payload_evidence(payload: dict[str, Any], *, report_path: Path | None = None) -> EvidenceItem:
    verdict = str(payload.get("verdict") or "UNKNOWN")
    claim_id = str(payload.get("result_id") or "unknown")
    status = "OK" if verdict == "PASS" else "ERROR" if verdict == "FAIL" else "UNAVAILABLE" if verdict == "INSUFFICIENT_ARTIFACT" else "UNKNOWN"
    family = payload.get("claim_family") or "self_orthogonal"
    matrix = payload.get("matrix_verification") if isinstance(payload.get("matrix_verification"), dict) else {}
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    checks = diagnostics.get("checks") if isinstance(diagnostics.get("checks"), dict) else {}
    binary_solutions = payload.get("solution_reports") if isinstance(payload.get("solution_reports"), list) else []
    first_binary = binary_solutions[0].get("diagnostics", {}) if binary_solutions else {}
    cnf = payload.get("cnf") if isinstance(payload.get("cnf"), dict) else {}
    solver = payload.get("solver") if isinstance(payload.get("solver"), dict) else {}
    optimality_cnf = payload.get("optimality_cnf") if isinstance(payload.get("optimality_cnf"), dict) else {}
    optimality_solver = payload.get("optimality_solver") if isinstance(payload.get("optimality_solver"), dict) else {}
    proof_obligations = payload.get("proof_obligations") if isinstance(payload.get("proof_obligations"), list) else []
    missing_obligations = payload.get("missing_obligations") if isinstance(payload.get("missing_obligations"), list) else []
    source_artifacts = payload.get("source_artifacts") if isinstance(payload.get("source_artifacts"), list) else []
    first_source_artifact = source_artifacts[0] if source_artifacts and isinstance(source_artifacts[0], dict) else {}
    binary_artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
    details = {
        "result_id": claim_id,
        "claim_family": family,
        "verdict": verdict,
        "field_order": payload.get("field_order"),
        "inner_product": payload.get("inner_product"),
        "parameters": payload.get("parameters"),
        "source_parameters": payload.get("source_parameters"),
        "extended_parameters": payload.get("extended_parameters"),
        "target_minimum_distance": payload.get("target_minimum_distance"),
        "grassl_bound": payload.get("grassl_bound"),
        "missing_obligations": missing_obligations,
        "optimality_required": payload.get("optimality_required"),
        "proof_obligations": proof_obligations,
        "matrix_minimum_distance": matrix.get("minimum_distance"),
        "matrix_self_orthogonal": matrix.get("self_orthogonal"),
        "matrix_weight_distribution": matrix.get("weight_distribution"),
        "matrix_ok": matrix.get("matrix_ok"),
        "gram_rank": matrix.get("gram_rank"),
        "lucas_vertex_count": diagnostics.get("vertex_count"),
        "lucas_expected_vertex_count": diagnostics.get("expected_vertex_count") or payload.get("expected_vertex_count"),
        "lucas_center_count": diagnostics.get("center_count"),
        "lucas_expected_center_count": diagnostics.get("expected_center_count") or payload.get("expected_center_count"),
        "lucas_exact_cover": checks.get("exact_cover"),
        "lucas_minimum_pairwise_distance": diagnostics.get("minimum_pairwise_distance"),
        "lucas_ball_size_distribution": diagnostics.get("ball_size_distribution"),
        "lucas_expected_ball_size_distribution": diagnostics.get("expected_ball_size_distribution") or payload.get("expected_ball_size_distribution"),
        "binary_minimum_distance": first_binary.get("minimum_distance"),
        "binary_weight_distribution": first_binary.get("weight_distribution"),
        "binary_A_d_progression": payload.get("A_d_progression"),
        "binary_best_A_d": payload.get("best_A_d"),
        "binary_bklc_A_d": payload.get("bklc_A_d"),
        "binary_reduction_percent": payload.get("reduction_percent"),
        "cnf_path": cnf.get("cnf_path"),
        "cnf_resolved_path": cnf.get("cnf_resolved_path"),
        "cnf_sha256": _safe_file_hash(cnf.get("cnf_path")),
        "cnf_variables": cnf.get("variables"),
        "cnf_clauses": cnf.get("clauses"),
        "solver_status": solver.get("status"),
        "solver_path": solver.get("solver_path"),
        "solver_resolved_path": solver.get("solver_resolved_path"),
        "solver_command": solver.get("command"),
        "solver_argv": solver.get("argv"),
        "solver_cwd": solver.get("cwd"),
        "solver_returncode": solver.get("returncode"),
        "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
        "solver_timeout_sec": solver.get("timeout"),
        "solver_memory_mb": solver.get("memory_mb"),
        "optimality_cnf_path": optimality_cnf.get("cnf_path"),
        "optimality_cnf_resolved_path": optimality_cnf.get("cnf_resolved_path"),
        "optimality_cnf_sha256": _safe_file_hash(optimality_cnf.get("cnf_path")),
        "optimality_cnf_variables": optimality_cnf.get("variables"),
        "optimality_cnf_clauses": optimality_cnf.get("clauses"),
        "optimality_solver_status": optimality_solver.get("status"),
        "optimality_solver_path": optimality_solver.get("solver_path"),
        "optimality_solver_resolved_path": optimality_solver.get("solver_resolved_path"),
        "optimality_solver_command": optimality_solver.get("command"),
        "optimality_solver_argv": optimality_solver.get("argv"),
        "optimality_solver_cwd": optimality_solver.get("cwd"),
        "optimality_solver_returncode": optimality_solver.get("returncode"),
        "optimality_solver_elapsed_ms": optimality_solver.get("solver_elapsed_ms"),
        "optimality_solver_timeout_sec": optimality_solver.get("timeout"),
        "optimality_solver_memory_mb": optimality_solver.get("memory_mb"),
        "report_path": str(report_path) if report_path else payload.get("report_path"),
        "artifact_path": payload.get("artifact_path") or binary_artifact.get("path") or first_source_artifact.get("path"),
        "artifact_sha256": binary_artifact.get("sha256") or first_source_artifact.get("sha256"),
        "elapsed_ms": payload.get("elapsed_ms"),
    }
    if family == "lucas_cubes":
        params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        summary_bits = [
            f"verdict={verdict}",
            "family=lucas_cubes",
            f"n={params.get('n')}" if params.get("n") is not None else "",
            f"s={params.get('s')} (forbidden_run_length)" if params.get("s") is not None else "",
            f"expected_centers={details.get('lucas_expected_center_count')}"
            if details.get("lucas_expected_center_count") is not None
            else "",
            f"vertices={details.get('lucas_vertex_count')}/{details.get('lucas_expected_vertex_count')}"
            if details.get("lucas_vertex_count") is not None or details.get("lucas_expected_vertex_count") is not None
            else "",
            f"centers={details.get('lucas_center_count')}/{details.get('lucas_expected_center_count')}"
            if details.get("lucas_center_count") is not None or details.get("lucas_expected_center_count") is not None
            else "",
            f"exact_cover={checks.get('exact_cover')}" if checks.get("exact_cover") is not None else "",
            f"min_pairwise_distance={details.get('lucas_minimum_pairwise_distance')}"
            if details.get("lucas_minimum_pairwise_distance") is not None
            else "",
            f"missing={','.join(missing_obligations)}" if missing_obligations else "",
            f"elapsed_ms={payload.get('elapsed_ms')}" if payload.get("elapsed_ms") is not None else "",
        ]
    else:
        summary_bits = [
            f"verdict={verdict}",
            f"family={family}",
            f"field=GF({payload.get('field_order')})" if payload.get("field_order") else "",
            f"target_d={payload.get('target_minimum_distance')}",
            f"matrix_d={matrix.get('minimum_distance')}",
            f"A_d={payload.get('A_d_progression')}" if payload.get("A_d_progression") else "",
            f"best_A_d={payload.get('best_A_d')}" if payload.get("best_A_d") is not None else "",
            f"matrix_ok={matrix.get('matrix_ok')}",
            f"solver={solver.get('status')} in {solver.get('solver_elapsed_ms')}ms" if solver.get("status") else "",
            f"d+1={optimality_solver.get('status')} in {optimality_solver.get('solver_elapsed_ms')}ms"
            if optimality_solver.get("status")
            else "",
        ]
    return EvidenceItem(
        name=f"paper_claim_{claim_id}",
        status=status,
        command_or_function=f"reproduce_paper_claim({claim_id})",
        summary="; ".join(bit for bit in summary_bits if bit),
        artifacts=[
            str(path)
            for path in (
                [report_path or payload.get("report_path"), cnf.get("cnf_path"), optimality_cnf.get("cnf_path"), payload.get("artifact_path"), binary_artifact.get("path")]
                + [item.get("path") for item in source_artifacts if isinstance(item, dict)]
            )
            if path
        ],
        error=None if status == "OK" else f"claim verdict={verdict}; missing={','.join(missing_obligations)}",
        details={key: value for key, value in details.items() if value is not None},
    )


@traceable_run("solevolve.paper_claims", run_type="chain")
def _run_paper_claim_evidence(
    *,
    paper_claim_id: str | None,
    artifact_dir: str | Path | None,
    cadical_path: str | None,
    timeout: int,
) -> EvidenceItem | None:
    if not paper_claim_id:
        return _ternary_bch_report_evidence(artifact_dir)
    try:
        expanded = expand_reviewer_claim_ids(paper_claim_id)
    except KeyError:
        return EvidenceItem(
            name="paper_claim_reproduction",
            status="UNAVAILABLE",
            command_or_function=f"paper_claim({paper_claim_id})",
            summary=f"No deterministic paper-claim runner is registered for {paper_claim_id}.",
            artifacts=[],
            error="Unsupported SOLEVOLVE_PAPER_CLAIM_ID.",
        )
    if not expanded:
        return None
    summary = reproduce_reviewer_claims(
        claim_id=paper_claim_id,
        artifact_dir=artifact_dir or ROOT_DIR / "artifacts",
        cadical_path=cadical_path,
        timeout=timeout,
        prove_optimality=True,
    )
    return _paper_claim_payload_evidence(
        {
            **summary,
            "result_id": paper_claim_id.strip().lower(),
            "field_order": "mixed",
            "target_minimum_distance": None,
        },
        report_path=Path(str(summary.get("report_path") or Path(artifact_dir or ROOT_DIR / "artifacts") / "paper_claims")),
    )


@traceable_run("solevolve.paper_claims", run_type="chain")
def _run_paper_claim_evidence_items(
    *,
    paper_claim_id: str,
    artifact_dir: str | Path | None,
    cadical_path: str | None,
    timeout: int,
) -> list[EvidenceItem]:
    try:
        expanded = expand_reviewer_claim_ids(paper_claim_id)
    except KeyError:
        return [
            EvidenceItem(
                name="paper_claim_reproduction",
                status="UNAVAILABLE",
                command_or_function=f"paper_claim({paper_claim_id})",
                summary=f"No deterministic paper-claim runner is registered for {paper_claim_id}.",
                artifacts=[],
                error="Unsupported SOLEVOLVE_PAPER_CLAIM_ID.",
            )
        ]
    summary = reproduce_reviewer_claims(
        claim_id=paper_claim_id,
        artifact_dir=artifact_dir or ROOT_DIR / "artifacts",
        cadical_path=cadical_path,
        timeout=timeout,
        prove_optimality=True,
    )
    reports = summary.get("reports") if isinstance(summary.get("reports"), list) else []
    items = [
        _paper_claim_payload_evidence(
            report,
            report_path=Path(str(report.get("report_path"))) if report.get("report_path") else None,
        )
        for report in reports
    ]
    if len(items) > 1:
        items.append(
            EvidenceItem(
                name=f"paper_claim_{paper_claim_id.strip().lower()}",
                status="OK" if summary.get("verdict") == "PASS" else "ERROR" if summary.get("verdict") == "FAIL" else "UNKNOWN",
                command_or_function=f"reproduce_paper_claims({paper_claim_id})",
                summary=f"verdict={summary.get('verdict')}; claims={','.join(expanded)}",
                artifacts=[str(summary.get("report_path"))] if summary.get("report_path") else [],
                error=None if summary.get("verdict") == "PASS" else f"aggregate verdict={summary.get('verdict')}",
                details={
                    "result_id": paper_claim_id.strip().lower(),
                    "verdict": summary.get("verdict"),
                    "claim_ids": expanded,
                    "elapsed_ms": summary.get("elapsed_ms"),
                },
            )
        )
    return items


@traceable_run("solevolve.bounds_lookup", run_type="tool")
def _run_codetables_evidence_items(
    *,
    paper_claim_id: str,
    artifact_dir: str | Path | None,
) -> list[EvidenceItem]:
    try:
        expanded = expand_reviewer_claim_ids(paper_claim_id)
    except KeyError:
        return []
    items: list[EvidenceItem] = []
    for claim in expanded:
        query = codetables_query_for_claim(claim)
        if query is None:
            continue
        payload = _loads_tool_json(
            codetables_lookup.func(
                q=query["q"],
                n=query["n"],
                k=query["k"],
                artifact_dir=str(artifact_dir or ROOT_DIR / "artifacts"),
                timeout=15,
                use_cache=True,
            )
        )
        status = "OK" if payload.get("fetch_status") in {"LIVE", "CACHE", "STALE"} else "UNAVAILABLE"
        items.append(
            _evidence_from_payload(
                name=f"codetables_{claim}",
                command_or_function=f"codetables_lookup(q={query['q']}, n={query['n']}, k={query['k']})",
                payload={**payload, "status": status},
                artifacts=[str(payload.get("cache_path"))] if payload.get("cache_path") else [],
            )
        )
    return items


def run_repro_checks(
    *,
    artifact_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    solver_preference: str = "cadical",
    paper_claim_id: str | None = None,
    cadical_path: str | None = None,
    paper_claim_timeout: int = 1200,
) -> list[EvidenceItem]:
    manifest_path = _repo_manifest_path(artifact_dir)
    root = Path(repo_root) if repo_root is not None else ROOT_DIR

    artifact_payload = _loads_tool_json(
        artifact_check.func(manifest_path=str(manifest_path))
    )
    solver_payload = _loads_tool_json(solver_check.func(root=str(root), preferred_solver=solver_preference))

    evidence = [
        _evidence_from_payload(
            name="artifact_manifest",
            command_or_function=f"artifact_check({manifest_path})",
            payload=artifact_payload,
            artifacts=[str(manifest_path)],
        ),
        _evidence_from_payload(
            name="solver_check",
            command_or_function=f"solver_check({root}, preferred_solver={solver_preference})",
            payload=solver_payload,
            artifacts=[str(item.get("path")) for item in solver_payload.get("checked", []) if item.get("path")],
        ),
    ]
    if paper_claim_id:
        evidence.extend(
            _run_paper_claim_evidence_items(
                paper_claim_id=paper_claim_id,
                artifact_dir=artifact_dir,
                cadical_path=cadical_path,
                timeout=paper_claim_timeout,
            )
        )
    else:
        report_evidence = _run_paper_claim_evidence(
            paper_claim_id=paper_claim_id,
            artifact_dir=artifact_dir,
            cadical_path=cadical_path,
            timeout=paper_claim_timeout,
        )
        if report_evidence is not None:
            evidence.append(report_evidence)
    return evidence


def evidence_status_summary(evidence: list[EvidenceItem]) -> dict[str, str]:
    return {item.name: item.status for item in _coerce_evidence_items(evidence)}


def _message_trace_metadata(message: BaseMessage) -> dict[str, Any] | None:
    response_metadata = getattr(message, "response_metadata", None) or {}
    trace_metadata = response_metadata.get("solevolve_trace")
    return trace_metadata if isinstance(trace_metadata, dict) else None


def _llm_run_summaries(state: AgentState) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for message in state.get("messages", []):
        trace_metadata = _message_trace_metadata(message)
        if not trace_metadata:
            continue
        cost = trace_metadata.get("llm_cost") if isinstance(trace_metadata.get("llm_cost"), dict) else {}
        usage = trace_metadata.get("llm_usage") if isinstance(trace_metadata.get("llm_usage"), dict) else {}
        summaries.append(
            {
                "agent": trace_metadata.get("agent") or getattr(message, "name", None),
                "model": trace_metadata.get("llm_model_name"),
                "elapsed_ms": trace_metadata.get("llm_elapsed_ms"),
                "finish_reason": trace_metadata.get("llm_finish_reason"),
                "usage": usage,
                "cost": cost,
            }
        )
    return summaries


def _llm_totals(llm_runs: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost = 0.0
    cost_available = False
    cost_sources: set[str] = set()
    for run in llm_runs:
        cost = run.get("cost") if isinstance(run.get("cost"), dict) else {}
        prompt_tokens += int(cost.get("prompt_tokens") or run.get("usage", {}).get("input_tokens") or 0)
        completion_tokens += int(cost.get("completion_tokens") or run.get("usage", {}).get("output_tokens") or 0)
        total_tokens += int(cost.get("total_tokens") or run.get("usage", {}).get("total_tokens") or 0)
        if cost.get("estimated_cost_usd") is not None:
            estimated_cost += float(cost["estimated_cost_usd"])
            cost_available = True
        if cost.get("pricing_source"):
            cost_sources.add(str(cost["pricing_source"]))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost_usd": round(estimated_cost, 8) if cost_available else None,
        "cost_available": cost_available,
        "currency": "USD",
        "pricing_sources": sorted(cost_sources),
    }


def _solver_execution_summary(evidence: list[EvidenceItem]) -> dict[str, Any] | None:
    for item in evidence:
        if item.name != "solver_check":
            continue
        details = item.details or {}
        return {
            "phase": "solver_check",
            "note": "Availability/version check only; this is not a BCH CNF solve.",
            "status": item.status,
            "preferred_solver": details.get("preferred_solver"),
            "selected_solver": details.get("selected_solver") or details.get("solver"),
            "selected_path": details.get("selected_resolved_path") or details.get("selected_path"),
            "selected_source": details.get("selected_source"),
            "cwd": details.get("cwd"),
            "argv": details.get("argv"),
            "command": details.get("command"),
            "version": details.get("version"),
            "returncode": details.get("returncode"),
            "elapsed_ms": details.get("elapsed_ms"),
            "version_elapsed_ms": details.get("version_elapsed_ms"),
            "checked": details.get("checked", []),
        }
    return None


def _paper_claim_execution_summary(evidence: list[EvidenceItem]) -> dict[str, Any] | None:
    results = []
    for item in evidence:
        if not item.name.startswith("paper_claim_"):
            continue
        details = item.details or {}
        if not details.get("result_id"):
            continue
        results.append(
            {
                "phase": "paper_claim_reproduction",
                "result_id": details.get("result_id"),
                "claim_family": details.get("claim_family"),
                "verdict": details.get("verdict"),
                "field_order": details.get("field_order"),
                "inner_product": details.get("inner_product"),
                "parameters": details.get("parameters"),
                "missing_obligations": details.get("missing_obligations", []),
                "source_parameters": details.get("source_parameters"),
                "extended_parameters": details.get("extended_parameters"),
                "target_minimum_distance": details.get("target_minimum_distance"),
                "minimum_distance": details.get("matrix_minimum_distance") or details.get("binary_minimum_distance") or details.get("minimum_distance"),
                "matrix_ok": details.get("matrix_ok"),
                "weight_distribution": details.get("matrix_weight_distribution") or details.get("binary_weight_distribution") or details.get("weight_distribution"),
                "lucas_vertex_count": details.get("lucas_vertex_count"),
                "lucas_expected_vertex_count": details.get("lucas_expected_vertex_count"),
                "lucas_center_count": details.get("lucas_center_count"),
                "lucas_expected_center_count": details.get("lucas_expected_center_count"),
                "lucas_exact_cover": details.get("lucas_exact_cover"),
                "lucas_minimum_pairwise_distance": details.get("lucas_minimum_pairwise_distance"),
                "lucas_ball_size_distribution": details.get("lucas_ball_size_distribution"),
                "lucas_expected_ball_size_distribution": details.get("lucas_expected_ball_size_distribution"),
                "binary_A_d_progression": details.get("binary_A_d_progression"),
                "binary_best_A_d": details.get("binary_best_A_d"),
                "binary_bklc_A_d": details.get("binary_bklc_A_d"),
                "binary_reduction_percent": details.get("binary_reduction_percent"),
                "grassl_bound": details.get("grassl_bound"),
                "cnf_path": details.get("cnf_path"),
                "cnf_variables": details.get("cnf_variables"),
                "cnf_clauses": details.get("cnf_clauses"),
                "solver_status": details.get("solver_status"),
                "solver_command": details.get("solver_command"),
                "solver_elapsed_ms": details.get("solver_elapsed_ms"),
                "optimality_required": details.get("optimality_required"),
                "optimality_cnf_path": details.get("optimality_cnf_path"),
                "optimality_cnf_variables": details.get("optimality_cnf_variables"),
                "optimality_cnf_clauses": details.get("optimality_cnf_clauses"),
                "optimality_solver_status": details.get("optimality_solver_status"),
                "optimality_solver_command": details.get("optimality_solver_command"),
                "optimality_solver_elapsed_ms": details.get("optimality_solver_elapsed_ms"),
                "proof_obligations": details.get("proof_obligations", []),
                "report_path": details.get("report_path"),
            }
        )
    if not results:
        return None
    aggregate_verdict = "PASS" if all(result.get("verdict") == "PASS" for result in results) else "FAIL" if any(result.get("verdict") == "FAIL" for result in results) else "PARTIAL"
    return {
        "phase": "paper_claim_reproduction",
        "verdict": aggregate_verdict,
        "claim_results": results,
    }


def _codetables_refs_summary(evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    refs = []
    for item in evidence:
        if not item.name.startswith("codetables_"):
            continue
        details = item.details or {}
        refs.append(
            {
                "name": item.name,
                "status": item.status,
                "q": details.get("q"),
                "n": details.get("n"),
                "k": details.get("k"),
                "url": details.get("url"),
                "fetch_status": details.get("fetch_status"),
                "lower_bound": details.get("lower_bound"),
                "upper_bound": details.get("upper_bound"),
                "construction_d": details.get("construction_d"),
                "content_sha256": details.get("content_sha256"),
                "cache_path": details.get("cache_path"),
            }
        )
    return refs


def _proof_obligation_satisfied(obligation: dict[str, Any]) -> bool:
    status = str(obligation.get("status") or "")
    if status == "PASS":
        return True
    # Backward compatibility for reports produced before SAT feasibility was
    # normalized to PASS. A target feasibility SAT result satisfies the
    # existence obligation; optional stronger-distance checks remain advisory.
    return obligation.get("name") == "sat_feasibility_minimum_distance_target" and status == "SAT"


def _missing_obligations(evidence: list[EvidenceItem]) -> list[str]:
    missing: list[str] = []
    for item in evidence:
        if not item.name.startswith("paper_claim_"):
            continue
        details = item.details or {}
        for obligation in details.get("missing_obligations", []) or []:
            missing.append(f"{item.name}:{obligation}")
        for obligation in details.get("proof_obligations", []) or []:
            if obligation.get("required") is False:
                continue
            if not _proof_obligation_satisfied(obligation):
                missing.append(f"{item.name}:{obligation.get('name')}={obligation.get('status')}")
    return missing


def _codetables_results(evidence: list[EvidenceItem]) -> list[CodetablesLookupResult]:
    results: list[CodetablesLookupResult] = []
    for item in evidence:
        if not item.name.startswith("codetables_"):
            continue
        details = item.details or {}
        try:
            results.append(
                CodetablesLookupResult(
                    q=int(details["q"]),
                    n=int(details["n"]),
                    k=int(details["k"]),
                    url=str(details["url"]),
                    fetch_status=details.get("fetch_status") or "UNAVAILABLE",
                    lower_bound=details.get("lower_bound"),
                    upper_bound=details.get("upper_bound"),
                    construction_d=details.get("construction_d"),
                    content_sha256=details.get("content_sha256"),
                    fetched_at=details.get("fetched_at"),
                    cache_path=details.get("cache_path"),
                    error=item.error,
                )
            )
        except Exception:
            continue
    return results


def _solver_runs_from_evidence(evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for item in evidence:
        if not item.name.startswith("paper_claim_"):
            continue
        details = item.details or {}
        claim_id = details.get("result_id")
        if not claim_id or claim_id == "all_so_table":
            continue
        for prefix, obligation_name in (
            ("", f"{claim_id}:d>={details.get('target_minimum_distance')}"),
            ("optimality_", f"{claim_id}:d>={int(details.get('target_minimum_distance') or 0) + 1}"),
        ):
            status_key = f"{prefix}solver_status"
            command_key = f"{prefix}solver_command"
            if not details.get(status_key):
                continue
            cnf_prefix = "optimality_" if prefix else ""
            run = {
                "claim_id": claim_id,
                "obligation": obligation_name,
                "solver": "cadical",
                "status": details.get(status_key),
                "command": details.get(command_key),
                "argv": details.get(f"{prefix}solver_argv"),
                "cwd": details.get(f"{prefix}solver_cwd"),
                "solver_path": details.get(f"{prefix}solver_resolved_path") or details.get(f"{prefix}solver_path"),
                "returncode": details.get(f"{prefix}solver_returncode"),
                "elapsed_ms": details.get(f"{prefix}solver_elapsed_ms"),
                "timeout_sec": details.get(f"{prefix}solver_timeout_sec"),
                "memory_mb": details.get(f"{prefix}solver_memory_mb"),
                "cnf_path": details.get(f"{cnf_prefix}cnf_path"),
                "cnf_sha256": details.get(f"{cnf_prefix}cnf_sha256"),
                "cnf_variables": details.get(f"{cnf_prefix}cnf_variables"),
                "cnf_clauses": details.get(f"{cnf_prefix}cnf_clauses"),
            }
            runs.append({key: value for key, value in run.items() if value is not None})
    return runs


def _verifier_diagnostics_from_evidence(evidence: list[EvidenceItem]) -> list[VerifierDiagnostics]:
    diagnostics: list[VerifierDiagnostics] = []
    for item in evidence:
        if not item.name.startswith("paper_claim_"):
            continue
        details = item.details or {}
        claim_id = details.get("result_id")
        if not claim_id or claim_id == "all_so_table":
            continue
        target_d = details.get("target_minimum_distance")
        d_min = (
            details.get("matrix_minimum_distance")
            or details.get("binary_minimum_distance")
            or details.get("lucas_minimum_pairwise_distance")
            or details.get("minimum_distance")
        )
        low_weight_count = None
        if target_d is not None and isinstance(details.get("matrix_weight_distribution"), dict):
            low_weight_count = sum(
                int(count)
                for weight, count in details["matrix_weight_distribution"].items()
                if int(weight) < int(target_d) and int(weight) != 0
            )
        hashes = {
            key: value
            for key, value in {
                "cnf": details.get("cnf_sha256"),
                "optimality_cnf": details.get("optimality_cnf_sha256"),
                "artifact": details.get("artifact_sha256"),
            }.items()
            if value
        }
        diagnostics.append(
            VerifierDiagnostics(
                claim_id=claim_id,
                rank=details.get("gram_rank"),
                self_orthogonal=details.get("matrix_self_orthogonal"),
                d_min=d_min,
                weight_distribution=details.get("matrix_weight_distribution")
                or details.get("binary_weight_distribution")
                or details.get("lucas_ball_size_distribution")
                or details.get("weight_distribution"),
                low_weight_count=low_weight_count,
                solver_status=details.get("solver_status"),
                elapsed_ms=details.get("solver_elapsed_ms") or details.get("elapsed_ms"),
                memory_mb=details.get("solver_memory_mb"),
                artifact_hashes=hashes,
            )
        )
    return diagnostics


@traceable_run("solevolve.paper_output", run_type="chain")
def build_paper_workflow_output(
    *,
    paper_input: PaperWorkflowInput | None,
    evidence: list[EvidenceItem],
    decision: CoordinatorDecision,
    artifact_path: str | None = None,
) -> PaperWorkflowOutput:
    diagnostics = _verifier_diagnostics_from_evidence(evidence)
    missing_obligations = _missing_obligations(evidence)
    artifact_paths = [path for item in evidence for path in item.artifacts]
    if artifact_path:
        artifact_paths.append(artifact_path)
    return PaperWorkflowOutput(
        target=paper_input.target if paper_input else None,
        targets=paper_input.targets if paper_input else [],
        codetables_results=_codetables_results(evidence),
        solver_runs=_solver_runs_from_evidence(evidence),
        diagnostics=diagnostics,
        reflector_action=decision.reflector_action,
        decision=decision.model_dump(mode="python"),
        verdict=decision.verdict,
        missing_obligations=missing_obligations,
        artifact_paths=list(dict.fromkeys(str(path) for path in artifact_paths if path)),
    )


def _elapsed_ms(started_at: Any) -> float | None:
    if isinstance(started_at, (float, int)):
        return round((time.perf_counter() - float(started_at)) * 1000, 3)
    return None


def _build_run_summary(
    *,
    state: AgentState,
    max_turns: int,
    decision: CoordinatorDecision,
    evidence: list[EvidenceItem],
    summary_context: dict[str, Any],
) -> dict[str, Any]:
    evidence_status = evidence_status_summary(evidence)
    paper_input = state.get("paper_input")
    llm_runs = _llm_run_summaries(state)
    llm_totals = _llm_totals(llm_runs)
    solver_execution = _solver_execution_summary(evidence)
    paper_claim_execution = _paper_claim_execution_summary(evidence)
    codetables_refs = _codetables_refs_summary(evidence)
    missing_obligations = _missing_obligations(evidence)
    paper_output = state.get("paper_output")
    workflow_elapsed_ms = _elapsed_ms(summary_context.get("workflow_started_at"))
    graph_elapsed_ms = _elapsed_ms(summary_context.get("graph_started_at"))
    paper_output_solver_runs = paper_output.solver_runs if paper_output else []
    paper_output_solver_run_count = len(paper_output_solver_runs)
    paper_output_solver_elapsed_ms = round(
        sum(float(run.get("elapsed_ms") or 0.0) for run in paper_output_solver_runs),
        3,
    )
    decision_payload = decision.model_dump()
    metadata = dict(summary_context.get("metadata") or {})
    metadata.update(
        {
            "workflow_elapsed_ms": workflow_elapsed_ms,
            "graph_elapsed_ms": graph_elapsed_ms,
            "decision": decision_payload,
            "verdict": decision.verdict,
            "mode": decision.mode,
            "thresholds_applied": decision.thresholds_applied,
            "evidence_status": dict(evidence_status),
            "missing_obligations": missing_obligations or None,
            "paper_target": paper_input.target.model_dump(mode="python") if paper_input and paper_input.target else None,
            "paper_target_count": len(paper_input.targets) if paper_input else None,
            "llm_prompt_tokens": llm_totals["prompt_tokens"] or None,
            "llm_completion_tokens": llm_totals["completion_tokens"] or None,
            "llm_total_tokens": llm_totals["total_tokens"] or None,
            "estimated_total_cost_usd": llm_totals["estimated_total_cost_usd"],
            "cost_currency": llm_totals["currency"] if llm_totals["cost_available"] else None,
            "cost_pricing_sources": llm_totals["pricing_sources"] or None,
        }
    )
    if solver_execution:
        metadata.update(
            {
                "solver_backend": solver_execution.get("selected_solver"),
                "solver_path": solver_execution.get("selected_path"),
                "solver_command": solver_execution.get("command"),
                "solver_cwd": solver_execution.get("cwd"),
                "solver_version": solver_execution.get("version"),
                "solver_check_elapsed_ms": solver_execution.get("elapsed_ms"),
            }
        )
    if paper_claim_execution:
        first_claim = (paper_claim_execution.get("claim_results") or [{}])[0]
        metadata.update(
            {
                "paper_claim_verdict": paper_claim_execution.get("verdict"),
                "paper_claim_ids": [item.get("result_id") for item in paper_claim_execution.get("claim_results", [])],
                "paper_claim_min_distance_target": first_claim.get("target_minimum_distance"),
                "paper_claim_minimum_distance": first_claim.get("minimum_distance"),
                "paper_claim_solver_status": first_claim.get("solver_status"),
                "paper_claim_solver_command": first_claim.get("solver_command"),
                "paper_claim_solver_elapsed_ms": first_claim.get("solver_elapsed_ms"),
                "paper_claim_optimality_status": first_claim.get("optimality_solver_status"),
                "paper_claim_optimality_command": first_claim.get("optimality_solver_command"),
                "paper_claim_optimality_elapsed_ms": first_claim.get("optimality_solver_elapsed_ms"),
            }
        )
    if codetables_refs:
        metadata["codetables_refs"] = [
            {
                "name": ref.get("name"),
                "fetch_status": ref.get("fetch_status"),
                "url": ref.get("url"),
                "content_sha256": ref.get("content_sha256"),
            }
            for ref in codetables_refs
        ]
    if paper_output:
        metadata.update(
            {
                "paper_output_verdict": paper_output.verdict,
                "paper_output_solver_run_count": paper_output_solver_run_count,
                "paper_output_solver_elapsed_ms": paper_output_solver_elapsed_ms,
                "paper_output_diagnostic_count": len(paper_output.diagnostics),
                "paper_output_missing_obligations": paper_output.missing_obligations or None,
                "reflector_action": paper_output.reflector_action.model_dump(mode="python")
                if paper_output.reflector_action
                else None,
            }
        )
    return {
        "thread_id": summary_context.get("thread_id"),
        "model": summary_context.get("model"),
        "agent_steps": state.get("turn", 0),
        "max_turns": max_turns,
        "solver_preference": summary_context.get("solver_preference"),
        "workflow_elapsed_ms": workflow_elapsed_ms,
        "graph_elapsed_ms": graph_elapsed_ms,
        "solver_run_count": paper_output_solver_run_count,
        "solver_total_elapsed_ms": paper_output_solver_elapsed_ms,
        "paper_output_solver_run_count": paper_output_solver_run_count,
        "paper_output_solver_elapsed_ms": paper_output_solver_elapsed_ms,
        "decision": decision.decision,
        "mode": decision.mode,
        "verdict": decision.verdict,
        "thresholds_applied": decision.thresholds_applied,
        "evidence_used": decision.evidence_used,
        "paper_input": paper_input.model_dump(mode="python") if paper_input else None,
        "paper_output": paper_output.model_dump(mode="python") if paper_output else None,
        "verifier_diagnostics": [
            diagnostic.model_dump(mode="python") for diagnostic in state.get("verifier_diagnostics", [])
        ],
        "reflector_action": state.get("reflector_action").model_dump(mode="python")
        if state.get("reflector_action")
        else None,
        "evidence_status": evidence_status,
        "evidence": [item.model_dump() for item in evidence],
        "solver_execution": solver_execution,
        "paper_claim_execution": paper_claim_execution,
        "claim_results": paper_claim_execution.get("claim_results", []) if paper_claim_execution else [],
        "codetables_refs": codetables_refs,
        "missing_obligations": missing_obligations,
        "llm_runs": llm_runs,
        "llm_usage": {
            "prompt_tokens": llm_totals["prompt_tokens"],
            "completion_tokens": llm_totals["completion_tokens"],
            "total_tokens": llm_totals["total_tokens"],
        },
        "llm_cost": {
            "estimated_total_cost_usd": llm_totals["estimated_total_cost_usd"],
            "currency": llm_totals["currency"],
            "cost_available": llm_totals["cost_available"],
            "pricing_sources": llm_totals["pricing_sources"],
        },
        "estimated_total_cost_usd": llm_totals["estimated_total_cost_usd"],
        "cost_currency": llm_totals["currency"] if llm_totals["cost_available"] else None,
        "cost_pricing_sources": llm_totals["pricing_sources"],
        "message_count": len(state.get("messages", [])),
        "metadata": {key: value for key, value in metadata.items() if value is not None},
    }


def _coerce_evidence_items(items: list[Any] | None) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for index, item in enumerate(items or []):
        try:
            evidence.append(EvidenceItem.model_validate(item))
        except ValidationError as exc:
            evidence.append(
                EvidenceItem(
                    name=f"invalid_evidence_{index}",
                    status="UNKNOWN",
                    command_or_function="graph.evidence_validation",
                    summary="Invalid evidence item in graph state.",
                    artifacts=[],
                    error=exc.errors()[0]["msg"],
                )
            )
    return evidence


def _repro_check_node(
    *,
    artifact_dir: str | Path | None,
    repo_root: str | Path | None,
    solver_preference: str,
    paper_claim_id: str | None,
    cadical_path: str | None,
    paper_claim_timeout: int,
):
    def node(state: AgentState) -> AgentState:
        prior_evidence = _coerce_evidence_items(state.get("evidence", []))
        evidence = _coerce_evidence_items(
            run_repro_checks(
                artifact_dir=artifact_dir,
                repo_root=repo_root,
                solver_preference=solver_preference,
                paper_claim_id=paper_claim_id,
                cadical_path=cadical_path,
                paper_claim_timeout=paper_claim_timeout,
            )
        )
        evidence = prior_evidence + evidence
        decision = default_coordinator_decision(evidence)
        return {
            **state,
            "evidence": evidence,
            "next_instruction": state.get("next_instruction") or decision.next_instruction,
            "coordinator_decision": decision,
        }

    return node


def _paper_input_node(
    *,
    paper_claim_id: str | None,
    target_json: str | None,
    solver_preference: str,
    paper_claim_timeout: int,
):
    def node(state: AgentState) -> AgentState:
        paper_input = build_paper_workflow_input(
            goal=state.get("goal", ""),
            paper_claim_id=paper_claim_id,
            target_json=target_json,
            solver_preference=solver_preference,
            solver_budget_sec=paper_claim_timeout,
        )
        evidence = _coerce_evidence_items(state.get("evidence", []))
        evidence.append(
            EvidenceItem(
                name="paper_input",
                status="OK",
                command_or_function="build_paper_workflow_input",
                summary=f"targets={len(paper_input.targets)}; primary={paper_input.target.claim_id if paper_input.target else 'multi'}",
                artifacts=[],
                error=None,
                details=paper_input.model_dump(mode="python"),
            )
        )
        return {
            **state,
            "paper_input": paper_input,
            "evidence": evidence,
        }

    return node


def _json_object_from_text(content: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []

    fenced_blocks = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE))
    for match in reversed(fenced_blocks):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)

    for line in reversed(content.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        if isinstance(payload, dict):
            candidates.append(payload)

    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", content))):
        try:
            payload, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)

    for payload in candidates:
        if "decision" in payload:
            return payload
    for payload in candidates:
        if "action" in payload:
            return payload
    return candidates[0] if candidates else None


def _evolver_action_from_text(content: str) -> tuple[EvolverAction | None, str | None]:
    payload = _json_object_from_text(content)
    if payload is None:
        return None, "Evolver did not emit a valid EvolverAction JSON object."
    try:
        return EvolverAction.model_validate(payload), None
    except ValidationError as exc:
        return None, f"EvolverAction JSON failed validation: {exc.errors()[0]['msg']}"


def _evolver_action_node(
    *,
    artifact_dir: str | Path | None,
    cadical_path: str | None,
    paper_claim_id: str | None,
    paper_claim_timeout: int,
):
    def node(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        latest = messages[-1] if messages else None
        evidence = _coerce_evidence_items(state.get("evidence", []))
        if latest is None or getattr(latest, "name", None) != "evolver":
            return state
        action, error = _evolver_action_from_text(str(latest.content))
        if action is None:
            evidence.append(
                EvidenceItem(
                    name="evolver_action",
                    status="ERROR",
                    command_or_function="EvolverAction.model_validate",
                    summary="Invalid EvolverAction; coordinator enters repair mode.",
                    artifacts=[],
                    error=error,
                    details={"mode": "repair"},
                )
            )
            return {
                **state,
                "evidence": evidence,
                "next_instruction": "Repair the Evolver action JSON and target one whitelisted deterministic action.",
            }

        target_claim = action.claim_id or paper_claim_id
        new_items: list[EvidenceItem] = [
            EvidenceItem(
                name="evolver_action",
                status="OK",
                command_or_function="EvolverAction.model_validate",
                summary=f"action={action.action}; claim_id={target_claim}; reason={action.reason}",
                artifacts=[],
                error=None,
                details=action.model_dump(),
            )
        ]
        if action.action == "query_codetables":
            if target_claim:
                new_items.extend(_run_codetables_evidence_items(paper_claim_id=target_claim, artifact_dir=artifact_dir))
            else:
                new_items.append(
                    EvidenceItem(
                        name="codetables_lookup",
                        status="SKIPPED",
                        command_or_function="codetables_lookup",
                        summary="No paper claim id was available for codetables lookup.",
                        artifacts=[],
                        error=None,
                    )
                )
        elif action.action == "run_claim_proof":
            if target_claim:
                new_items.extend(_run_codetables_evidence_items(paper_claim_id=target_claim, artifact_dir=artifact_dir))
                new_items.extend(
                    _run_paper_claim_evidence_items(
                        paper_claim_id=target_claim,
                        artifact_dir=artifact_dir,
                        cadical_path=cadical_path,
                        timeout=action.timeout_sec or paper_claim_timeout,
                    )
                )
            else:
                new_items.append(
                    EvidenceItem(
                        name="paper_claim_reproduction",
                        status="SKIPPED",
                        command_or_function="reproduce_paper_claims",
                        summary="No paper claim id was available for claim proof.",
                        artifacts=[],
                        error=None,
                    )
                )
        elif action.action == "construction_search":
            new_items.append(
                EvidenceItem(
                    name="construction_search_request",
                    status="UNAVAILABLE",
                    command_or_function="coordinator.construction_search",
                    summary=(
                        "Construction search was requested, but no deterministic Lucas n=15 center-set "
                        "construction backend is registered yet. Public center-set artifacts remain required."
                    ),
                    artifacts=[],
                    error="Missing deterministic construction-search backend.",
                    details={
                        "action": action.action,
                        "claim_id": target_claim,
                        "timeout_sec": action.timeout_sec or paper_claim_timeout,
                        "supported_now": False,
                    },
                )
            )
        elif action.action == "propose_repair":
            new_items.append(
                EvidenceItem(
                    name="evolver_repair_proposal",
                    status="OK",
                    command_or_function="evolver.propose_repair",
                    summary=action.reason,
                    artifacts=[],
                    error=None,
                    details=action.model_dump(),
                )
            )
        elif action.action in {"request_reflection", "stop"}:
            pass

        evidence.extend(new_items)
        return {**state, "evidence": evidence, "next_instruction": action.reason}

    return node


def _dump_model(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def _agent_node(agent):
    def node(state: AgentState) -> AgentState:
        kwargs = {
            "goal": state.get("goal"),
            "evidence": [item.model_dump(mode="python") for item in _coerce_evidence_items(state.get("evidence", []))],
            "next_instruction": state.get("next_instruction"),
            "paper_input": _dump_model(state.get("paper_input")),
            "paper_output": _dump_model(state.get("paper_output")),
            "reflector_action": _dump_model(state.get("reflector_action")),
        }
        try:
            reply = agent.invoke(state.get("messages", []), **kwargs)
        except TypeError:
            reply = agent.invoke(
                state.get("messages", []),
                goal=kwargs["goal"],
                evidence=kwargs["evidence"],
                next_instruction=kwargs["next_instruction"],
            )
        next_state: AgentState = {"messages": state["messages"] + [reply], "turn": state["turn"] + 1}
        if agent.name == "reflector":
            decision = _decision_from_text(str(reply.content), _coerce_evidence_items(state.get("evidence", [])))
            next_state["coordinator_decision"] = decision
            next_state["next_instruction"] = decision.next_instruction
            if decision.reflector_action is not None:
                next_state["reflector_action"] = decision.reflector_action
        return next_state

    return node


def _route_before_agent(max_turns: int, next_node: str):
    def route(state: AgentState) -> str:
        if state["turn"] >= max_turns:
            return "finalize"
        return next_node

    return route


def _decision_from_text(content: str, evidence: list[EvidenceItem] | None = None) -> CoordinatorDecision:
    payload = _json_object_from_text(content)
    if isinstance(payload, dict):
        try:
            decision = CoordinatorDecision.model_validate(payload)
            return _apply_threshold_policy(decision, evidence or [])
        except ValidationError as exc:
            return fallback_coordinator_decision(
                reason=f"Reflector decision JSON failed validation: {exc.errors()[0]['msg']}",
                evidence=evidence,
            )
    return fallback_coordinator_decision(
        reason="Reflector did not emit a valid CoordinatorDecision JSON object.",
        evidence=evidence,
    )


REFLECTOR_THRESHOLDS = {
    "tau_rho": 0.01,
    "N_stag": 50,
    "tau_D": 0.15,
    "T_max": 300,
}


def _metrics_from_evidence(evidence: list[EvidenceItem]) -> ReflectorMetrics:
    metrics: dict[str, Any] = {}
    for item in evidence:
        details = item.details or {}
        if item.name.startswith("paper_claim_"):
            if details.get("claim_family") == "lucas_cubes":
                params = details.get("parameters") if isinstance(details.get("parameters"), dict) else {}
                metrics["target_parameters"] = (
                    f"Lucas Lambda_{params.get('n')}(1^{params.get('s')}); "
                    f"expected_center_count={details.get('lucas_expected_center_count')}"
                )
                if details.get("lucas_minimum_pairwise_distance") is not None:
                    metrics["d_min"] = details.get("lucas_minimum_pairwise_distance")
                if details.get("lucas_ball_size_distribution") is not None:
                    metrics["weight_distribution"] = details.get("lucas_ball_size_distribution")
            if details.get("target_minimum_distance") is not None:
                metrics["target_parameters"] = str(details.get("extended_parameters") or details.get("source_parameters"))
                metrics["d_min"] = details.get("matrix_minimum_distance")
                metrics["rank"] = details.get("gram_rank") or details.get("cnf_base_gram_rank")
                metrics["weight_distribution"] = details.get("matrix_weight_distribution")
            if details.get("solver_status"):
                metrics["solver_status"] = details.get("solver_status")
            if details.get("optimality_solver_status") == "TIMEOUT" or details.get("solver_status") == "TIMEOUT":
                metrics["timeout"] = True
        if item.name == "evolver_action" and item.status == "ERROR":
            metrics["solver_status"] = "ACTION_ERROR"
    try:
        return ReflectorMetrics.model_validate(metrics)
    except ValidationError:
        return ReflectorMetrics()


def _claim_obligations_satisfied(evidence: list[EvidenceItem]) -> bool:
    if any(item.name == "ternary_bch_d9_report" and item.status == "OK" for item in evidence):
        return True
    claim_items = [item for item in evidence if item.name.startswith("paper_claim_")]
    if not claim_items:
        return False
    concrete = [item for item in claim_items if item.details.get("result_id") and item.details.get("result_id") != "all_so_table"]
    return bool(concrete) and all(item.details.get("verdict") == "PASS" for item in concrete)


def _solver_check_is_ok(evidence: list[EvidenceItem]) -> bool:
    return any(item.name == "solver_check" and item.status == "OK" for item in evidence)


def _has_lucas_missing_center_artifact(evidence: list[EvidenceItem]) -> bool:
    for item in evidence:
        details = item.details or {}
        if details.get("claim_family") != "lucas_cubes":
            continue
        if "public_center_set_artifact" in (details.get("missing_obligations") or []):
            return True
    return False


def _normalize_lucas_text(text: str) -> str:
    replacements = {
        "targeting both s=11 and s=12 center counts": (
            "targeting both forbidden-run settings s=11 and s=12 with expected_center_count=2047"
        ),
        "for s=11 and s=12 center counts": (
            "for forbidden-run settings s=11 and s=12 with expected_center_count=2047"
        ),
    }
    updated = text
    for before, after in replacements.items():
        updated = updated.replace(before, after)
    return updated


def _correct_decision_contradictions(decision: CoordinatorDecision, evidence: list[EvidenceItem]) -> CoordinatorDecision:
    updates: dict[str, Any] = {}
    text = " ".join(
        part
        for part in (
            decision.reason,
            decision.next_instruction,
            decision.reflector_action.reason if decision.reflector_action else "",
            decision.reflector_action.next_instruction if decision.reflector_action else "",
        )
        if part
    ).lower()
    if _solver_check_is_ok(evidence) and "solver" in text and (
        "unavailable" in text or "not available" in text or "no solver is available" in text
    ):
        updates["reason"] = (
            f"{decision.reason} Coordinator correction: solver_check.status is OK; "
            "for Lucas claims the solver is not applicable unless a SAT/ILP encoding is explicitly executed."
        )
    if decision.next_instruction:
        normalized_next = _normalize_lucas_text(decision.next_instruction)
        if normalized_next != decision.next_instruction:
            updates["next_instruction"] = normalized_next

    if _has_lucas_missing_center_artifact(evidence) and decision.reflector_action is not None:
        action_updates: dict[str, Any] = {}
        if decision.reflector_action.action == "sat_repair":
            action_updates["action"] = "artifact_repair"
        normalized_action_next = _normalize_lucas_text(decision.reflector_action.next_instruction)
        if normalized_action_next != decision.reflector_action.next_instruction:
            action_updates["next_instruction"] = normalized_action_next
        if not decision.reflector_action.fixed_target or "expected_center_count" not in decision.reflector_action.fixed_target:
            action_updates["fixed_target"] = (
                "Lucas n=15 center-set artifacts for forbidden-run settings s=11 and s=12; "
                "expected_center_count=2047 for each claim."
            )
        if action_updates:
            updates["reflector_action"] = decision.reflector_action.model_copy(update=action_updates)

    if not updates:
        return decision
    return decision.model_copy(update=updates)


def _apply_threshold_policy(decision: CoordinatorDecision, evidence: list[EvidenceItem]) -> CoordinatorDecision:
    metrics = _metrics_from_evidence(evidence)
    thresholds = dict(REFLECTOR_THRESHOLDS)
    thresholds["metrics"] = metrics.model_dump()
    updates: dict[str, Any] = {"thresholds_applied": thresholds}

    if decision.verdict == "PASS" and not _claim_obligations_satisfied(evidence):
        updates.update(
            {
                "verdict": "PARTIAL",
                "decision": "stop",
                "mode": "repair",
                "reason": f"{decision.reason} Coordinator downgraded PASS because required proof obligations are missing.",
                "next_instruction": "",
            }
        )
    elif metrics.timeout:
        updates.setdefault("mode", "exploration")
        if decision.decision == "continue":
            updates["next_instruction"] = decision.next_instruction or "Switch encoding or restart after timeout."
    elif metrics.solver_status in {"ACTION_ERROR", "ERROR"}:
        updates.setdefault("mode", "repair")
    elif metrics.improvement_rate is not None and metrics.improvement_rate > thresholds["tau_rho"]:
        updates.setdefault("mode", "exploitation")
    elif metrics.diversity is not None and metrics.diversity < thresholds["tau_D"]:
        updates.setdefault("mode", "exploration")
    elif metrics.stagnation_length >= thresholds["N_stag"]:
        updates.setdefault("mode", "repair")
    else:
        updates.setdefault("mode", decision.mode)
    updated = decision.model_copy(update=updates)
    if updated.reflector_action is None:
        action_name = "finalize" if updated.decision == "stop" else "sat_repair" if updated.mode == "repair" else "encoding_switch"
        updated = updated.model_copy(
            update={
                "reflector_action": ReflectorAction(
                    action=action_name,
                    mode=updated.mode,
                    reason=updated.reason,
                    fixed_target=metrics.target_parameters,
                    next_instruction=updated.next_instruction,
                    evidence_used=updated.evidence_used,
                )
            }
        )
    return _correct_decision_contradictions(updated, evidence)


def _reflector_decision(state: AgentState) -> CoordinatorDecision:
    decision = state.get("coordinator_decision")
    if decision is not None:
        return decision
    return fallback_coordinator_decision(
        reason="Coordinator decision missing from graph state.",
        evidence=state.get("evidence", []),
    )


def _normalize_decision_evidence_used(
    decision: CoordinatorDecision,
    evidence: list[EvidenceItem],
) -> CoordinatorDecision:
    if decision.verdict != "PASS":
        return decision
    updates: dict[str, Any] = {}
    if decision.decision != "stop":
        updates["decision"] = "stop"
        updates["next_instruction"] = ""
        updates["reason"] = f"{decision.reason} Coordinator finalized PASS as a terminal decision."
    used = list(dict.fromkeys(decision.evidence_used))
    for item in evidence:
        if item.status == "OK" and item.name not in used:
            used.append(item.name)
    if used != decision.evidence_used:
        updates["evidence_used"] = used
    if not updates:
        return decision
    return decision.model_copy(update=updates)


def _route_after_reflection(max_turns: int):
    def route(state: AgentState) -> Literal["generator", "finalize"]:
        if state["turn"] >= max_turns:
            return "finalize"
        decision = _reflector_decision(state)
        if decision.decision == "stop":
            return "finalize"
        return "generator"

    return route


def _coordinator_finalize_node(
    max_turns: int,
    *,
    artifact_dir: str | Path | None = None,
    summary_context: dict[str, Any] | None = None,
):
    def node(state: AgentState) -> AgentState:
        evidence = _coerce_evidence_items(state.get("evidence", []))
        decision = state.get("coordinator_decision") or fallback_coordinator_decision(
            reason="Coordinator decision missing before finalization.",
            evidence=evidence,
        )
        if not isinstance(decision, CoordinatorDecision):
            try:
                decision = CoordinatorDecision.model_validate(decision)
            except ValidationError as exc:
                decision = fallback_coordinator_decision(
                    reason=f"Coordinator decision failed validation during finalization: {exc.errors()[0]['msg']}",
                    evidence=evidence,
                )
        decision = _normalize_decision_evidence_used(decision, evidence)
        if state.get("turn", 0) >= max_turns and decision.decision == "continue":
            decision = fallback_coordinator_decision(
                reason=f"Stopped after reaching max_turns={max_turns}.",
                evidence=evidence,
            )
        expected_artifact_path = str(Path(artifact_dir) / "last_run_summary.json") if artifact_dir is not None else None
        paper_output = build_paper_workflow_output(
            paper_input=state.get("paper_input"),
            evidence=evidence,
            decision=decision,
            artifact_path=expected_artifact_path,
        )
        verifier_diagnostics = paper_output.diagnostics
        next_state: AgentState = {
            "coordinator_decision": decision,
            "next_instruction": decision.next_instruction,
            "paper_output": paper_output,
            "verifier_diagnostics": verifier_diagnostics,
        }
        if decision.reflector_action is not None:
            next_state["reflector_action"] = decision.reflector_action
        if summary_context is not None:
            summary_state = {**state, **next_state}
            summary = _build_run_summary(
                state=summary_state,
                max_turns=max_turns,
                decision=decision,
                evidence=evidence,
                summary_context=summary_context,
            )
            next_state["run_summary"] = summary
            if artifact_dir is not None:
                artifact_path = write_artifact_summary(Path(artifact_dir), summary)
                next_state["artifact_path"] = str(artifact_path)
        return next_state

    return node


def build_sol_evolve_network(
    *,
    generator: GeneratorAgent,
    evolver: EvolverAgent,
    verifier: VerifierAgent,
    reflector: ReflectorAgent,
    max_turns: int = 6,
    artifact_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    solver_preference: str = "cadical",
    paper_claim_id: str | None = None,
    target_json: str | None = None,
    cadical_path: str | None = None,
    paper_claim_timeout: int = 1200,
    summary_context: dict[str, Any] | None = None,
):
    """Construct a LangGraph Deep Agents network for SolEvolve roles."""
    graph = StateGraph(AgentState)

    graph.add_node(
        "paper_input",
        _paper_input_node(
            paper_claim_id=paper_claim_id,
            target_json=target_json,
            solver_preference=solver_preference,
            paper_claim_timeout=paper_claim_timeout,
        ),
    )
    graph.add_node(
        "repro_check",
        _repro_check_node(
            artifact_dir=artifact_dir,
            repo_root=repo_root,
            solver_preference=solver_preference,
            paper_claim_id=paper_claim_id,
            cadical_path=cadical_path,
            paper_claim_timeout=paper_claim_timeout,
        ),
    )
    graph.add_node("generator", _agent_node(generator))
    graph.add_node("evolver", _agent_node(evolver))
    graph.add_node(
        "evolver_action",
        _evolver_action_node(
            artifact_dir=artifact_dir,
            cadical_path=cadical_path,
            paper_claim_id=paper_claim_id,
            paper_claim_timeout=paper_claim_timeout,
        ),
    )
    graph.add_node("verifier", _agent_node(verifier))
    graph.add_node("reflector", _agent_node(reflector))
    graph.add_node(
        "coordinator_finalize",
        _coordinator_finalize_node(
            max_turns,
            artifact_dir=artifact_dir,
            summary_context=summary_context,
        ),
    )

    graph.add_edge("paper_input", "repro_check")
    graph.add_conditional_edges(
        "repro_check",
        _route_before_agent(max_turns, "generator"),
        {"generator": "generator", "finalize": "coordinator_finalize"},
    )
    graph.add_conditional_edges(
        "generator",
        _route_before_agent(max_turns, "evolver"),
        {"evolver": "evolver", "finalize": "coordinator_finalize"},
    )
    graph.add_edge("evolver", "evolver_action")
    graph.add_conditional_edges(
        "evolver_action",
        _route_before_agent(max_turns, "verifier"),
        {"verifier": "verifier", "finalize": "coordinator_finalize"},
    )
    graph.add_conditional_edges(
        "verifier",
        _route_before_agent(max_turns, "reflector"),
        {"reflector": "reflector", "finalize": "coordinator_finalize"},
    )
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflection(max_turns),
        {
            "generator": "generator",
            "finalize": "coordinator_finalize",
        },
    )
    graph.add_edge("coordinator_finalize", END)

    graph.set_entry_point("paper_input")
    return graph.compile()
