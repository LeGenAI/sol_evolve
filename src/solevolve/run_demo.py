from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace
from typing import Any

from .agents import EvolverAgent, GeneratorAgent, ReflectorAgent, VerifierAgent
from .contracts import CoordinatorDecision, default_coordinator_decision
from .config import load_settings
from .graph import _build_paper_workflow_input_impl, build_sol_evolve_network, evidence_status_summary
from .llm import build_llm
from .tracing import base_metadata, flush_tracing, sanitize_trace_payload, traceable_run, update_current_run_context


def build_agents(settings):
    llm = build_llm(settings)
    return (
        GeneratorAgent(llm),
        EvolverAgent(llm),
        VerifierAgent(llm),
        ReflectorAgent(llm),
    )

def _coordinator_decision_from_state(final_state) -> CoordinatorDecision:
    coordinator_decision = final_state.get("coordinator_decision")
    if coordinator_decision is None:
        coordinator_decision = default_coordinator_decision(final_state.get("evidence", []))
    if not isinstance(coordinator_decision, CoordinatorDecision):
        coordinator_decision = CoordinatorDecision.model_validate(coordinator_decision)
    return coordinator_decision


def _sanitize_run_workflow_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "goal": inputs.get("goal"),
        "max_turns": inputs.get("max_turns"),
        "thread_id": inputs.get("thread_id"),
        "target_json": inputs.get("target_json"),
    }
    settings = inputs.get("settings")
    try:
        payload["paper_input"] = _build_paper_workflow_input_impl(
            goal=inputs.get("goal") or "",
            paper_claim_id=getattr(settings, "paper_claim_id", None),
            target_json=inputs.get("target_json"),
            solver_preference=getattr(settings, "solver_preference", "cadical"),
            solver_budget_sec=getattr(settings, "paper_claim_timeout", 300),
        ).model_dump(mode="python")
    except Exception as exc:
        payload["paper_input_error"] = str(exc)
    return sanitize_trace_payload(payload)


def _sanitize_run_workflow_outputs(output: Any) -> dict[str, Any]:
    try:
        final_state, artifact_path = output
        coordinator_decision = _coordinator_decision_from_state(final_state)
        payload = {
            "artifact_path": str(artifact_path),
            "agent_steps": final_state.get("turn"),
            "decision": coordinator_decision.decision,
            "mode": coordinator_decision.mode,
            "verdict": coordinator_decision.verdict,
            "evidence_used": coordinator_decision.evidence_used,
            "evidence_status": evidence_status_summary(final_state.get("evidence", [])),
            "message_count": len(final_state.get("messages", [])),
        }
        summary = final_state.get("run_summary") or {}
        for key in (
            "solver_execution",
            "paper_claim_execution",
            "claim_results",
            "codetables_refs",
            "llm_usage",
            "llm_cost",
            "solver_run_count",
            "solver_total_elapsed_ms",
            "paper_output_solver_run_count",
            "paper_output_solver_elapsed_ms",
            "estimated_total_cost_usd",
            "cost_currency",
            "cost_pricing_sources",
            "workflow_elapsed_ms",
            "graph_elapsed_ms",
        ):
            if summary.get(key):
                payload[key] = summary[key]
        if summary.get("paper_input"):
            payload["paper_input"] = summary["paper_input"]
        if summary.get("paper_output"):
            payload["paper_output"] = summary["paper_output"]
    except Exception:
        payload = {"output": output}
    return sanitize_trace_payload(payload)


@traceable_run(
    "solevolve.run",
    run_type="chain",
    process_inputs=_sanitize_run_workflow_inputs,
    process_outputs=_sanitize_run_workflow_outputs,
)
def run_workflow(*, goal: str, max_turns: int, thread_id: str, settings, target_json: str | None = None):
    workflow_started = time.perf_counter()
    generator, evolver, verifier, reflector = build_agents(settings)

    metadata = base_metadata(
        settings,
        thread_id=thread_id,
        max_turns=max_turns,
        run_kind="hybrid_repro",
        solver_preference=settings.solver_preference,
    )
    summary_context: dict[str, Any] = {
        "thread_id": thread_id,
        "model": settings.openrouter_model,
        "max_turns": max_turns,
        "solver_preference": settings.solver_preference,
        "workflow_started_at": workflow_started,
        "graph_started_at": None,
        "metadata": metadata,
    }
    graph = build_sol_evolve_network(
        generator=generator,
        evolver=evolver,
        verifier=verifier,
        reflector=reflector,
        max_turns=max_turns,
        artifact_dir=settings.artifact_dir,
        solver_preference=settings.solver_preference,
        paper_claim_id=settings.paper_claim_id,
        target_json=target_json,
        cadical_path=settings.cadical_path,
        paper_claim_timeout=settings.paper_claim_timeout,
        summary_context=summary_context,
    )
    graph_started = time.perf_counter()
    summary_context["graph_started_at"] = graph_started
    final_state = graph.invoke(
        {"goal": goal, "messages": generator.kickoff(goal), "turn": 0, "evidence": [], "next_instruction": None},
        config={
            "configurable": {"thread_id": thread_id},
            "metadata": metadata,
            "tags": list(settings.trace_tags),
        },
    )
    _coordinator_decision_from_state(final_state)
    summary = final_state.get("run_summary") or {}
    update_current_run_context(metadata=summary.get("metadata") or metadata, tags=list(settings.trace_tags))
    artifact_path = final_state.get("artifact_path") or str(settings.artifact_dir / "last_run_summary.json")
    return final_state, artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SolEvolve Deep Agents demo.")
    parser.add_argument(
        "goal",
        type=str,
        nargs="?",
        default="Goal: outline a SAT/GA pipeline for new code search.",
        help="Initial user message.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max turns for the demo loop.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Optional thread_id for repeated runs/checkpointing.",
    )
    parser.add_argument(
        "--paper-claim-id",
        type=str,
        default=None,
        help="Paper claim to reproduce, e.g. ternary_bch_d9, gf4_hermitian, gf5_so, or all_so_table.",
    )
    parser.add_argument(
        "--target-json",
        type=str,
        default=None,
        help="Inline JSON or path to JSON describing PaperWorkflowInput or PaperTarget.",
    )
    parser.add_argument(
        "--claim-timeout-sec",
        type=int,
        default=None,
        help="Timeout in seconds for each SAT-backed paper claim obligation.",
    )
    parser.add_argument(
        "--cadical-path",
        type=str,
        default=None,
        help="Optional CaDiCaL binary or source directory. Directories are resolved via build/cadical then cadical.",
    )
    args = parser.parse_args()

    try:
        settings = load_settings(require_api_key=True)
        if args.paper_claim_id or args.claim_timeout_sec or args.cadical_path:
            settings = replace(
                settings,
                paper_claim_id=args.paper_claim_id or settings.paper_claim_id,
                paper_claim_timeout=args.claim_timeout_sec or settings.paper_claim_timeout,
                cadical_path=args.cadical_path or settings.cadical_path,
            )
        thread_id = args.thread_id or f"demo-{uuid.uuid4()}"
        final_state, artifact_path = run_workflow(
            goal=args.goal,
            max_turns=args.max_turns or settings.max_turns,
            thread_id=thread_id,
            settings=settings,
            target_json=args.target_json,
        )

        for msg in final_state["messages"]:
            role = getattr(msg, "type", "ai").upper()
            name = getattr(msg, "name", None)
            label = f"{role}" if not name else f"{role}:{name}"
            print(f"[{label}] {msg.content}")
        coordinator_decision = _coordinator_decision_from_state(final_state)
        print(
            json.dumps(
                {
                    "artifact_summary": str(artifact_path),
                    "decision": coordinator_decision.decision,
                    "mode": coordinator_decision.mode,
                    "verdict": coordinator_decision.verdict,
                    "evidence_used": coordinator_decision.evidence_used,
                    "evidence_status": evidence_status_summary(final_state.get("evidence", [])),
                    "estimated_total_cost_usd": (final_state.get("run_summary") or {})
                    .get("llm_cost", {})
                    .get("estimated_total_cost_usd"),
                    "solver_execution": (final_state.get("run_summary") or {}).get("solver_execution"),
                    "paper_claim_execution": (final_state.get("run_summary") or {}).get("paper_claim_execution"),
                    "claim_results": (final_state.get("run_summary") or {}).get("claim_results"),
                    "codetables_refs": (final_state.get("run_summary") or {}).get("codetables_refs"),
                    "paper_input": (final_state.get("run_summary") or {}).get("paper_input"),
                    "paper_output": (final_state.get("run_summary") or {}).get("paper_output"),
                },
                sort_keys=True,
            )
        )
    finally:
        flush_tracing()


if __name__ == "__main__":
    main()
