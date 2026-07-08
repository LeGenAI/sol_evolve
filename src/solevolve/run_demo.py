from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace
from typing import Any

from .agents import EvolverAgent, GeneratorAgent, ReflectorAgent, VerifierAgent
from .contracts import CoordinatorDecision, HybridGAPolicy, default_coordinator_decision
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
        "hybrid_ga_policy": inputs.get("hybrid_ga_policy"),
    }
    settings = inputs.get("settings")
    try:
        payload["paper_input"] = _build_paper_workflow_input_impl(
            goal=inputs.get("goal") or "",
            paper_claim_id=getattr(settings, "paper_claim_id", None),
            target_json=inputs.get("target_json"),
            solver_preference=getattr(settings, "solver_preference", "cadical"),
            solver_budget_sec=getattr(settings, "paper_claim_timeout", 300),
            hybrid_ga_policy=inputs.get("hybrid_ga_policy"),
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
            "hybrid_ga_reports",
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
def run_workflow(
    *,
    goal: str,
    max_turns: int,
    thread_id: str,
    settings,
    target_json: str | None = None,
    hybrid_ga_policy: HybridGAPolicy | None = None,
):
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
        hybrid_ga_policy=hybrid_ga_policy,
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
        help=(
            "Paper claim to reproduce, e.g. ternary_bch_d9, gf4_hermitian, gf5_so, all_so_table, "
            "binary_22_11_7_hybrid_ga, or cegar_eager_scaling."
        ),
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
    parser.add_argument("--hybrid-ga-mode", choices=["off", "archived", "replay", "live", "frontier_repair"], default=None, help="Enable the coordinator-gated Hybrid SAT-GA branch.")
    parser.add_argument("--hybrid-ga-seed", type=int, default=None, help="Deterministic seed for Hybrid SAT-GA replay/live modes.")
    parser.add_argument("--hybrid-ga-population", type=int, default=None, help="Population size for Hybrid SAT-GA replay/live modes.")
    parser.add_argument("--hybrid-ga-generations", type=int, default=None, help="Generation budget for Hybrid SAT-GA replay/live modes.")
    parser.add_argument("--hybrid-ga-repair-interval", type=int, default=None, help="SAT repair interval in Hybrid SAT-GA replay/live modes; 0 disables repair.")
    parser.add_argument("--hybrid-ga-timeout-sec", type=int, default=None, help="Timeout in seconds for each Hybrid SAT-GA SAT-backed step.")
    parser.add_argument("--hybrid-ga-target-distance", type=int, default=None, help="Target minimum distance for Hybrid SAT-GA frontier repair.")
    parser.add_argument("--hybrid-ga-frontier-distance", type=int, default=None, help="Frontier seed distance for Hybrid SAT-GA frontier repair.")
    parser.add_argument("--hybrid-ga-frontier-seed-count", type=int, default=None, help="Number of frontier SAT seeds to request.")
    parser.add_argument("--hybrid-ga-frontier-target-timeout-sec", type=int, default=None, help="Timeout for the direct target-distance SAT attempt.")
    parser.add_argument("--hybrid-ga-frontier-seed-timeout-sec", type=int, default=None, help="Timeout per frontier seed SAT attempt.")
    parser.add_argument("--hybrid-ga-repair-strategy", type=str, default=None, help="SAT repair strategy for frontier repair mode.")
    args = parser.parse_args()

    try:
        settings = load_settings(require_api_key=True)
        if (
            args.paper_claim_id
            or args.claim_timeout_sec
            or args.cadical_path
            or args.hybrid_ga_mode
            or args.hybrid_ga_seed is not None
            or args.hybrid_ga_population is not None
            or args.hybrid_ga_generations is not None
            or args.hybrid_ga_repair_interval is not None
            or args.hybrid_ga_timeout_sec is not None
            or args.hybrid_ga_target_distance is not None
            or args.hybrid_ga_frontier_distance is not None
            or args.hybrid_ga_frontier_seed_count is not None
            or args.hybrid_ga_frontier_target_timeout_sec is not None
            or args.hybrid_ga_frontier_seed_timeout_sec is not None
            or args.hybrid_ga_repair_strategy is not None
        ):
            settings = replace(
                settings,
                paper_claim_id=args.paper_claim_id or settings.paper_claim_id,
                paper_claim_timeout=args.claim_timeout_sec or settings.paper_claim_timeout,
                cadical_path=args.cadical_path or settings.cadical_path,
                hybrid_ga_mode=args.hybrid_ga_mode or settings.hybrid_ga_mode,
                hybrid_ga_seed=args.hybrid_ga_seed if args.hybrid_ga_seed is not None else settings.hybrid_ga_seed,
                hybrid_ga_population=args.hybrid_ga_population if args.hybrid_ga_population is not None else settings.hybrid_ga_population,
                hybrid_ga_generations=args.hybrid_ga_generations if args.hybrid_ga_generations is not None else settings.hybrid_ga_generations,
                hybrid_ga_repair_interval=args.hybrid_ga_repair_interval if args.hybrid_ga_repair_interval is not None else settings.hybrid_ga_repair_interval,
                hybrid_ga_timeout_sec=args.hybrid_ga_timeout_sec if args.hybrid_ga_timeout_sec is not None else settings.hybrid_ga_timeout_sec,
                hybrid_ga_target_distance=args.hybrid_ga_target_distance if args.hybrid_ga_target_distance is not None else settings.hybrid_ga_target_distance,
                hybrid_ga_frontier_distance=args.hybrid_ga_frontier_distance if args.hybrid_ga_frontier_distance is not None else settings.hybrid_ga_frontier_distance,
                hybrid_ga_frontier_seed_count=args.hybrid_ga_frontier_seed_count if args.hybrid_ga_frontier_seed_count is not None else settings.hybrid_ga_frontier_seed_count,
                hybrid_ga_frontier_target_timeout_sec=args.hybrid_ga_frontier_target_timeout_sec if args.hybrid_ga_frontier_target_timeout_sec is not None else settings.hybrid_ga_frontier_target_timeout_sec,
                hybrid_ga_frontier_seed_timeout_sec=args.hybrid_ga_frontier_seed_timeout_sec if args.hybrid_ga_frontier_seed_timeout_sec is not None else settings.hybrid_ga_frontier_seed_timeout_sec,
                hybrid_ga_repair_strategy=args.hybrid_ga_repair_strategy if args.hybrid_ga_repair_strategy is not None else settings.hybrid_ga_repair_strategy,
            )
        hybrid_ga_policy = HybridGAPolicy(
            enabled=settings.hybrid_ga_mode != "off",
            mode=settings.hybrid_ga_mode,
            seed=settings.hybrid_ga_seed,
            population=settings.hybrid_ga_population,
            generations=settings.hybrid_ga_generations,
            repair_interval=settings.hybrid_ga_repair_interval,
            timeout_sec=settings.hybrid_ga_timeout_sec,
            solver_preference=settings.solver_preference,
            target_distance=settings.hybrid_ga_target_distance,
            frontier_distance=settings.hybrid_ga_frontier_distance,
            frontier_seed_count=settings.hybrid_ga_frontier_seed_count,
            frontier_target_timeout_sec=settings.hybrid_ga_frontier_target_timeout_sec,
            frontier_seed_timeout_sec=settings.hybrid_ga_frontier_seed_timeout_sec,
            repair_strategy=settings.hybrid_ga_repair_strategy,
        )
        thread_id = args.thread_id or f"demo-{uuid.uuid4()}"
        final_state, artifact_path = run_workflow(
            goal=args.goal,
            max_turns=args.max_turns or settings.max_turns,
            thread_id=thread_id,
            settings=settings,
            target_json=args.target_json,
            hybrid_ga_policy=hybrid_ga_policy,
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
                    "hybrid_ga_reports": (final_state.get("run_summary") or {}).get("hybrid_ga_reports"),
                },
                sort_keys=True,
            )
        )
    finally:
        flush_tracing()


if __name__ == "__main__":
    main()
