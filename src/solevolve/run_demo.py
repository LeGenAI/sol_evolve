from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .agents import EvolverAgent, GeneratorAgent, ReflectorAgent, VerifierAgent
from .config import load_settings
from .graph import build_sol_evolve_network
from .llm import build_llm
from .tracing import base_metadata, traceable_run, write_artifact_summary


@dataclass
class DryRunAgent:
    name: str

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1].content if messages else ""
        content = (
            f"dry-run {self.name}: received {len(messages)} message(s). "
            f"Last message summary: {str(last)[:120]}"
        )
        return AIMessage(content=content, name=self.name)

    def kickoff(self, user_text: str) -> list[BaseMessage]:
        return [HumanMessage(content=user_text, name="human")]


def build_agents(settings):
    llm = build_llm(settings)
    return (
        GeneratorAgent(llm),
        EvolverAgent(llm),
        VerifierAgent(llm),
        ReflectorAgent(llm),
    )


def build_dry_agents():
    return (
        DryRunAgent("generator"),
        DryRunAgent("evolver"),
        DryRunAgent("verifier"),
        DryRunAgent("reflector"),
    )


@traceable_run("solevolve.run", run_type="chain")
def run_workflow(*, goal: str, max_turns: int, thread_id: str, dry_run: bool, settings):
    if dry_run:
        generator, evolver, verifier, reflector = build_dry_agents()
    else:
        generator, evolver, verifier, reflector = build_agents(settings)

    graph = build_sol_evolve_network(
        generator=generator,
        evolver=evolver,
        verifier=verifier,
        reflector=reflector,
        max_turns=max_turns,
    )
    final_state = graph.invoke(
        {"messages": generator.kickoff(goal), "turn": 0},
        config={
            "configurable": {"thread_id": thread_id},
            "metadata": base_metadata(
                settings,
                dry_run=dry_run,
                thread_id=thread_id,
                max_turns=max_turns,
            ),
            "tags": list(settings.trace_tags),
        },
    )
    summary = {
        "dry_run": dry_run,
        "thread_id": thread_id,
        "message_count": len(final_state["messages"]),
        "metadata": base_metadata(settings, dry_run=dry_run, thread_id=thread_id),
    }
    artifact_path = write_artifact_summary(settings.artifact_dir, summary)
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
        "--dry-run",
        action="store_true",
        help="Run the graph with deterministic stub agents; no LLM API key or solver is required.",
    )
    args = parser.parse_args()

    settings = load_settings(require_api_key=not args.dry_run)
    thread_id = args.thread_id or f"demo-{uuid.uuid4()}"
    final_state, artifact_path = run_workflow(
        goal=args.goal,
        max_turns=args.max_turns or settings.max_turns,
        thread_id=thread_id,
        dry_run=args.dry_run,
        settings=settings,
    )

    for msg in final_state["messages"]:
        role = getattr(msg, "type", "ai").upper()
        name = getattr(msg, "name", None)
        label = f"{role}" if not name else f"{role}:{name}"
        print(f"[{label}] {msg.content}")
    print(json.dumps({"artifact_summary": str(artifact_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
