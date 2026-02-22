from __future__ import annotations

import argparse
import uuid

from .agents import EvolverAgent, GeneratorAgent, ReflectorAgent, VerifierAgent
from .config import load_settings
from .graph import AgentState, build_sol_evolve_network
from .llm import build_llm


def build_agents(settings):
    llm = build_llm(settings)
    return (
        GeneratorAgent(llm),
        EvolverAgent(llm),
        VerifierAgent(llm),
        ReflectorAgent(llm),
    )


def main():
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
    args = parser.parse_args()

    settings = load_settings()
    generator, evolver, verifier, reflector = build_agents(settings)
    graph = build_sol_evolve_network(
        generator=generator,
        evolver=evolver,
        verifier=verifier,
        reflector=reflector,
        max_turns=args.max_turns or settings.max_turns,
    )

    # thread id is useful when adding checkpointing/human-in-loop later
    thread_id = args.thread_id or f"demo-{uuid.uuid4()}"
    final_state = graph.invoke(
        {"messages": generator.kickoff(args.goal), "turn": 0},
        config={"configurable": {"thread_id": thread_id}},
    )

    for msg in final_state["messages"]:
        role = getattr(msg, "type", "ai").upper()
        name = getattr(msg, "name", None)
        label = f"{role}" if not name else f"{role}:{name}"
        print(f"[{label}] {msg.content}")


if __name__ == "__main__":
    main()
