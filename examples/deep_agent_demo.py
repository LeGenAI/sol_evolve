from __future__ import annotations

import argparse
import sys
import uuid

from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .config import load_settings
from .llm import build_llm


def main():
    parser = argparse.ArgumentParser(description="SolEvolve deep-agents middleware demo.")
    parser.add_argument(
        "goal",
        type=str,
        nargs="?",
        default="Plan a SAT search workflow, jot todos, and save notes to /memories/sat_notes.md.",
        help="Initial user request.",
    )
    parser.add_argument("--thread-id", type=str, default=None, help="Thread id for checkpointing/resume.")
    args = parser.parse_args()

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
        from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware
        from langgraph.store.memory import InMemoryStore
    except ImportError:
        print("deepagents is not installed or missing expected middleware. Install with `pip3 install deepagents` to run this demo.")
        sys.exit(1)

    settings = load_settings()
    llm = build_llm(settings, temperature=0.2)

    @tool
    def count_codewords(k: int) -> str:
        """Return the number of non-zero messages (2^k - 1)."""
        if k < 1:
            return "k must be >= 1"
        return str(2**k - 1)

    subagents = [
        {
            "name": "code-math",
            "description": "Helps with quick code math (counts, bounds).",
            "system_prompt": "Use tools to compute simple properties; respond with brief results.",
            "tools": [count_codewords],
        }
    ]

    store = InMemoryStore()
    backend = lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={"/memories/": StoreBackend(rt)},
    )

    checkpointer = MemorySaver()
    agent = create_deep_agent(
        model=llm,
        middleware=[
            FilesystemMiddleware(backend=backend),
            SubAgentMiddleware(
                default_model=settings.openrouter_model,
                default_tools=[],
                subagents=subagents,
            ),
        ],
        interrupt_on={"write_file": {"allowed_decisions": ["approve", "edit", "reject"]}},
        checkpointer=checkpointer,
        store=store,
    )

    thread_id = args.thread_id or f"deep-demo-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": args.goal}]}, config=config)

    if "__interrupt__" in result:
        interrupts = result["__interrupt__"][0].value
        actions = interrupts["action_requests"]
        review_configs = {cfg["action_name"]: cfg for cfg in interrupts["review_configs"]}

        decisions = []
        for action in actions:
            cfg = review_configs[action["name"]]
            allowed = ", ".join(cfg["allowed_decisions"])
            print(f"\nTool call pending: {action['name']} {action['args']} (allowed: {allowed})")
            choice = ""
            while choice not in cfg["allowed_decisions"]:
                choice = input(f"Decision [{allowed}]: ").strip()
            if choice == "edit":
                edited_path = input("New path (blank to keep): ").strip() or action["args"].get("path", "")
                edited_content = input("New content (blank to keep): ").strip() or action["args"].get("content", "")
                decisions.append(
                    {"type": "edit", "edited_action": {"name": action["name"], "args": {"path": edited_path, "content": edited_content}}}
                )
            elif choice == "approve":
                decisions.append({"type": "approve"})
            elif choice == "reject":
                decisions.append({"type": "reject"})
            else:
                raise ValueError(f"Unsupported decision: {choice}")

        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    for msg in result["messages"]:
        role = msg.get("role", "unknown").upper()
        print(f"[{role}] {msg['content']}")


if __name__ == "__main__":
    main()
