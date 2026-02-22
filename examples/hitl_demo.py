from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend

from .config import load_settings
from .llm import build_llm
from .tools.solver_tools import locate_solver, run_python, run_sat_search, run_solver, verify_code
from .tools.web import fetch_url


def _build_agent(llm):
    store = InMemoryStore()
    root_path = Path(__file__).resolve().parents[1]
    backend = lambda rt: CompositeBackend(
        default=FilesystemBackend(root_dir=root_path),
        routes={"/memories/": StoreBackend(rt)},
    )
    checkpointer = MemorySaver()
    return create_deep_agent(
        model=llm,
        tools=[fetch_url, locate_solver, run_solver, run_sat_search, verify_code, run_python],
        backend=backend,
        system_prompt=(
            "You are a SAT code-search assistant. Start with quick web lookup for bounds "
            "using fetch_url (codetables URLs) to anchor expectations. Then plan and run solver "
            "actions with human approval. If asked to run local analysis scripts, use run_python "
            "instead of saying you cannot execute code. Save concise notes to /memories/ when useful."
        ),
        interrupt_on={
            "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            "run_solver": True,
            "run_sat_search": True,
            "verify_code": True,
            "run_python": True,
        },
        checkpointer=checkpointer,
        store=store,
    )


def main():
    parser = argparse.ArgumentParser(description="SolEvolve human-in-the-loop demo (Deep Agents).")
    parser.add_argument(
        "goal",
        type=str,
        nargs="?",
        default="Propose a SAT plan, locate solver, run it, and summarize results including a /memories/ note.",
        help="Initial user request.",
    )
    parser.add_argument("--thread-id", type=str, default=None, help="Optional thread id for resuming.")
    # Optional direct SAT run (manual approval)
    parser.add_argument("--n", type=int, help="Code length n for direct SAT run")
    parser.add_argument("--k", type=int, help="Code dimension k for direct SAT run")
    parser.add_argument("--d", type=int, help="Target minimum distance for direct SAT run")
    parser.add_argument("--num-solutions", type=int, default=1, help="Solutions to search (default 1)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per instance (seconds)")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for SAT run")
    parser.add_argument("--solver", type=str, default="cadical", help="Solver type (cadical|kissat|hcadsbva|...)")
    parser.add_argument("--solver-path", type=str, default=None, help="Explicit solver binary path")
    parser.add_argument("--auto-approve", action="store_true", help="Skip prompt and auto-approve direct run")
    parser.add_argument(
        "--auto-approve-agent",
        action="store_true",
        help="Auto-approve all agent tool calls (HITL off). Use with care.",
    )
    args = parser.parse_args()

    settings = load_settings()
    llm = build_llm(settings, temperature=0.2)

    # Direct SAT run path with manual approval (no agent)
    if args.n is not None and args.k is not None and args.d is not None:
        print(
            f"Requested direct SAT run: n={args.n}, k={args.k}, d={args.d}, "
            f"num_solutions={args.num_solutions}, timeout={args.timeout}s, solver={args.solver}"
        )
        try:
            approve = "y" if args.auto_approve else input("Approve running run_solver? [y/N]: ").strip().lower()
        except EOFError:
            approve = "y" if args.auto_approve else "n"
        if approve != "y":
            print("Run cancelled.")
            return
        print("Running run_solver...")
        summary = run_solver.invoke(
            {
                "n": args.n,
                "k": args.k,
                "d": args.d,
                "num_solutions": args.num_solutions,
                "timeout": args.timeout,
                "workdir": args.workdir,
                "solver": args.solver,
                "solver_path": args.solver_path,
            }
        )
        print(summary)
        return

    agent = _build_agent(llm)
    config = {"configurable": {"thread_id": args.thread_id or str(uuid.uuid4())}}

    def print_new_messages(result, last_idx):
        msgs = result.get("messages", [])
        for msg in msgs[last_idx:]:
            role = msg.get("role", "unknown").upper() if isinstance(msg, dict) else getattr(msg, "type", "unknown").upper()
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            print(f"[{role}] {content}")
        return len(msgs)

    result = agent.invoke({"messages": [{"role": "user", "content": args.goal}]}, config=config)
    printed = print_new_messages(result, 0)

    while True:
        # Handle any pending interrupts
        if "__interrupt__" in result:
            interrupts = result["__interrupt__"][0].value
            actions = interrupts["action_requests"]
            review_configs = {cfg["action_name"]: cfg for cfg in interrupts["review_configs"]}

            decisions = []
            for action in actions:
                cfg = review_configs[action["name"]]
                allowed = ", ".join(cfg["allowed_decisions"])
                print(f"\nTool call pending: {action['name']} {action['args']} (allowed: {allowed})")
                if args.auto_approve_agent and "approve" in cfg["allowed_decisions"]:
                    choice = "approve"
                else:
                    choice = ""
                    while choice not in cfg["allowed_decisions"]:
                        try:
                            choice = input(f"Decision for {action['name']} [{allowed}]: ").strip()
                        except EOFError:
                            choice = "approve" if "approve" in cfg["allowed_decisions"] else cfg["allowed_decisions"][0]
                if choice == "edit":
                    edited_args = {**action.get("args", {})}
                    for key in list(edited_args.keys()):
                        new_val = input(f"New value for {key} (blank to keep '{edited_args[key]}'): ").strip()
                        if new_val:
                            edited_args[key] = new_val
                    decisions.append({"type": "edit", "edited_action": {"name": action["name"], "args": edited_args}})
                elif choice == "approve":
                    decisions.append({"type": "approve"})
                elif choice == "reject":
                    decisions.append({"type": "reject"})
                else:
                    raise ValueError(f"Unsupported decision: {choice}")

            extra_msg = ""
            if not args.auto_approve_agent:
                try:
                    extra_msg = input("Additional instruction to the agent before resuming (blank to skip): ").strip()
                except EOFError:
                    extra_msg = ""

            result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
            printed = print_new_messages(result, printed)

            if extra_msg:
                result = agent.invoke({"messages": [{"role": "user", "content": extra_msg}]}, config=config)
                printed = print_new_messages(result, printed)
            continue

        # No interrupts: allow user to converse or exit
        if args.auto_approve_agent:
            break
        else:
            try:
                user_msg = input("Send a message to the agent (blank to exit): ").strip()
            except EOFError:
                user_msg = ""
            if not user_msg:
                break
            result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]}, config=config)
            printed = print_new_messages(result, printed)


if __name__ == "__main__":
    main()
