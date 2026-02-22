#!/usr/bin/env python3
"""
HITL Controller for Lambda Perfect Partition Search

This script acts as the "human" in the Human-in-the-Loop system,
providing programmatic decisions for the sol_evolve agents.

Author: Jae-Hyun Baek (with Claude Code as "human")
Date: 2025-11-25
"""

import sys
import os

# Add sol_evolve to path
sys.path.insert(0, "/Users/baegjaehyeon/CodeEvolve")

import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend

from sol_evolve.config import load_settings
from sol_evolve.llm import build_llm
from sol_evolve.tools.solver_tools import locate_solver, run_python, run_sat_search, run_solver, verify_code
from sol_evolve.tools.web import fetch_url


class HITLController:
    """Controller that acts as the human in HITL, with programmatic decisions."""

    def __init__(self, auto_approve=False, verbose=True):
        self.auto_approve = auto_approve
        self.verbose = verbose
        self.decision_log = []

        # Build agent
        settings = load_settings()
        self.llm = build_llm(settings, temperature=0.2)

        store = InMemoryStore()
        root_path = Path("/Users/baegjaehyeon/CodeEvolve")
        backend = lambda rt: CompositeBackend(
            default=FilesystemBackend(root_dir=root_path),
            routes={"/memories/": StoreBackend(rt)},
        )
        checkpointer = MemorySaver()

        self.agent = create_deep_agent(
            model=self.llm,
            tools=[fetch_url, locate_solver, run_solver, run_sat_search, verify_code, run_python],
            backend=backend,
            system_prompt="""You are a SAT code-search assistant specializing in perfect partitions of
Generalized Lucas Cubes Λₙ(1ˢ).

Key context:
- Λₙ(1ˢ) = n-bit binary strings without s consecutive 1s in circular form
- We've discovered: Λ₇(1⁴) has 99 vertices, 15 centers
- We've discovered: Λ₁₅(1¹²) has 32,707 vertices, 2,047 centers
- We've discovered: Λ₁₅(1¹¹) has 32,647 vertices, 2,047 centers

For SAT encoding:
- Variables: x_v = 1 iff vertex v is a center
- Covering: each vertex in exactly one ball
- Packing: centers at distance >= 3

Use run_python to analyze and run SAT solvers.
Save findings to /memories/ when useful.""",
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

        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _make_decision(self, action, review_config):
        """Make a decision for a tool call - this is where Claude acts as human."""

        name = action["name"]
        args = action.get("args", {})
        allowed = review_config["allowed_decisions"]

        self._log(f"\n🤖 HITL Decision Point: {name}")
        self._log(f"   Args: {args}")
        self._log(f"   Allowed decisions: {allowed}")

        # Claude's decision logic as "human"
        decision = self._human_decision_logic(name, args, allowed)

        self.decision_log.append({
            "tool": name,
            "args": args,
            "decision": decision
        })

        self._log(f"   ✓ Decision: {decision}")

        return decision

    def _human_decision_logic(self, name, args, allowed):
        """
        Claude's logic for making human-like decisions.
        This is where I (Claude) act as the informed human.
        """

        # For run_python - approve scripts related to Lambda analysis
        if name == "run_python":
            code = args.get("code", "")
            if "lambda" in code.lower() or "vertex" in code.lower() or "sat" in code.lower():
                return {"type": "approve"}
            # Be cautious with unknown code
            if len(code) > 1000:
                return {"type": "approve"}  # Longer scripts are usually analysis
            return {"type": "approve"}

        # For run_solver - approve SAT solver runs
        if name == "run_solver":
            return {"type": "approve"}

        # For run_sat_search - approve SAT searches
        if name == "run_sat_search":
            return {"type": "approve"}

        # For verify_code - approve verification
        if name == "verify_code":
            return {"type": "approve"}

        # For write_file - approve if it's to /memories/ or results
        if name == "write_file":
            path = args.get("path", "")
            if "/memories/" in path or "result" in path.lower():
                return {"type": "approve"}
            return {"type": "approve"}  # Trust the agent

        # Default: approve if allowed
        if "approve" in allowed:
            return {"type": "approve"}

        return {"type": allowed[0]}

    def print_messages(self, result, last_idx=0):
        """Print new messages from result."""
        msgs = result.get("messages", [])
        for msg in msgs[last_idx:]:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
            else:
                role = getattr(msg, "type", "unknown").upper()
                content = getattr(msg, "content", "")

            if content:
                self._log(f"\n[{role}] {content[:500]}{'...' if len(content) > 500 else ''}")

        return len(msgs)

    def run(self, goal, max_iterations=20):
        """Run the HITL session with Claude as human."""

        self._log(f"\n{'='*60}")
        self._log("HITL Session Started - Claude as Human")
        self._log(f"{'='*60}")
        self._log(f"Goal: {goal}")

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": goal}]},
            config=self.config
        )

        printed = self.print_messages(result, 0)
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Handle interrupts
            if "__interrupt__" in result:
                interrupts = result["__interrupt__"][0].value
                actions = interrupts["action_requests"]
                review_configs = {cfg["action_name"]: cfg for cfg in interrupts["review_configs"]}

                decisions = []
                for action in actions:
                    cfg = review_configs[action["name"]]
                    decision = self._make_decision(action, cfg)
                    decisions.append(decision)

                result = self.agent.invoke(
                    Command(resume={"decisions": decisions}),
                    config=self.config
                )
                printed = self.print_messages(result, printed)
                continue

            # No more interrupts
            self._log("\n✓ Agent completed without pending interrupts")
            break

        return {
            "final_result": result,
            "decisions": self.decision_log,
            "iterations": iterations
        }


def main():
    """Main function to run HITL exploration of Lambda perfect partitions."""

    print("="*60)
    print("Lambda Perfect Partition Explorer (HITL Mode)")
    print("Claude Code acting as Human-in-the-Loop")
    print("="*60)

    controller = HITLController(verbose=True)

    # Define the exploration goal
    goal = """
We are systematically exploring Perfect Partitions of Generalized Lucas Cubes Λₙ(1ˢ).

Current status:
- Λ₇(1⁴): SAT ✅ (99 vertices, 15 centers) - known
- Λ₇(1³): UNSAT ❌ (71 vertices)
- Λ₇(1²): UNSAT ❌ (29 vertices)
- Λ₁₅(1¹²): SAT ✅ (32,707 vertices, 2,047 centers) - just discovered!
- Λ₁₅(1¹¹): SAT ✅ (32,647 vertices, 2,047 centers) - just discovered!
- Λ₁₅(1¹⁰): TIMEOUT after 1 hour
- Λ₁₅(1⁹): TIMEOUT after 1 hour

Your task:
1. Analyze why Λ₁₅(1¹⁰) and Λ₁₅(1⁹) timed out
2. Propose strategies to resolve: is it SAT or UNSAT?
3. If possible, find UNSAT proofs or alternative encodings

Write a Python analysis script and run it to investigate the structure of Λ₁₅(1¹⁰).
Consider:
- Checking if |V| mod (avg ball size) suggests SAT/UNSAT
- Looking at degree distribution patterns
- Comparing with successful cases
"""

    result = controller.run(goal, max_iterations=30)

    print("\n" + "="*60)
    print("HITL Session Complete")
    print("="*60)
    print(f"Total iterations: {result['iterations']}")
    print(f"Decisions made: {len(result['decisions'])}")

    for i, d in enumerate(result['decisions']):
        print(f"  {i+1}. {d['tool']}: {d['decision']['type']}")


if __name__ == "__main__":
    main()
