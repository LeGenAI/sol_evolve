from __future__ import annotations

import argparse
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage

from .config import load_settings
from .llm import build_llm
from .tools.web import fetch_url


def main():
    parser = argparse.ArgumentParser(description="Use OpenRouter search model to fetch coding bounds.")
    parser.add_argument(
        "--n",
        type=int,
        default=32,
        help="Code length n (for context in the query).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=14,
        help="Code dimension k (for context in the query).",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=8,
        help="Field size q for the code table (default 8; use 2 for binary).",
    )
    parser.add_argument(
        "--table-url",
        type=str,
        default="https://www.codetables.de/BKLC/Tables.php?q=8&n0=1&n1=130&k0=1&k1=130",
        help="Table URL for fallback scans.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Find the best known minimum distance for the given (n,k) from the provided table and summarize.",
        help="Additional instruction for the agent.",
    )
    args = parser.parse_args()

    settings = load_settings()
    llm = build_llm(settings, model=settings.openrouter_search_model, temperature=0.2)

    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=[fetch_url],
        interrupt_before=[],
        checkpointer=checkpointer,
    )

    thread_id = f"search-{uuid.uuid4()}"
    entry_url = f"https://www.codetables.de/BKLC/BKLC.php?q={args.q}&n={args.n}&k={args.k}"
    goal = (
        f"First fetch the entry page {entry_url} for the exact (n={args.n}, k={args.k}, q={args.q}). "
        f"If distance info is missing, fetch the table page {args.table_url} as fallback. "
        f"{args.query} Extract the minimum distance if present and cite the source snippet."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]}, config={"configurable": {"thread_id": thread_id}})

    for msg in result["messages"]:
        if isinstance(msg, BaseMessage):
            role = getattr(msg, "type", "unknown").upper()
            content = getattr(msg, "content", "")
        elif isinstance(msg, dict):  # fallback for dict-based messages
            role = str(msg.get("role", "unknown")).upper()
            content = msg.get("content", "")
        else:
            role = "UNKNOWN"
            content = str(msg)
        print(f"[{role}] {content}")


if __name__ == "__main__":
    main()
