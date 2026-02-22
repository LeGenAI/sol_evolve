from __future__ import annotations

from typing import List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from .agents import EvolverAgent, GeneratorAgent, ReflectorAgent, VerifierAgent


class AgentState(TypedDict):
    messages: List[BaseMessage]
    turn: int


def _agent_node(agent):
    def node(state: AgentState) -> AgentState:
        reply = agent.invoke(state["messages"])
        return {"messages": state["messages"] + [reply], "turn": state["turn"] + 1}

    return node


def _route_after_reflection(max_turns: int):
    def route(state: AgentState) -> Literal["generator", "end"]:
        if state["turn"] >= max_turns:
            return "end"
        return "generator"

    return route


def build_sol_evolve_network(
    *,
    generator: GeneratorAgent,
    evolver: EvolverAgent,
    verifier: VerifierAgent,
    reflector: ReflectorAgent,
    max_turns: int = 6,
):
    """Construct a LangGraph Deep Agents network for SolEvolve roles."""
    graph = StateGraph(AgentState)

    graph.add_node("generator", _agent_node(generator))
    graph.add_node("evolver", _agent_node(evolver))
    graph.add_node("verifier", _agent_node(verifier))
    graph.add_node("reflector", _agent_node(reflector))

    graph.add_edge("generator", "evolver")
    graph.add_edge("evolver", "verifier")
    graph.add_edge("verifier", "reflector")
    graph.add_conditional_edges(
        "reflector",
        _route_after_reflection(max_turns),
        {
            "generator": "generator",
            "end": END,
        },
    )

    graph.set_entry_point("generator")
    return graph.compile()
