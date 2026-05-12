from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..costing import estimate_llm_cost
from ..tracing import traceable_run, update_current_run_context
from .prompts import COORDINATOR_POLICY

_SECTION_PREVIEW_LIMIT = 1800

def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, BaseModel):
        return content.model_dump_json()
    try:
        return json.dumps(content, sort_keys=True, default=str)
    except Exception:
        return str(content)


def _preview(value: Any, *, limit: int = _SECTION_PREVIEW_LIMIT) -> str:
    text = _content_text(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated: {len(text)} chars]"


def _message_label(message: BaseMessage, index: int) -> str:
    msg_type = getattr(message, "type", message.__class__.__name__)
    name = getattr(message, "name", None)
    return f"{index}. {msg_type}:{name}" if name else f"{index}. {msg_type}"


def _truth_ledger(
    *,
    evidence: list[dict[str, Any]],
    paper_input: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact factual ledger that role prose must not contradict."""
    ledger: dict[str, Any] = {
        "evidence_status": {},
        "solver_check": None,
        "paper_claims": [],
        "missing_obligations": [],
        "lucas_targets": [],
    }
    targets = (paper_input or {}).get("targets", []) if isinstance(paper_input, dict) else []
    for target in targets:
        if not isinstance(target, dict):
            continue
        lucas = target.get("lucas")
        if isinstance(lucas, dict):
            ledger["lucas_targets"].append(
                {
                    "claim_id": target.get("claim_id"),
                    "n": lucas.get("n"),
                    "s": lucas.get("s"),
                    "expected_center_count": lucas.get("expected_center_count"),
                    "expected_vertex_count": lucas.get("expected_vertex_count"),
                    "expected_ball_size_distribution": lucas.get("expected_ball_size_distribution"),
                    "solver_preference": target.get("solver_preference"),
                }
            )

    for item in evidence:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        status = item.get("status")
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        if name:
            ledger["evidence_status"][name] = status
        if name == "solver_check":
            ledger["solver_check"] = {
                "status": status,
                "selected_solver": details.get("selected_solver") or details.get("solver"),
                "selected_path": details.get("selected_resolved_path") or details.get("selected_path"),
                "command": details.get("command"),
                "version": details.get("version"),
                "elapsed_ms": details.get("elapsed_ms"),
                "note": "Availability check only; solver may be irrelevant for targets with solver_preference=none.",
            }
        if name.startswith("paper_claim_"):
            missing = list(details.get("missing_obligations") or [])
            ledger["missing_obligations"].extend(f"{name}:{obligation}" for obligation in missing)
            claim = {
                "evidence_name": name,
                "status": status,
                "result_id": details.get("result_id"),
                "claim_family": details.get("claim_family"),
                "verdict": details.get("verdict"),
                "parameters": details.get("parameters"),
                "missing_obligations": missing,
                "report_path": details.get("report_path"),
            }
            if details.get("claim_family") == "lucas_cubes":
                claim["lucas"] = {
                    "n": (details.get("parameters") or {}).get("n") if isinstance(details.get("parameters"), dict) else None,
                    "s": (details.get("parameters") or {}).get("s") if isinstance(details.get("parameters"), dict) else None,
                    "expected_center_count": details.get("lucas_expected_center_count"),
                    "expected_vertex_count": details.get("lucas_expected_vertex_count"),
                    "center_count": details.get("lucas_center_count"),
                    "vertex_count": details.get("lucas_vertex_count"),
                    "exact_cover": details.get("lucas_exact_cover"),
                    "minimum_pairwise_distance": details.get("lucas_minimum_pairwise_distance"),
                    "ball_size_distribution": details.get("lucas_ball_size_distribution"),
                }
            ledger["paper_claims"].append(claim)

    if ledger["lucas_targets"]:
        ledger["domain_glossary"] = {
            "Lucas n": "graph length in Lambda_n(1^s)",
            "Lucas s": "forbidden circular run length, not a center count",
            "expected_center_count": "number of centers required for a perfect partition",
            "Lucas PASS": "requires a public center-set artifact and all deterministic verifier checks",
        }
    ledger["missing_obligations"] = list(dict.fromkeys(ledger["missing_obligations"]))
    return {key: value for key, value in ledger.items() if value not in (None, [], {})}


@dataclass
class Agent:
    """Simple wrapper around a ChatOpenAI-backed agent with a system prompt."""

    name: str
    system_prompt: str
    llm: ChatOpenAI

    def _chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{handoff}"),
            ]
        )
        return prompt | self.llm

    def invoke(
        self,
        messages: List[BaseMessage],
        *,
        goal: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        next_instruction: str | None = None,
        paper_input: dict[str, Any] | None = None,
        paper_output: dict[str, Any] | None = None,
        reflector_action: dict[str, Any] | None = None,
    ) -> AIMessage:
        handoff = self._build_handoff(
            messages,
            goal=goal,
            evidence=evidence or [],
            next_instruction=next_instruction,
            paper_input=paper_input,
            paper_output=paper_output,
            reflector_action=reflector_action,
        )
        result: AIMessage = self._invoke_chain(handoff)
        content = result.content if result.content else "[empty response]"
        response_metadata = getattr(result, "response_metadata", {}) or {}
        usage = getattr(result, "usage_metadata", None)
        kwargs: dict[str, Any] = {
            "content": content,
            "name": self.name,
            "response_metadata": {"solevolve_trace": response_metadata.get("solevolve_trace", {})},
        }
        if usage:
            kwargs["usage_metadata"] = usage
        return AIMessage(**kwargs)

    def _build_handoff(
        self,
        messages: List[BaseMessage],
        *,
        goal: str | None,
        evidence: list[dict[str, Any]],
        next_instruction: str | None,
        paper_input: dict[str, Any] | None,
        paper_output: dict[str, Any] | None,
        reflector_action: dict[str, Any] | None,
    ) -> str:
        inferred_goal = goal
        if inferred_goal is None and messages:
            inferred_goal = _content_text(messages[0].content)

        sections = [
            "You are receiving a provider-compatible handoff as a single user message.",
            "",
            COORDINATOR_POLICY,
            "",
            "Evidence Truth Ledger (authoritative; prefer these facts over prior role prose):",
            _preview(_truth_ledger(evidence=evidence, paper_input=paper_input), limit=2600),
            "",
            "Original goal:",
            _preview(inferred_goal or "[not provided]"),
            "",
            "Paper workflow input:",
            _preview(paper_input or "[not resolved yet]", limit=1600),
            "",
            "Deterministic repro evidence:",
        ]
        if evidence:
            for item in evidence:
                sections.append(_preview(item, limit=1200))
        else:
            sections.append("[no deterministic evidence recorded]")

        if next_instruction:
            sections.extend(["", "Current reflector instruction:", _preview(next_instruction)])
        if reflector_action:
            sections.extend(["", "Current paper reflector action:", _preview(reflector_action, limit=1200)])
        if paper_output:
            sections.extend(["", "Current paper workflow output:", _preview(paper_output, limit=1600)])

        sections.extend(["", "Prior role outputs:"])
        if messages:
            for index, message in enumerate(messages, start=1):
                sections.append(f"{_message_label(message, index)}\n{_preview(message.content)}")
        else:
            sections.append("[no prior messages]")

        sections.extend(
            [
                "",
                "Task:",
                "Use only the goal, deterministic evidence, and prior role outputs above. "
                "Do not claim that an experiment, solver, matrix, or raw artifact was verified unless the evidence says so. "
                "If prior role prose conflicts with the Evidence Truth Ledger, treat the ledger as the correction.",
            ]
        )
        return "\n".join(sections)

    @traceable_run("solevolve.agent.invoke", run_type="chain")
    def _invoke_chain(self, handoff: str) -> AIMessage:
        started = time.perf_counter()
        result = self.llm.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=handoff)],
            config={"run_name": f"{self.name}.llm", "metadata": {"agent": self.name}},
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        usage = getattr(result, "usage_metadata", None)
        response_metadata = getattr(result, "response_metadata", {}) or {}
        model_name = response_metadata.get("model_name") or response_metadata.get("model")
        cost = estimate_llm_cost(
            usage,
            model=model_name,
            base_url=getattr(self.llm, "openai_api_base", None) or getattr(self.llm, "base_url", None),
        )
        metadata = {
            "agent": self.name,
            "llm_elapsed_ms": elapsed_ms,
            "llm_usage": usage,
            "llm_model_name": model_name,
            "llm_finish_reason": response_metadata.get("finish_reason"),
            "llm_cost": cost,
            "llm_estimated_cost_usd": cost.get("estimated_cost_usd"),
        }
        update_current_run_context(
            metadata=metadata
        )
        updated_response_metadata = dict(response_metadata)
        updated_response_metadata["solevolve_trace"] = metadata
        if hasattr(result, "model_copy"):
            result = result.model_copy(update={"response_metadata": updated_response_metadata})
        else:
            result = result.copy(update={"response_metadata": updated_response_metadata})
        return result

    def kickoff(self, user_text: str) -> List[BaseMessage]:
        """Start a conversation with a human seed message."""
        return [HumanMessage(content=user_text, name="human")]
