from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from ..tracing import traceable_run


@dataclass
class Agent:
    """Simple wrapper around a ChatOpenAI-backed agent with a system prompt."""

    name: str
    system_prompt: str
    llm: ChatOpenAI

    def _chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
            ]
        )
        return prompt | self.llm

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        chain = self._chain()
        result: AIMessage = self._invoke_chain(chain, messages)
        content = result.content if result.content else "[empty response]"
        return AIMessage(content=content, name=self.name)

    @traceable_run("solevolve.agent.invoke", run_type="chain")
    def _invoke_chain(self, chain, messages: List[BaseMessage]) -> AIMessage:
        return chain.invoke(
            {"history": messages},
            config={"run_name": self.name, "metadata": {"agent": self.name}},
        )

    def kickoff(self, user_text: str) -> List[BaseMessage]:
        """Start a conversation with a human seed message."""
        return [HumanMessage(content=user_text, name="human")]
