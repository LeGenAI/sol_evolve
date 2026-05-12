from .base import Agent
from .prompts import VERIFIER_PROMPT


class VerifierAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="verifier", system_prompt=VERIFIER_PROMPT, llm=llm)
