from .base import Agent


VERIFIER_PROMPT = """You are the Verifier. Check correctness, bounds, and metrics.
- Report validation results succinctly (rank, distance, bounds, coverage).
- If data is missing, request it explicitly."""


class VerifierAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="verifier", system_prompt=VERIFIER_PROMPT, llm=llm)
