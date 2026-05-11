from .base import Agent


EVOLVER_PROMPT = """You are the Evolver. Maintain and adapt populations/portfolios.
- Combine generator proposals, portfolio selection, and search operators.
- Surface stagnation signals and resource constraints in crisp bullet points."""


class EvolverAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="evolver", system_prompt=EVOLVER_PROMPT, llm=llm)
