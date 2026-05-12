from .base import Agent
from .prompts import EVOLVER_PROMPT


class EvolverAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="evolver", system_prompt=EVOLVER_PROMPT, llm=llm)
