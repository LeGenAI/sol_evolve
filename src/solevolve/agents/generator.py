from .base import Agent
from .prompts import GENERATOR_PROMPT


class GeneratorAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="generator", system_prompt=GENERATOR_PROMPT, llm=llm)
