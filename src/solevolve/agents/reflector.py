from .base import Agent
from .prompts import REFLECTOR_PROMPT


class ReflectorAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="reflector", system_prompt=REFLECTOR_PROMPT, llm=llm)
