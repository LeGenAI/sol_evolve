from .base import Agent


REFLECTOR_PROMPT = """You are the Reflector. Interpret verifier/evolver signals and decide the next prompt.
- Detect stagnation, bottlenecks, or missing information.
- Propose targeted instructions for the Generator or decide to terminate when goals are met."""


class ReflectorAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="reflector", system_prompt=REFLECTOR_PROMPT, llm=llm)
