from .base import Agent


GENERATOR_PROMPT = """You are the Generator in the SolEvolve architecture.
- Goal: propose executable algorithmic components (encoders, mutations, SAT/ILP scripts).
- Format ideas clearly and be concise.
- If you need data from other agents, state it explicitly."""


class GeneratorAgent(Agent):
    def __init__(self, llm):
        super().__init__(name="generator", system_prompt=GENERATOR_PROMPT, llm=llm)
