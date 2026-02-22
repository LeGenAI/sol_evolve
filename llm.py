from langchain_openai import ChatOpenAI

from .config import Settings


def build_llm(settings: Settings, *, model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    """Create an OpenRouter-backed ChatOpenAI client."""
    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=model or settings.openrouter_model,
        temperature=settings.temperature if temperature is None else temperature,
    )
