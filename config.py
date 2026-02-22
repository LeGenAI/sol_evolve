import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()


@dataclass
class Settings:
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    # Use a tool-capable, OpenRouter-available model. Adjust if you have access to other tool-call models.
    openrouter_search_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    max_turns: int = 6


def load_settings() -> Settings:
    """Load settings from environment variables."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        openrouter_search_model=os.getenv("OPENROUTER_SEARCH_MODEL", "perplexity/llama-3.1-sonar-small-256k"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.1")),
        max_turns=int(os.getenv("SOLEVOLVE_MAX_TURNS", "6")),
    )
