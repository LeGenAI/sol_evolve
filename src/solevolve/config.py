import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()


@dataclass
class Settings:
    openrouter_api_key: str | None
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_search_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    max_turns: int = 6
    artifact_dir: Path = Path("artifacts")
    kissat_path: str = "./kissat/build/kissat"
    cadical_path: str | None = None
    langsmith_project: str = "solevolve-repro"
    trace_tags: tuple[str, ...] = ("revision", "reproducibility", "github")
    paper_claim_id: str | None = None


def _split_tags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("revision", "reproducibility", "github")
    return tuple(tag.strip() for tag in value.split(",") if tag.strip())


def load_settings(*, require_api_key: bool = True) -> Settings:
    """Load settings from environment variables."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if require_api_key and not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        openrouter_search_model=os.getenv("OPENROUTER_SEARCH_MODEL", "openai/gpt-4o-mini"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.1")),
        max_turns=int(os.getenv("SOLEVOLVE_MAX_TURNS", "6")),
        artifact_dir=Path(os.getenv("SOLEVOLVE_ARTIFACT_DIR", "artifacts")),
        kissat_path=os.getenv("KISSAT_PATH", "./kissat/build/kissat"),
        cadical_path=os.getenv("CADICAL_PATH") or None,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "solevolve-repro"),
        trace_tags=_split_tags(os.getenv("SOLEVOLVE_TRACE_TAGS")),
        paper_claim_id=os.getenv("SOLEVOLVE_PAPER_CLAIM_ID") or None,
    )


def solver_path_from_env(solver_type: str, settings: Settings | None = None) -> str | None:
    """Return a configured solver path, if one is available for the solver type."""
    settings = settings or load_settings(require_api_key=False)
    lower = solver_type.lower()
    if lower == "kissat":
        return settings.kissat_path
    if lower == "cadical":
        return settings.cadical_path
    return None
