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
    solver_preference: str = "cadical"
    langsmith_project: str = "solevolve-repro"
    trace_tags: tuple[str, ...] = ("revision", "reproducibility", "github")
    paper_claim_id: str | None = None
    paper_claim_timeout: int = 300
    hybrid_ga_mode: str = "off"
    hybrid_ga_seed: int = 0
    hybrid_ga_population: int = 100
    hybrid_ga_generations: int = 100
    hybrid_ga_repair_interval: int = 50
    hybrid_ga_timeout_sec: int = 300
    hybrid_ga_target_distance: int = 7
    hybrid_ga_frontier_distance: int | None = None
    hybrid_ga_frontier_seed_count: int = 20
    hybrid_ga_frontier_target_timeout_sec: int = 60
    hybrid_ga_frontier_seed_timeout_sec: int = 30
    hybrid_ga_repair_strategy: str = "low_weight_support_mask"


def _split_tags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ("revision", "reproducibility", "github")
    return tuple(tag.strip() for tag in value.split(",") if tag.strip())


def _solver_preference(value: str | None) -> str:
    normalized = (value or "cadical").lower()
    return normalized if normalized in {"cadical", "kissat"} else "cadical"


def _hybrid_ga_mode(value: str | None) -> str:
    normalized = (value or "off").lower()
    return normalized if normalized in {"off", "archived", "replay", "live", "frontier_repair"} else "off"


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
        solver_preference=_solver_preference(os.getenv("SOLEVOLVE_SOLVER")),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "solevolve-repro"),
        trace_tags=_split_tags(os.getenv("SOLEVOLVE_TRACE_TAGS")),
        paper_claim_id=os.getenv("SOLEVOLVE_PAPER_CLAIM_ID") or None,
        paper_claim_timeout=int(os.getenv("SOLEVOLVE_PAPER_CLAIM_TIMEOUT", "300")),
        hybrid_ga_mode=_hybrid_ga_mode(os.getenv("SOLEVOLVE_HYBRID_GA_MODE")),
        hybrid_ga_seed=int(os.getenv("SOLEVOLVE_HYBRID_GA_SEED", "0")),
        hybrid_ga_population=int(os.getenv("SOLEVOLVE_HYBRID_GA_POPULATION", "100")),
        hybrid_ga_generations=int(os.getenv("SOLEVOLVE_HYBRID_GA_GENERATIONS", "100")),
        hybrid_ga_repair_interval=int(os.getenv("SOLEVOLVE_HYBRID_GA_REPAIR_INTERVAL", "50")),
        hybrid_ga_timeout_sec=int(os.getenv("SOLEVOLVE_HYBRID_GA_TIMEOUT_SEC", "300")),
        hybrid_ga_target_distance=int(os.getenv("SOLEVOLVE_HYBRID_GA_TARGET_DISTANCE", "7")),
        hybrid_ga_frontier_distance=(
            int(os.environ["SOLEVOLVE_HYBRID_GA_FRONTIER_DISTANCE"])
            if os.getenv("SOLEVOLVE_HYBRID_GA_FRONTIER_DISTANCE")
            else None
        ),
        hybrid_ga_frontier_seed_count=int(os.getenv("SOLEVOLVE_HYBRID_GA_FRONTIER_SEED_COUNT", "20")),
        hybrid_ga_frontier_target_timeout_sec=int(os.getenv("SOLEVOLVE_HYBRID_GA_FRONTIER_TARGET_TIMEOUT_SEC", "60")),
        hybrid_ga_frontier_seed_timeout_sec=int(os.getenv("SOLEVOLVE_HYBRID_GA_FRONTIER_SEED_TIMEOUT_SEC", "30")),
        hybrid_ga_repair_strategy=os.getenv("SOLEVOLVE_HYBRID_GA_REPAIR_STRATEGY", "low_weight_support_mask"),
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
