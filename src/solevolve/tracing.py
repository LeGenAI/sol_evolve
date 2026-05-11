from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from .config import Settings

P = ParamSpec("P")
R = TypeVar("R")


def tracing_enabled() -> bool:
    flag = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes", "on"}
    return flag and bool(os.getenv("LANGSMITH_API_KEY"))


def traceable_run(name: str, *, run_type: str = "chain") -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Use LangSmith tracing when configured; otherwise return a no-op wrapper."""

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        if tracing_enabled():
            try:
                from langsmith import traceable

                return traceable(name=name, run_type=run_type)(func)
            except Exception:
                pass

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return wrapper

    return decorate


def git_sha(root: Path | None = None) -> str | None:
    cwd = root or Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def base_metadata(settings: Settings, **extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "artifact_dir": str(settings.artifact_dir),
        "git_sha": git_sha(),
        "paper_claim_id": settings.paper_claim_id,
        "tags": list(settings.trace_tags),
    }
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@traceable_run("solevolve.artifact_write", run_type="tool")
def write_artifact_summary(artifact_dir: Path, payload: dict[str, Any]) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "last_run_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
