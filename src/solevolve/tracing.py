from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import re
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from pydantic import BaseModel

from .config import Settings

P = ParamSpec("P")
R = TypeVar("R")

REDACTED = "[REDACTED]"
_PREVIEW_LIMIT = 4000
_MAX_DEPTH = 8
_MAX_COLLECTION_ITEMS = 50
_TRUTHY = {"1", "true", "yes", "on"}
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "langsmith_api_key",
    "openrouter_api_key",
    "password",
    "secret",
    "token",
}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"lsv2_[A-Za-z0-9_=-]+"),
    re.compile(r"sk-or-[A-Za-z0-9_-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
)

_LANGSMITH_CLIENT: Any | None = None
_WARNED_NOOP: set[str] = set()


def tracing_enabled() -> bool:
    return _tracing_requested() and bool(os.getenv("LANGSMITH_API_KEY"))


def strict_tracing_enabled() -> bool:
    return os.getenv("SOLEVOLVE_STRICT_TRACING", "").lower() in _TRUTHY


def _tracing_requested() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in _TRUTHY


def _resolved_project() -> str:
    return os.getenv("LANGSMITH_PROJECT") or "solevolve-repro"


def _disabled_reason() -> str | None:
    if not _tracing_requested():
        return "LANGSMITH_TRACING is not enabled"
    if not os.getenv("LANGSMITH_API_KEY"):
        return "LANGSMITH_API_KEY is not set"
    if not _resolved_project():
        return "LANGSMITH_PROJECT could not be resolved"
    return None


def _warn_noop(name: str, reason: str) -> None:
    if name in _WARNED_NOOP:
        return
    _WARNED_NOOP.add(name)
    warnings.warn(
        f"LangSmith tracing disabled for {name}: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )


def _import_langsmith() -> tuple[Any, Any]:
    from langsmith import Client, traceable

    return Client, traceable


def _get_langsmith_client(client_cls: Any | None = None) -> Any:
    global _LANGSMITH_CLIENT

    if _LANGSMITH_CLIENT is not None:
        return _LANGSMITH_CLIENT

    if client_cls is None:
        client_cls, _ = _import_langsmith()

    try:
        _LANGSMITH_CLIENT = client_cls(
            hide_inputs=sanitize_trace_payload,
            hide_outputs=sanitize_trace_payload,
            hide_metadata=sanitize_trace_metadata,
        )
    except TypeError:
        _LANGSMITH_CLIENT = client_cls(
            hide_inputs=sanitize_trace_payload,
            hide_outputs=sanitize_trace_payload,
        )
    return _LANGSMITH_CLIENT


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_token")
        or "secret" in normalized
        or "password" in normalized
    )


def _is_large_payload_key(key: Any, value: Any = None) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    if normalized == "model" and isinstance(value, str):
        return len(value) > _PREVIEW_LIMIT
    return normalized in {
        "model",
        "model_true_vars",
        "sat_model",
        "solver_model",
        "generator_matrix",
        "matrix_payload",
        "cnf_payload",
    }


def _summarize_large_payload(value: Any) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        encoded = str(value)
    result = {
        "type": _safe_type(value),
        "length": len(value) if hasattr(value, "__len__") else None,
        "sha256": _sha256_text(encoded),
        "redacted": "large_payload",
    }
    if isinstance(value, dict):
        result["keys_preview"] = [str(key) for key in list(value.keys())[:20]]
    return result


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_type(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _preview_string(value: str, *, limit: int = _PREVIEW_LIMIT) -> str | dict[str, Any]:
    redacted = _redact_string(value)
    if len(redacted) <= limit:
        return redacted
    return {
        "type": "str",
        "preview": redacted[:limit],
        "length": len(value),
        "sha256": _sha256_text(redacted),
        "truncated": True,
    }


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True, default=str)
    except Exception:
        return str(content)


def _is_message_like(value: Any) -> bool:
    return hasattr(value, "content") and (
        hasattr(value, "type") or value.__class__.__name__.endswith("Message")
    )


def _summarize_message(message: Any) -> dict[str, Any]:
    content = _redact_string(_message_content_text(getattr(message, "content", "")))
    summary: dict[str, Any] = {
        "type": getattr(message, "type", message.__class__.__name__),
        "content_preview": content[:_PREVIEW_LIMIT],
        "content_sha256": _sha256_text(content),
        "content_length": len(content),
    }
    name = getattr(message, "name", None)
    if name is not None:
        summary["name"] = str(name)
    return summary


def _summarize_settings(settings: Settings) -> dict[str, Any]:
    return {
        "type": _safe_type(settings),
        "artifact_dir": str(settings.artifact_dir),
        "solver_preference": settings.solver_preference,
        "paper_claim_id": settings.paper_claim_id,
        "redacted": "Settings details omitted from trace payload; use run metadata for non-secret run configuration.",
    }


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _looks_like_client_or_chain(value: Any) -> bool:
    type_name = _safe_type(value).lower()
    return any(
        marker in type_name
        for marker in (
            "chatopenai",
            "client",
            "langchain",
            "langsmith",
            "openai",
            "runnable",
            "sequence",
        )
    )


def _summarize_pydantic_model(value: BaseModel, *, _depth: int, _seen: set[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"type": _safe_type(value)}
    try:
        dumped = value.model_dump(mode="python")
    except Exception:
        return result
    for key, item in list(dumped.items())[:_MAX_COLLECTION_ITEMS]:
        result[str(key)] = sanitize_trace_payload(
            item,
            _depth=_depth + 1,
            _key=key,
            _seen=set(_seen),
        )
    if len(dumped) > _MAX_COLLECTION_ITEMS:
        result["_truncated_items"] = len(dumped) - _MAX_COLLECTION_ITEMS
    return result


def sanitize_trace_payload(value: Any, *, _depth: int = 0, _key: Any = None, _seen: set[int] | None = None) -> Any:
    if _is_sensitive_key(_key):
        return REDACTED
    if _is_large_payload_key(_key, value):
        return _summarize_large_payload(value)

    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _preview_string(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if _depth >= _MAX_DEPTH:
        return {"type": _safe_type(value), "truncated": "max_depth"}

    obj_id = id(value)
    if isinstance(value, (dict, list, tuple, set)) or dataclasses.is_dataclass(value) or hasattr(value, "__dict__"):
        if obj_id in _seen:
            return {"type": _safe_type(value), "circular": True}
        _seen.add(obj_id)

    if isinstance(value, Settings):
        return _summarize_settings(value)
    if _is_message_like(value):
        return _summarize_message(value)
    if isinstance(value, BaseModel):
        return _summarize_pydantic_model(value, _depth=_depth, _seen=_seen)
    if _is_array_like(value):
        return {
            "type": _safe_type(value),
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "")),
        }
    if isinstance(value, BaseException):
        return {
            "type": _safe_type(value),
            "message": _preview_string(str(value)),
        }
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_MAX_COLLECTION_ITEMS]:
            key_text = str(key)
            result[key_text] = sanitize_trace_payload(item, _depth=_depth + 1, _key=key_text, _seen=_seen)
        if len(items) > _MAX_COLLECTION_ITEMS:
            result["_truncated_items"] = len(items) - _MAX_COLLECTION_ITEMS
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        result = [
            sanitize_trace_payload(item, _depth=_depth + 1, _seen=_seen)
            for item in items[:_MAX_COLLECTION_ITEMS]
        ]
        if len(items) > _MAX_COLLECTION_ITEMS:
            result.append({"_truncated_items": len(items) - _MAX_COLLECTION_ITEMS})
        return result
    if dataclasses.is_dataclass(value):
        result = {"type": _safe_type(value)}
        for field in dataclasses.fields(value):
            result[field.name] = sanitize_trace_payload(
                getattr(value, field.name),
                _depth=_depth + 1,
                _key=field.name,
                _seen=_seen,
            )
        return result
    if _looks_like_client_or_chain(value):
        return {"type": _safe_type(value)}

    return {"type": _safe_type(value)}


def sanitize_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_trace_payload(inputs)
    return sanitized if isinstance(sanitized, dict) else {"inputs": sanitized}


def sanitize_trace_outputs(output: Any) -> dict[str, Any]:
    sanitized = sanitize_trace_payload(output)
    return sanitized if isinstance(sanitized, dict) else {"output": sanitized}


def sanitize_trace_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized = sanitize_trace_payload(metadata)
    return sanitized if isinstance(sanitized, dict) else {"metadata": sanitized}


def traceable_run(
    name: str,
    *,
    run_type: str = "chain",
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    process_outputs: Callable[[Any], dict[str, Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Use LangSmith tracing when configured; otherwise return a no-op wrapper."""

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            langsmith_extra = kwargs.pop("langsmith_extra", None)
            strict = strict_tracing_enabled()
            reason = _disabled_reason()
            if reason is not None:
                if strict:
                    raise RuntimeError(f"Strict LangSmith tracing requires {reason}.")
                if _tracing_requested():
                    _warn_noop(name, reason)
                return func(*args, **kwargs)

            try:
                client_cls, traceable = _import_langsmith()
                client = _get_langsmith_client(client_cls)

                @functools.wraps(func)
                def timed_func(*inner_args: P.args, **inner_kwargs: P.kwargs) -> R:
                    started = time.perf_counter()
                    try:
                        return func(*inner_args, **inner_kwargs)
                    finally:
                        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
                        update_current_run_context(
                            metadata={
                                "elapsed_ms": elapsed_ms,
                                "solevolve_elapsed_ms": elapsed_ms,
                            },
                            strict=False,
                        )

                traced = traceable(
                    name=name,
                    run_type=run_type,
                    client=client,
                    project_name=_resolved_project(),
                    process_inputs=process_inputs or sanitize_trace_inputs,
                    process_outputs=process_outputs or sanitize_trace_outputs,
                )(timed_func)
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"Failed to prepare LangSmith tracing for {name}: {exc}") from exc
                _warn_noop(name, str(exc))
                return func(*args, **kwargs)

            if langsmith_extra is None:
                return traced(*args, **kwargs)
            return traced(*args, langsmith_extra=langsmith_extra, **kwargs)

        wrapper.__solevolve_trace_name__ = name  # type: ignore[attr-defined]
        wrapper.__solevolve_trace_run_type__ = run_type  # type: ignore[attr-defined]
        return wrapper

    return decorate


def flush_tracing(*, strict: bool | None = None) -> None:
    strict_mode = strict_tracing_enabled() if strict is None else strict
    reason = _disabled_reason()
    if reason is not None:
        if strict_mode:
            raise RuntimeError(f"Strict LangSmith tracing requires {reason}.")
        return

    if _LANGSMITH_CLIENT is None:
        return

    flush = getattr(_LANGSMITH_CLIENT, "flush", None)
    if not callable(flush):
        return

    try:
        flush()
    except Exception as exc:
        if strict_mode:
            raise RuntimeError(f"Failed to flush LangSmith traces: {exc}") from exc
        warnings.warn(f"Failed to flush LangSmith traces: {exc}", RuntimeWarning, stacklevel=2)


def update_current_run_context(
    *,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    strict: bool | None = None,
) -> None:
    """Attach final metadata/tags to the currently active LangSmith run."""
    if not metadata and not tags:
        return

    strict_mode = strict_tracing_enabled() if strict is None else strict
    reason = _disabled_reason()
    if reason is not None:
        if strict_mode:
            raise RuntimeError(f"Strict LangSmith tracing requires {reason}.")
        return

    try:
        from langsmith.run_helpers import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree is None:
            return
        if metadata:
            run_tree.add_metadata(sanitize_trace_metadata(metadata))
        if tags:
            run_tree.add_tags([_redact_string(str(tag)) for tag in tags])
    except Exception as exc:
        if strict_mode:
            raise RuntimeError(f"Failed to update current LangSmith run context: {exc}") from exc
        warnings.warn(f"Failed to update current LangSmith run context: {exc}", RuntimeWarning, stacklevel=2)


def _reset_tracing_state_for_tests() -> None:
    global _LANGSMITH_CLIENT

    _LANGSMITH_CLIENT = None
    _WARNED_NOOP.clear()


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
    return sanitize_trace_metadata(metadata)


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
