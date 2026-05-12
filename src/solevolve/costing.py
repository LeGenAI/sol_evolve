from __future__ import annotations

import functools
import json
import os
import re
import urllib.request
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _float_from_env(*names: str) -> float | None:
    for name in names:
        value = os.getenv(name)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _models_url(base_url: str | None) -> str:
    base = (base_url or os.getenv("OPENROUTER_BASE_URL") or _DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    if base.endswith("/models"):
        return base
    return f"{base}/models"


def normalize_openrouter_model_id(model: str | None) -> str | None:
    if not model:
        return None
    normalized = re.sub(r"-\d{8}$", "", model.strip())
    match = re.match(r"^(anthropic/claude)-(?P<version>\d+(?:\.\d+)?)-sonnet$", normalized)
    if match:
        return f"{match.group(1)}-sonnet-{match.group('version')}"
    return normalized


@functools.lru_cache(maxsize=8)
def _fetch_openrouter_models(url: str, timeout: float) -> tuple[dict[str, Any], ...]:
    request = urllib.request.Request(url, headers={"User-Agent": "solevolve-cost-tracker/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return tuple(item for item in data if isinstance(item, dict))


def resolve_model_pricing(model: str | None, *, base_url: str | None = None) -> dict[str, Any] | None:
    prompt_per_1m = _float_from_env("SOLEVOLVE_PROMPT_COST_PER_1M", "OPENROUTER_PROMPT_COST_PER_1M")
    completion_per_1m = _float_from_env(
        "SOLEVOLVE_COMPLETION_COST_PER_1M",
        "OPENROUTER_COMPLETION_COST_PER_1M",
    )
    if prompt_per_1m is not None and completion_per_1m is not None:
        return {
            "model_id": model,
            "prompt_per_token": prompt_per_1m / 1_000_000,
            "completion_per_token": completion_per_1m / 1_000_000,
            "prompt_per_1m": prompt_per_1m,
            "completion_per_1m": completion_per_1m,
            "currency": "USD",
            "source": "env",
        }

    if os.getenv("SOLEVOLVE_DISABLE_OPENROUTER_PRICING", "").lower() in _TRUTHY:
        return None

    model_candidates = [candidate for candidate in {model, normalize_openrouter_model_id(model)} if candidate]
    if not model_candidates:
        return None

    timeout = _float_from_env("SOLEVOLVE_OPENROUTER_PRICING_TIMEOUT") or 2.5
    try:
        models = _fetch_openrouter_models(_models_url(base_url), timeout)
    except Exception:
        return None

    for item in models:
        item_id = item.get("id")
        if item_id not in model_candidates:
            continue
        pricing = item.get("pricing") or {}
        try:
            prompt_per_token = float(pricing["prompt"])
            completion_per_token = float(pricing["completion"])
        except Exception:
            return None
        return {
            "model_id": item_id,
            "prompt_per_token": prompt_per_token,
            "completion_per_token": completion_per_token,
            "prompt_per_1m": prompt_per_token * 1_000_000,
            "completion_per_1m": completion_per_token * 1_000_000,
            "currency": "USD",
            "source": "openrouter_models_api",
        }
    return None


def token_usage_counts(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not isinstance(usage, dict):
        try:
            usage = dict(usage)
        except Exception:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def estimate_llm_cost(
    usage: Any,
    *,
    model: str | None,
    base_url: str | None = None,
) -> dict[str, Any]:
    counts = token_usage_counts(usage)
    pricing = resolve_model_pricing(model, base_url=base_url)
    if pricing is None:
        return {
            **counts,
            "estimated_cost_available": False,
            "estimated_cost_usd": None,
            "currency": "USD",
            "pricing_source": "unavailable",
            "pricing_model_id": normalize_openrouter_model_id(model) or model,
        }

    prompt_cost = counts["prompt_tokens"] * pricing["prompt_per_token"]
    completion_cost = counts["completion_tokens"] * pricing["completion_per_token"]
    total_cost = prompt_cost + completion_cost
    return {
        **counts,
        "estimated_cost_available": True,
        "estimated_cost_usd": round(total_cost, 8),
        "prompt_cost_usd": round(prompt_cost, 8),
        "completion_cost_usd": round(completion_cost, 8),
        "currency": pricing["currency"],
        "pricing_source": pricing["source"],
        "pricing_model_id": pricing["model_id"],
        "prompt_cost_per_1m_tokens_usd": round(pricing["prompt_per_1m"], 8),
        "completion_cost_per_1m_tokens_usd": round(pricing["completion_per_1m"], 8),
    }
