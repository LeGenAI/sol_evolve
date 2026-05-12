from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import CodetablesLookupResult
from .tracing import update_current_run_context

CODETABLES_USER_AGENT = "solevolve-repro/0.1"


def bklc_url(*, q: int, n: int, k: int) -> str:
    return f"https://codetables.de/BKLC/BKLC.php?k={k}&n={n}&q={q}"


def _cache_path(artifact_dir: str | Path, *, q: int, n: int, k: int) -> Path:
    return Path(artifact_dir) / "codetables" / f"q{q}_n{n}_k{k}.json"


def _parse_bounds(text: str, *, q: int, n: int, k: int) -> dict[str, int | None]:
    construction_d: int | None = None
    lower_bound: int | None = None
    upper_bound: int | None = None

    construction = re.search(
        rf"linear code\s*\[\s*{n}\s*,\s*{k}\s*,\s*(\d+)\s*\]\s*over\s*GF\(\s*{q}\s*\)",
        text,
        re.IGNORECASE,
    )
    if construction:
        construction_d = int(construction.group(1))
        lower_bound = construction_d

    bound_patterns = [
        r"(?i)(?:lower\s+bound|lb)\D{0,40}(\d+)",
        r"(?i)(?:upper\s+bound|ub)\D{0,40}(\d+)",
        r"(?i)(\d+)\s*(?:<=|&le;)\s*d\s*(?:<=|&le;)\s*(\d+)",
    ]
    lower_match = re.search(bound_patterns[0], text)
    upper_match = re.search(bound_patterns[1], text)
    sandwich = re.search(bound_patterns[2], text)
    if lower_match:
        lower_bound = int(lower_match.group(1))
    if upper_match:
        upper_bound = int(upper_match.group(1))
    if sandwich:
        lower_bound = int(sandwich.group(1))
        upper_bound = int(sandwich.group(2))

    if construction_d is not None and upper_bound is None:
        upper_bound = construction_d
    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "construction_d": construction_d,
    }


def _fetch_text(url: str, *, timeout: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": CODETABLES_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _result_from_cache(path: Path, *, stale: bool, error: str | None = None) -> CodetablesLookupResult | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fetch_status"] = "STALE" if stale else "CACHE"
        if error:
            payload["error"] = error
        return CodetablesLookupResult.model_validate(payload)
    except Exception:
        return None


def lookup_codetables(
    *,
    q: int,
    n: int,
    k: int,
    artifact_dir: str | Path = "artifacts",
    timeout: int = 15,
    use_cache: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    url = bklc_url(q=q, n=n, k=k)
    cache_path = _cache_path(artifact_dir, q=q, n=n, k=k)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache:
        cached = _result_from_cache(cache_path, stale=False)
        if cached is not None:
            payload = cached.model_dump()
            payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            update_current_run_context(metadata={"codetables": payload})
            return payload

    try:
        text = _fetch_text(url, timeout=timeout)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parsed = _parse_bounds(text, q=q, n=n, k=k)
        result = CodetablesLookupResult(
            q=q,
            n=n,
            k=k,
            url=url,
            fetch_status="LIVE",
            content_sha256=digest,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            cache_path=str(cache_path),
            **parsed,
        )
        cache_path.write_text(json.dumps(result.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        stale = _result_from_cache(cache_path, stale=True, error=str(exc))
        if stale is not None:
            result = stale
        else:
            result = CodetablesLookupResult(
                q=q,
                n=n,
                k=k,
                url=url,
                fetch_status="UNAVAILABLE",
                cache_path=str(cache_path),
                error=str(exc),
            )

    payload = result.model_dump()
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    update_current_run_context(metadata={"codetables": payload})
    return payload
