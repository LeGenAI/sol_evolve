from __future__ import annotations

import requests
from langchain_core.tools import tool

from ..tracing import traceable_run
from .schemas import FetchUrlArgs


@tool(args_schema=FetchUrlArgs)
@traceable_run("solevolve.fetch_url", run_type="tool")
def fetch_url(url: str, limit: int = 100000) -> str:
    """Manual-only network fetch; not used by the default repro graph. Fetch HTTP/HTTPS text truncated to limit."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: failed to fetch {url}: {e}"
    text = resp.text
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text)-limit} chars]..."
    return text
