from __future__ import annotations

import requests
from langchain_core.tools import tool


@tool
def fetch_url(url: str, limit: int = 100000) -> str:
    """Fetch raw text/HTML from a URL (truncated to `limit` chars). Increase limit when large tables are needed."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return f"ERROR: failed to fetch {url}: {e}"
    text = resp.text
    if len(text) > limit:
        return text[:limit] + f"\n...[truncated {len(text)-limit} chars]..."
    return text
