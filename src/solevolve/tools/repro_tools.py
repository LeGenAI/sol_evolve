from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from ..artifact_check import check_manifest
from ..codetables import lookup_codetables as _lookup_codetables
from ..solver_check import check_solver
from ..tracing import traceable_run
from .schemas import ArtifactCheckArgs, CodetablesLookupArgs, SolverCheckArgs


@tool(args_schema=ArtifactCheckArgs)
@traceable_run("solevolve.artifact_check", run_type="tool")
def artifact_check(manifest_path: str = "artifacts/manifest.json") -> str:
    """Validate the public SolEvolve artifact manifest and return compact JSON evidence."""
    payload = check_manifest(Path(manifest_path))
    return json.dumps(payload, sort_keys=True)


@tool(args_schema=SolverCheckArgs)
@traceable_run("solevolve.solver_check", run_type="tool")
def solver_check(root: str = ".", preferred_solver: str = "cadical") -> str:
    """Check SAT solver availability with CaDiCaL-first preference and return compact JSON evidence."""
    payload = check_solver(Path(root), preferred_solver=preferred_solver)
    return json.dumps(payload, sort_keys=True)


@tool(args_schema=CodetablesLookupArgs)
@traceable_run("solevolve.codetables_lookup", run_type="tool")
def codetables_lookup(
    q: int,
    n: int,
    k: int,
    artifact_dir: str = "artifacts",
    timeout: int = 15,
    use_cache: bool = True,
) -> str:
    """Query codetables.de BKLC with sanitized live/cache fallback and return compact JSON evidence."""
    payload = _lookup_codetables(q=q, n=n, k=k, artifact_dir=artifact_dir, timeout=timeout, use_cache=use_cache)
    return json.dumps(payload, sort_keys=True)
