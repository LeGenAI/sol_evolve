from __future__ import annotations

import shlex
import subprocess
import sys
import traceback
import os
from pathlib import Path
from shutil import which
from typing import Iterable

from langchain_core.tools import tool

from ..engines.code_generator import CodeGenerator
from ..tracing import traceable_run

ROOT_DIR = Path(__file__).resolve().parents[3]


def _repo_path(*parts: Iterable[str | Path]) -> Path:
    return ROOT_DIR.joinpath(*parts)


def _format_error(header: str, exc: Exception) -> str:
    return f"{header}: {exc}\n{traceback.format_exc()}"


def _import_code_generator():
    return CodeGenerator


def _guess_solver_candidates(name: str) -> list[Path]:
    lower = name.lower()
    candidates: list[Path] = []

    if lower in {"cadical", "cad"}:
        env_path = os.getenv("CADICAL_PATH")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(_repo_path("sat_solvers", "cadical", "build", "cadical"))
    if lower in {"kissat"}:
        env_path = os.getenv("KISSAT_PATH")
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(_repo_path("kissat", "build", "kissat"))
        candidates.append(_repo_path("sat_solvers", "kissat", "build", "kissat"))
    if lower in {"hcadsbva", "hcadpsbva"}:
        candidates.append(_repo_path("sat_solvers", "SBVA", "sbva_wrapped"))

    found_in_path = which(lower)
    if found_in_path:
        candidates.append(Path(found_in_path))

    return candidates


def _run_solver_impl(
    *,
    n: int,
    k: int,
    d: int,
    solver: str = "cadical",
    solver_path: str | None = None,
    num_solutions: int = 1,
    timeout: int = 120,
    workdir: str | None = None,
    systematic: bool = True,
    verbose: bool = False,
) -> str:
    CodeGenerator = _import_code_generator()

    candidates = []
    if solver_path:
        candidates.append(Path(solver_path))
    candidates.extend(_guess_solver_candidates(solver))

    resolved_solver = next((p for p in candidates if p.exists()), None)
    solver_msg = (
        f"Solver resolved: {resolved_solver}"
        if resolved_solver
        else f"Solver {solver} not found in candidates: {[str(c) for c in candidates]}"
    )

    if resolved_solver is None:
        return f"ERROR: solver missing. {solver_msg}"

    artifact_root = Path(os.getenv("SOLEVOLVE_ARTIFACT_DIR", "artifacts"))
    out_dir = Path(workdir) if workdir else _repo_path(artifact_root, "binary_codes", f"{solver}_{n}_{k}_{d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gen = CodeGenerator(n=n, k=k, d_min=d, systematic=systematic, working_dir=str(out_dir))
    except Exception as exc:
        return _format_error("ERROR: failed to construct CodeGenerator", exc)

    try:
        if num_solutions > 1:
            results = gen.solve_multiple(
                solver_type=solver,
                solver_path=str(resolved_solver),
                num_solutions=num_solutions,
                timeout_per_instance=timeout,
                verbose=verbose,
            )
            summary_lines = [
                f"Solver: {resolved_solver}",
                f"Workdir: {out_dir}",
                f"Requested: {num_solutions}, Found: {len(results)}",
            ]
            for res in results:
                md = res.get("verification", {}).get("minimum_distance") if res.get("verification") else None
                matrix_path = out_dir / f"{gen.code_name}_solution{res.get('solution_number')}_matrix.npy"
                summary_lines.append(
                    f"- #{res.get('solution_number')}: status={res.get('status')}, "
                    f"d_min={md}, solver_time={res.get('solver_time'):.2f}s, matrix={matrix_path}"
                )
            return "\n".join(summary_lines)

        result = gen.solve(
            kissat_path=str(resolved_solver),
            timeout=timeout,
            verbose=verbose,
        )
        md = result.get("verification", {}).get("minimum_distance") if result.get("verification") else None
        matrix_path = out_dir / f"{gen.code_name}_matrix.npy"
        txt_path = out_dir / f"{gen.code_name}_result.txt"
        return "\n".join(
            [
                f"Solver: {resolved_solver}",
                f"Workdir: {out_dir}",
                f"Status: {result.get('status')}",
                f"Min distance (verified): {md}",
                f"Solver time: {result.get('solver_time'):.2f}s" if result.get("solver_time") else "Solver time: n/a",
                f"Matrix: {matrix_path}",
                f"Result text: {txt_path}",
            ]
        )
    except Exception as exc:
        guidance = (
            "Guidance: verify solver binary exists and is executable; "
            "check CNF generation in src/kissat_code_search; "
            "consider lowering n/k/d or enabling systematic encoding."
        )
        return f"{_format_error('ERROR during solve', exc)}\n{guidance}"


@tool
@traceable_run("solevolve.locate_solver", run_type="tool")
def locate_solver(name: str = "cadical", hint: str | None = None) -> str:
    """
    Locate a SAT solver binary by common repo paths or PATH lookup.
    Provide `hint` to check a custom path first.
    """
    candidates = []
    if hint:
        candidates.append(Path(hint))
    candidates.extend(_guess_solver_candidates(name))
    existing = [p for p in candidates if p.exists()]
    lines = [f"Solver name: {name}", f"Hints tried: {[str(c) for c in candidates]}"]
    if existing:
        lines.append(f"Found: {existing[0]}")
    else:
        lines.append("Not found.")
        lines.append("Tip: build CaDiCaL with `make` in sat_solvers/cadical; Kissat in src/kissat_code_search/kissat.")
    return "\n".join(lines)


@tool
@traceable_run("solevolve.run_solver", run_type="tool")
def run_solver(
    n: int,
    k: int,
    d: int,
    solver: str = "cadical",
    solver_path: str | None = None,
    num_solutions: int = 1,
    timeout: int = 120,
    workdir: str | None = None,
    systematic: bool = True,
    verbose: bool = False,
) -> str:
    """
    Generate encoding and run a SAT solver via CodeGenerator.
    Returns a concise summary with file paths.
    """
    return _run_solver_impl(
        n=n,
        k=k,
        d=d,
        solver=solver,
        solver_path=solver_path,
        num_solutions=num_solutions,
        timeout=timeout,
        workdir=workdir,
        systematic=systematic,
        verbose=verbose,
    )


@tool
@traceable_run("solevolve.verify_code", run_type="tool")
def verify_code(matrix_path: str, n: int, k: int, d: int, systematic: bool = True) -> str:
    """
    Load a saved generator matrix (.npy) and verify minimum distance.
    """
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - defensive
        return _format_error("ERROR: numpy import failed", exc)

    try:
        CodeGenerator = _import_code_generator()
        gen = CodeGenerator(n=n, k=k, d_min=d, systematic=systematic, working_dir=str(Path(matrix_path).parent))
        G = np.load(matrix_path)
        verification = gen.parser.verify_generator_matrix(G, verbose=False)
        return "\n".join(
            [
                f"Matrix: {matrix_path}",
                f"n={n}, k={k}, d_target={d}, systematic={systematic}",
                f"Verified minimum distance: {verification.get('minimum_distance')}",
                f"Valid: {verification.get('is_valid')}",
            ]
        )
    except Exception as exc:
        return _format_error("ERROR during verification", exc)


# Backward-compatible alias for earlier demos.
@tool
@traceable_run("solevolve.run_sat_search", run_type="tool")
def run_sat_search(
    n: int,
    k: int,
    d: int,
    num_solutions: int = 1,
    timeout: int = 120,
    workdir: str | None = None,
    systematic: bool = True,
    solver: str = "cadical",
    solver_path: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Alias to run_solver; kept for compatibility with older notebooks/demos.
    """
    return _run_solver_impl(
        n=n,
        k=k,
        d=d,
        solver=solver,
        solver_path=solver_path,
        num_solutions=num_solutions,
        timeout=timeout,
        workdir=workdir,
        systematic=systematic,
        verbose=verbose,
    )


@tool
def run_python(
    path: str,
    args: str | None = None,
    cwd: str | None = None,
    timeout: int = 120,
    **kwargs: object,
) -> str:
    """
    Execute a local Python script and return stdout/stderr.
    Provide additional CLI args via `args` (space separated).
    """
    target = Path(path)
    if not target.exists():
        return f"ERROR: file not found: {target}"

    cmd = ["python3", str(target)]
    v_args = kwargs.get("v__args")
    if v_args and not args:
        # DeepAgents may pass vectorized arguments via v__args; use them if args is absent.
        if isinstance(v_args, (list, tuple)):
            args = " ".join(str(x) for x in v_args)
    if args:
        cmd.extend(shlex.split(args))

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or str(target.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        status = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = f"TIMEOUT after {timeout}s"
    except Exception as exc:
        return _format_error("ERROR: failed to execute python script", exc)

    def _trim(txt: str, label: str) -> str:
        if len(txt) > 4000:
            return f"[{label} truncated to 4000 chars]\n" + txt[:4000]
        return f"[{label}]\n{txt}" if txt else f"[{label}] (empty)"

    return "\n".join([f"Command: {' '.join(cmd)}", f"Status: {status}", _trim(stdout, "stdout"), _trim(stderr, "stderr")])
