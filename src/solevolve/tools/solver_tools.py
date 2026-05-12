from __future__ import annotations

import shlex
import subprocess
import sys
import time
import traceback
import os
from pathlib import Path
from shutil import which
from typing import Any, Iterable

from langchain_core.tools import tool

from ..engines.code_generator import CodeGenerator
from ..tracing import traceable_run, update_current_run_context
from .schemas import LocateSolverArgs, RunPythonArgs, RunSolverArgs, VerifyCodeArgs

ROOT_DIR = Path(__file__).resolve().parents[3]


def _repo_path(*parts: Iterable[str | Path]) -> Path:
    return ROOT_DIR.joinpath(*parts)


def _user_cadical_candidate() -> Path:
    return (
        Path.home()
        / "Desktop"
        / "CodingTheoryLib"
        / "references"
        / ("Code" + "Evolve")
        / "sat_solvers"
        / "cadical"
    )


def _expand_solver_path(path: Path, solver: str) -> list[Path]:
    lower = solver.lower()
    if path.is_dir() or not path.suffix:
        if lower in {"cadical", "cad"}:
            return [path, path / "build" / "cadical", path / "cadical"]
        if lower == "kissat":
            return [path, path / "build" / "kissat", path / "kissat"]
    return [path]


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
            candidates.extend(_expand_solver_path(Path(env_path), lower))
        candidates.append(_repo_path("sat_solvers", "cadical", "build", "cadical"))
        candidates.extend(_expand_solver_path(_user_cadical_candidate(), lower))
    if lower in {"kissat"}:
        env_path = os.getenv("KISSAT_PATH")
        if env_path:
            candidates.extend(_expand_solver_path(Path(env_path), lower))
        candidates.append(_repo_path("kissat", "build", "kissat"))
        candidates.append(_repo_path("sat_solvers", "kissat", "build", "kissat"))
    if lower in {"hcadsbva", "hcadpsbva"}:
        candidates.append(_repo_path("sat_solvers", "SBVA", "sbva_wrapped"))

    found_in_path = which(lower)
    if found_in_path:
        candidates.append(Path(found_in_path))

    return candidates


def _resolve_solver(*, solver: str, solver_path: str | None) -> tuple[Path | None, list[dict[str, Any]], float]:
    started = time.perf_counter()
    candidates = []
    if solver_path:
        candidates.extend(_expand_solver_path(Path(solver_path), solver))
    candidates.extend(_guess_solver_candidates(solver))

    checked: list[dict[str, Any]] = []
    resolved_solver: Path | None = None
    for path in candidates:
        exists = path.exists()
        executable = exists and os.access(path, os.X_OK) and path.is_file()
        checked.append({"path": str(path), "exists": exists, "executable": executable})
        if executable and resolved_solver is None:
            resolved_solver = path
            checked[-1]["selected"] = True
            break

    return resolved_solver, checked, time.perf_counter() - started


def _format_seconds(value: Any) -> str:
    return f"{float(value):.2f}s" if value is not None else "n/a"


def _record_solver_trace_metadata(**metadata: Any) -> None:
    update_current_run_context(
        metadata={
            "solver": metadata,
            "solver_backend": metadata.get("selected_solver") or metadata.get("requested_solver"),
            "solver_path": metadata.get("selected_solver_path"),
            "solver_elapsed_ms": metadata.get("solver_elapsed_ms"),
            "solver_total_elapsed_ms": metadata.get("total_elapsed_ms"),
        }
    )


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
    tool_started = time.perf_counter()
    CodeGenerator = _import_code_generator()

    resolved_solver, checked_candidates, resolve_elapsed = _resolve_solver(solver=solver, solver_path=solver_path)
    solver_msg = (
        f"Solver resolved: {resolved_solver}"
        if resolved_solver
        else f"Solver {solver} not found in candidates: {[item['path'] for item in checked_candidates]}"
    )

    if resolved_solver is None:
        _record_solver_trace_metadata(
            requested_solver=solver,
            selected_solver=None,
            selected_solver_path=None,
            resolution_elapsed_ms=round(resolve_elapsed * 1000, 3),
            total_elapsed_ms=round((time.perf_counter() - tool_started) * 1000, 3),
            checked_candidates=checked_candidates,
            status="ERROR",
        )
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
            total_elapsed = time.perf_counter() - tool_started
            solver_times = [res.get("solver_time") for res in results if res.get("solver_time") is not None]
            _record_solver_trace_metadata(
                requested_solver=solver,
                selected_solver=solver,
                selected_solver_path=str(resolved_solver),
                resolution_elapsed_ms=round(resolve_elapsed * 1000, 3),
                total_elapsed_ms=round(total_elapsed * 1000, 3),
                solver_elapsed_ms=round(sum(float(value) for value in solver_times) * 1000, 3) if solver_times else None,
                num_solutions_requested=num_solutions,
                num_solutions_found=len(results),
                statuses=[res.get("status") for res in results],
                checked_candidates=checked_candidates,
            )
            summary_lines = [
                f"Requested solver: {solver}",
                f"Selected solver: {solver}",
                f"Solver path: {resolved_solver}",
                f"Resolution time: {resolve_elapsed * 1000:.2f}ms",
                f"Workdir: {out_dir}",
                f"Requested: {num_solutions}, Found: {len(results)}",
                f"Tool elapsed: {_format_seconds(total_elapsed)}",
            ]
            for res in results:
                md = res.get("verification", {}).get("minimum_distance") if res.get("verification") else None
                matrix_path = out_dir / f"{gen.code_name}_solution{res.get('solution_number')}_matrix.npy"
                summary_lines.append(
                    f"- #{res.get('solution_number')}: status={res.get('status')}, "
                    f"d_min={md}, solver_time={_format_seconds(res.get('solver_time'))}, matrix={matrix_path}"
                )
            return "\n".join(summary_lines)

        result = gen.solve(
            solver_type=solver,
            solver_path=str(resolved_solver),
            timeout=timeout,
            verbose=verbose,
        )
        total_elapsed = time.perf_counter() - tool_started
        md = result.get("verification", {}).get("minimum_distance") if result.get("verification") else None
        matrix_path = out_dir / f"{gen.code_name}_matrix.npy"
        txt_path = out_dir / f"{gen.code_name}_result.txt"
        _record_solver_trace_metadata(
            requested_solver=solver,
            selected_solver=result.get("solver_used") or solver,
            selected_solver_path=str(resolved_solver),
            resolution_elapsed_ms=round(resolve_elapsed * 1000, 3),
            total_elapsed_ms=round(total_elapsed * 1000, 3),
            solver_elapsed_ms=round(float(result["solver_time"]) * 1000, 3) if result.get("solver_time") is not None else None,
            status=result.get("status"),
            num_solutions_requested=num_solutions,
            num_solutions_found=1 if result.get("status") == "SAT" else 0,
            checked_candidates=checked_candidates,
        )
        return "\n".join(
            [
                f"Requested solver: {solver}",
                f"Selected solver: {result.get('solver_used') or solver}",
                f"Solver path: {resolved_solver}",
                f"Resolution time: {resolve_elapsed * 1000:.2f}ms",
                f"Workdir: {out_dir}",
                f"Status: {result.get('status')}",
                f"Min distance (verified): {md}",
                f"Solver time: {_format_seconds(result.get('solver_time'))}",
                f"Total time: {_format_seconds(result.get('total_time'))}",
                f"Tool elapsed: {_format_seconds(total_elapsed)}",
                f"Matrix: {matrix_path}",
                f"Result text: {txt_path}",
            ]
        )
    except Exception as exc:
        _record_solver_trace_metadata(
            requested_solver=solver,
            selected_solver=solver,
            selected_solver_path=str(resolved_solver),
            resolution_elapsed_ms=round(resolve_elapsed * 1000, 3),
            total_elapsed_ms=round((time.perf_counter() - tool_started) * 1000, 3),
            checked_candidates=checked_candidates,
            status="ERROR",
            error=str(exc),
        )
        guidance = (
            "Guidance: verify solver binary exists and is executable; "
            "check CNF generation in src/kissat_code_search; "
            "consider lowering n/k/d or enabling systematic encoding."
        )
        return f"{_format_error('ERROR during solve', exc)}\n{guidance}"


@tool(args_schema=LocateSolverArgs)
@traceable_run("solevolve.locate_solver", run_type="tool")
def locate_solver(name: str = "cadical", hint: str | None = None) -> str:
    """
    Locate a SAT solver binary without running a SAT instance.

    Use before run_solver when solver availability is uncertain. Checks an optional
    explicit hint, environment variables, common repo build paths, and PATH.
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


@tool(args_schema=RunSolverArgs)
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
    Preferred binary-code SAT search tool; writes CNF, solver output, and matrices under workdir/artifacts.

    Requires an available SAT solver binary. Returns a concise status summary with
    output paths; missing solver binaries return ERROR rather than fabricating evidence.
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


@tool(args_schema=VerifyCodeArgs)
@traceable_run("solevolve.verify_code", run_type="tool")
def verify_code(matrix_path: str, n: int, k: int, d: int, systematic: bool = True) -> str:
    """
    Verify a saved binary generator matrix (.npy) against n, k, and target distance d.

    This is read-only for the matrix file and returns a concise verification summary.
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
@tool(args_schema=RunSolverArgs)
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
    Compatibility alias for run_solver; prefer run_solver for new workflows.

    It has the same solver requirements and file-writing side effects as run_solver.
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


@tool(args_schema=RunPythonArgs)
@traceable_run("solevolve.run_python", run_type="tool")
def run_python(
    path: str,
    args: str | None = None,
    cwd: str | None = None,
    timeout: int = 120,
    **kwargs: object,
) -> str:
    """
    Manual-only: not used by the default repro graph.

    Executes a local Python script and returns truncated stdout/stderr. Do not use
    for autonomous evidence claims unless the caller explicitly authorizes the script,
    cwd, arguments, timeout, and secret-handling risk.
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
