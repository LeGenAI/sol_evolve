from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from shutil import which
from typing import Any


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
    if path.is_dir() or not path.suffix:
        names = {
            "cadical": [path / "build" / "cadical", path / "cadical"],
            "kissat": [path / "build" / "kissat", path / "kissat"],
        }
        return [path, *names.get(solver, [])]
    return [path]


def _solver_order(preferred_solver: str = "cadical") -> list[str]:
    preferred = preferred_solver.lower()
    if preferred not in {"cadical", "kissat"}:
        preferred = "cadical"
    fallback = "kissat" if preferred == "cadical" else "cadical"
    return [preferred, fallback]


def _candidates(root: Path, preferred_solver: str = "cadical") -> list[tuple[str, Path, str]]:
    items: list[tuple[str, Path, str]] = []
    env_names = {"cadical": "CADICAL_PATH", "kissat": "KISSAT_PATH"}
    repo_paths = {
        "cadical": [
            root / "sat_solvers" / "cadical" / "build" / "cadical",
            root / "cadical" / "build" / "cadical",
            root / "cadical",
            _user_cadical_candidate(),
        ],
        "kissat": [
            root / "kissat" / "build" / "kissat",
            root / "sat_solvers" / "kissat" / "build" / "kissat",
        ],
    }

    for name in _solver_order(preferred_solver):
        env_name = env_names[name]
        env_value = os.getenv(env_name)
        if env_value:
            for path in _expand_solver_path(Path(env_value), name):
                items.append((name, path, f"env:{env_name}"))
        for path in repo_paths[name]:
            for expanded in _expand_solver_path(path, name):
                items.append((name, expanded, "repo_default"))

    for name in _solver_order(preferred_solver):
        found = which(name)
        if found:
            items.append((name, Path(found), "PATH"))
    return items


def check_solver(root: Path, preferred_solver: str = "cadical") -> dict[str, Any]:
    started = time.perf_counter()
    cwd = str(root.resolve() if root.exists() else root)
    checked: list[dict[str, Any]] = []
    normalized_preference = _solver_order(preferred_solver)[0]
    for name, path, source in _candidates(root, normalized_preference):
        candidate_started = time.perf_counter()
        exists = path.exists()
        resolved_path = str(path.resolve()) if exists else str(path)
        candidate: dict[str, Any] = {
            "solver": name,
            "path": str(path),
            "resolved_path": resolved_path,
            "source": source,
            "exists": exists,
            "executable": exists and path.is_file() and os.access(path, os.X_OK),
            "preferred": name == normalized_preference,
        }
        checked.append(candidate)
        if not candidate["executable"]:
            candidate["elapsed_ms"] = round((time.perf_counter() - candidate_started) * 1000, 3)
            continue
        argv = [str(path), "--version"]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                cwd=cwd if Path(cwd).exists() else None,
            )
            version_elapsed_ms = round((time.perf_counter() - candidate_started) * 1000, 3)
            candidate["elapsed_ms"] = version_elapsed_ms
            candidate["returncode"] = result.returncode
        except Exception as exc:
            return {
                "status": "ERROR",
                "preferred_solver": normalized_preference,
                "selected_solver": name,
                "solver": name,
                "selected_path": str(path),
                "selected_resolved_path": resolved_path,
                "selected_source": source,
                "path": str(path),
                "cwd": cwd,
                "argv": argv,
                "command": " ".join(argv),
                "error": str(exc),
                "checked": checked,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        version_lines = (result.stdout or result.stderr).strip().splitlines()[:3]
        return {
            "status": "OK" if result.returncode == 0 else "ERROR",
            "preferred_solver": normalized_preference,
            "selected_solver": name,
            "solver": name,
            "selected_path": str(path),
            "selected_resolved_path": resolved_path,
            "selected_source": source,
            "path": str(path),
            "cwd": cwd,
            "argv": argv,
            "command": " ".join(argv),
            "returncode": result.returncode,
            "version": version_lines,
            "version_stdout": result.stdout.strip().splitlines()[:3],
            "version_stderr": result.stderr.strip().splitlines()[:3],
            "version_elapsed_ms": version_elapsed_ms,
            "checked": checked,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    return {
        "status": "SKIPPED",
        "preferred_solver": normalized_preference,
        "reason": "No Kissat or CaDiCaL binary found. Set KISSAT_PATH or CADICAL_PATH to enable SAT checks.",
        "cwd": cwd,
        "checked": checked,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
