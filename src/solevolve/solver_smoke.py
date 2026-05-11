from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Any


def _candidates(root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for name, env_name in (("kissat", "KISSAT_PATH"), ("cadical", "CADICAL_PATH")):
        env_value = os.getenv(env_name)
        if env_value:
            items.append((name, Path(env_value)))

    items.extend(
        [
            ("kissat", root / "kissat" / "build" / "kissat"),
            ("kissat", root / "sat_solvers" / "kissat" / "build" / "kissat"),
            ("cadical", root / "sat_solvers" / "cadical" / "build" / "cadical"),
        ]
    )

    for name in ("kissat", "cadical"):
        found = which(name)
        if found:
            items.append((name, Path(found)))
    return items


def check_solver(root: Path) -> dict[str, Any]:
    checked: list[dict[str, str]] = []
    for name, path in _candidates(root):
        checked.append({"solver": name, "path": str(path)})
        if not path.exists():
            continue
        try:
            result = subprocess.run(
                [str(path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:
            return {
                "status": "ERROR",
                "solver": name,
                "path": str(path),
                "error": str(exc),
                "checked": checked,
            }
        return {
            "status": "OK" if result.returncode == 0 else "ERROR",
            "solver": name,
            "path": str(path),
            "returncode": result.returncode,
            "version": (result.stdout or result.stderr).strip().splitlines()[:3],
            "checked": checked,
        }

    return {
        "status": "SKIPPED",
        "reason": "No Kissat or CaDiCaL binary found. Set KISSAT_PATH or CADICAL_PATH to enable SAT smoke checks.",
        "checked": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether an optional SAT solver is available.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root for local solver path checks.")
    parser.add_argument(
        "--require-solver",
        action="store_true",
        help="Return non-zero if no solver is available.",
    )
    args = parser.parse_args()

    payload = check_solver(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] == "ERROR" or (args.require_solver and payload["status"] == "SKIPPED"):
        sys.exit(1)


if __name__ == "__main__":
    main()
