import json
import os
import subprocess
from pathlib import Path

from helpers import console_script


def test_solver_smoke_console_skips_without_solver(tmp_path):
    env = os.environ.copy()
    env.pop("KISSAT_PATH", None)
    env.pop("CADICAL_PATH", None)
    env["PATH"] = ""

    proc = subprocess.run(
        [console_script("solevolve-solver-smoke"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "SKIPPED"


def test_artifact_check_console_succeeds():
    root = Path(__file__).resolve().parents[1]

    proc = subprocess.run(
        [
            console_script("solevolve-artifact-check"),
            "--manifest",
            str(root / "artifacts" / "manifest.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OK"
