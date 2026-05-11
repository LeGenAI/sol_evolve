import json
import os
import subprocess

from helpers import console_script


def test_dry_run_cli(tmp_path):
    env = os.environ.copy()
    env["SOLEVOLVE_ARTIFACT_DIR"] = str(tmp_path)
    env.pop("OPENROUTER_API_KEY", None)

    proc = subprocess.run(
        [
            console_script("solevolve-demo"),
            "--dry-run",
            "--max-turns",
            "1",
            "Goal: dry-run test",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=True,
    )

    last_line = proc.stdout.strip().splitlines()[-1]
    payload = json.loads(last_line)
    assert payload["artifact_summary"].endswith("last_run_summary.json")
