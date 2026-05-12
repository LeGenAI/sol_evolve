from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_COMMANDS = {
    "solevolve-reviewer-reproduce --paper-claim-id all_reviewer_core",
}


def check_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "ERROR", "manifest": str(path), "error": "manifest not found"}

    root = path.parent.parent if path.parent.name == "artifacts" else Path.cwd()
    data = json.loads(path.read_text(encoding="utf-8"))
    bundled = data.get("bundled", [])
    missing = [item for item in bundled if not (root / item).exists()]
    required_fields = [
        "artifact_version",
        "bundled",
        "excluded_from_git",
        "official_commands",
        "raw_artifact_policy",
        "runtime_output_dir",
    ]
    missing_fields = [field for field in required_fields if not data.get(field)]
    missing_commands = sorted(REQUIRED_COMMANDS.difference(data.get("official_commands", [])))

    return {
        "status": "OK" if not (missing or missing_fields or missing_commands) else "ERROR",
        "manifest": str(path),
        "artifact_version": data.get("artifact_version"),
        "runtime_output_dir": data.get("runtime_output_dir"),
        "bundled_count": len(bundled),
        "missing": missing,
        "missing_fields": missing_fields,
        "missing_commands": missing_commands,
        "excluded_from_git": data.get("excluded_from_git", []),
    }
