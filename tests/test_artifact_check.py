from pathlib import Path

from solevolve.artifact_check import check_manifest


def test_artifact_manifest_is_valid():
    root = Path(__file__).resolve().parents[1]
    payload = check_manifest(root / "artifacts" / "manifest.json")

    assert payload["status"] == "OK"
    assert payload["bundled_count"] == 3
    assert payload["missing"] == []
    assert payload["missing_fields"] == []
    assert payload["missing_commands"] == []
