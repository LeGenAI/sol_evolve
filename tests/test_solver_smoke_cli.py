from pathlib import Path

from solevolve.solver_smoke import check_solver


def test_solver_smoke_skips_without_solver(monkeypatch, tmp_path):
    monkeypatch.delenv("KISSAT_PATH", raising=False)
    monkeypatch.delenv("CADICAL_PATH", raising=False)
    monkeypatch.setenv("PATH", "")

    payload = check_solver(Path(tmp_path))

    assert payload["status"] == "SKIPPED"
    assert "checked" in payload
