from pathlib import Path

from solevolve.tools.solver_tools import _guess_solver_candidates


def test_solver_candidates_include_env_path(monkeypatch, tmp_path):
    solver = tmp_path / "kissat"
    monkeypatch.setenv("KISSAT_PATH", str(solver))

    candidates = _guess_solver_candidates("kissat")

    assert Path(solver) in candidates
