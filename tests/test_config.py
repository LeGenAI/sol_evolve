from pathlib import Path

from solevolve.config import load_settings


def test_load_settings_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SOLEVOLVE_ARTIFACT_DIR", "tmp_artifacts")
    settings = load_settings(require_api_key=False)

    assert settings.openrouter_api_key is None
    assert settings.artifact_dir == Path("tmp_artifacts")
    assert settings.openrouter_search_model == "openai/gpt-4o-mini"


def test_load_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        load_settings()
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)
    else:
        raise AssertionError("load_settings() should require OPENROUTER_API_KEY by default")
