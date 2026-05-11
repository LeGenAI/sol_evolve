from solevolve.tracing import traceable_run, tracing_enabled


def test_tracing_noops_without_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    @traceable_run("test.noop")
    def add_one(value: int) -> int:
        return value + 1

    assert tracing_enabled() is False
    assert add_one(2) == 3
