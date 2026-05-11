# LangSmith Tracing

SolEvolve uses LangChain/LangGraph tracing when LangSmith is configured and a no-op fallback otherwise.

## Local setup

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=solevolve-repro
export LANGCHAIN_CALLBACKS_BACKGROUND=false
```

Run a no-API traceable smoke test:

```bash
solevolve-demo --dry-run --max-turns 1
```

Run an LLM-backed trace:

```bash
export OPENROUTER_API_KEY=...
solevolve-demo "Goal: outline CNF encoding steps for [32,14,8]."
```

## Inspect traces

```bash
langsmith trace list --project solevolve-repro --limit 10 --include-metadata --api-key "$LANGSMITH_API_KEY"
langsmith trace list --project solevolve-repro --limit 5 --show-hierarchy --api-key "$LANGSMITH_API_KEY"
langsmith trace export artifacts/langsmith_traces --project solevolve-repro --limit 20 --full --api-key "$LANGSMITH_API_KEY"
```

## Metadata policy

Traces should include run parameters, solver status, timing, artifact paths, hashes, and concise verifier summaries. They should not include raw API keys, full CNF payloads, full solver models, or large generator matrices.
