# Contributing

Contributions should preserve the public reproducibility contract:

1. Keep `solevolve-demo --dry-run --max-turns 1` working without API keys.
2. Do not commit `.env`, API keys, large CNF files, solver outputs, or generated NumPy matrices.
3. Add or update tests for changes to config, tracing, solver path resolution, or CLI behavior.
4. Keep README claims aligned with checked artifacts and the revised manuscript framing.

Run before opening a pull request:

```bash
python -m pip install -e ".[dev]"
python -m py_compile $(find src/solevolve -name '*.py')
pytest
solevolve-demo --dry-run --max-turns 1
solevolve-solver-smoke
solevolve-artifact-check
```
