# Contributing

Contributions should preserve the public reproducibility contract:

1. Keep `solevolve-reviewer-reproduce --paper-claim-id all_so_table` as the reviewer acceptance path.
2. Do not commit `.env`, API keys, large CNF files, solver outputs, or generated NumPy matrices.
3. Add or update focused tests for changes to contracts, tracing sanitization, solver path resolution, or proof logic.
4. Keep README claims aligned with checked artifacts and the revised manuscript framing.

Run before opening a pull request:

```bash
python -m pip install -e ".[dev]"
python -m py_compile $(find src/solevolve -name '*.py')
pytest
```
