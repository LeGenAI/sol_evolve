# Contributing

Keep the [reviewer commands](docs/reviewer_reproduction.md) and the checked artifacts in sync. Use the public CLI to verify changes to proof obligations, solver selection, or exit codes. Add regression checks for changes to these contracts.

Run the offline checks in a Python 3.12+ virtual environment:

```bash
python -m pip install -e .
python -m compileall -q src/solevolve
python scripts/check_reviewer_acceptance.py
```

Use the full reviewer command with CaDiCaL for changes to the SAT encodings. Keep generated CNFs, solver logs, and API credentials outside the repository.
