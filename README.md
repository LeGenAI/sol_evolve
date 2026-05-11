# SolEvolve

SolEvolve is a research codebase for LLM-guided, solver-verified combinatorial search. The current public artifact focuses on reproducible scaffolding for encoding selection, solver orchestration, verifier summaries, and Reflector-style feedback.

The repository is aligned with the Neurocomputing revision framing: SolEvolve is not presented as a general mathematical discovery engine or as a replacement for SAT/ILP/Magma. It is a workflow for selected combinatorial search tasks where LLM-generated strategies are checked by deterministic solvers or verifiers.

## Install

Python 3.12+ is required.

```bash
git clone https://github.com/LeGenAI/sol_evolve.git
cd sol_evolve
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For compatibility with older workflows:

```bash
python -m pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` for local runs. Keep `.env` private.

```bash
cp .env.example .env
```

For a no-API smoke test, no keys are required:

```bash
solevolve-demo --dry-run --max-turns 1
```

For LLM-backed runs through OpenRouter, set:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

For LangSmith tracing, set:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=solevolve-repro
```

## Official Commands

The public package exposes three supported commands:

```bash
solevolve-demo --dry-run --max-turns 1
solevolve-solver-smoke
solevolve-artifact-check
```

- `solevolve-demo`: runs the traceable SolEvolve graph. The default public smoke path is `--dry-run --max-turns 1`, which does not require API keys or SAT solvers.
- `solevolve-solver-smoke`: checks `KISSAT_PATH`, `CADICAL_PATH`, common local build paths, and `PATH`. Missing solvers are reported as `SKIPPED` unless `--require-solver` is passed.
- `solevolve-artifact-check`: validates `artifacts/manifest.json` and confirms that bundled summary files are present.

## Reproducibility Layout

- `src/solevolve/agents`: Generator, Evolver, Verifier, and Reflector agent wrappers.
- `src/solevolve/engines`: SAT/CNF encoding and solver interfaces for binary code search.
- `results/*/summary.txt`: compact summaries for archived experiment outputs.
- `artifacts/manifest.json`: public manifest describing which evidence is bundled and which raw logs should be released separately.

Large CNF files, NumPy matrices, solver outputs, private raw console logs, and long-form exploratory drivers are intentionally excluded from the git tree. They should be distributed through a release asset, Zenodo record, or raw artifact archive. Use `SOLEVOLVE_ARTIFACT_DIR` to point runtime commands to local or downloaded artifacts.

See `docs/reproducibility.md` for the paper-code checklist, included/excluded artifact policy, and benchmarked open-source release standards used for this cleanup.

## What This Release Does Not Include

- It does not include private `.env` files or API keys.
- It does not include the stale manuscript source from the earlier paper draft.
- It does not include full raw solver logs, large CNF files, or generated matrix dumps in git.
- It does not claim model independence; multi-model robustness requires separate experiments.

## LangSmith Tracing

SolEvolve records deterministic run metadata without uploading large CNF/model payloads. When LangSmith is configured, trace nodes include:

- `solevolve.run`
- `solevolve.generator`, `solevolve.evolver`, `solevolve.verifier`, `solevolve.reflector`
- `solevolve.run_solver`, `solevolve.verify_code`, `solevolve.artifact_write`

Metadata includes the artifact directory, git SHA, thread id, tags, and optional `SOLEVOLVE_PAPER_CLAIM_ID`.

See `docs/langsmith.md` for trace inspection commands.

## Optional SAT Solvers

SAT smoke tests are optional and only run when a solver path exists.

```bash
export KISSAT_PATH=./kissat/build/kissat
export CADICAL_PATH=/path/to/cadical
```

If these variables are not set, the code checks common local build paths and then falls back to `PATH`.

## Development

```bash
python -m pip install -e ".[dev]"
python -m py_compile $(find src/solevolve -name '*.py')
pytest
solevolve-demo --dry-run --max-turns 1
solevolve-solver-smoke
solevolve-artifact-check
```

The CI path intentionally avoids OpenRouter, LangSmith, Kissat, and CaDiCaL credentials so public contributors can run it without private infrastructure.

## Citation

```bibtex
@article{solevolve2026,
  title={{SolEvolve: LLM-Driven Evolutionary Search for SAT-Verified Constructions in Combinatorics}},
  author={Kim, Jon-Lark and Baek, Jae-Hyun},
  journal={Neurocomputing},
  year={2026}
}
```
