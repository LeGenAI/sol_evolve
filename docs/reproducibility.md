# Reproducibility Checklist

This checklist aligns SolEvolve with current research-code release expectations for LLM-guided search systems and ML reproducibility checklists.

## External release patterns used as references

- OpenEvolve: package install, development install, examples, deterministic seeds, Docker option, cost/provider notes, and citation metadata.
- FunSearch: explicit per-domain artifact folders and a clear statement about which production components are not included.
- ML reproducibility checklist: dependency specification, source-code link, data/artifact availability, exact run counts, metrics, error bars, and compute environment.

## Current compliance target

| Area | SolEvolve status |
| --- | --- |
| Installable code | `pyproject.toml`, `requirements.txt`, and the three console scripts |
| Official commands | `solevolve-demo`, `solevolve-solver-smoke`, `solevolve-artifact-check` |
| No-secret smoke test | `solevolve-demo --dry-run --max-turns 1` |
| Dependencies | pinned LangChain/LangGraph stack plus NumPy, LangSmith, dotenv, pytest |
| Artifact manifest | `artifacts/manifest.json` |
| Raw artifact policy | large CNF/model/matrix/log files excluded from git and referenced via `SOLEVOLVE_ARTIFACT_DIR` |
| Deterministic metadata | git SHA, thread id, artifact dir, tags, and paper claim id recorded in run summaries/traces |
| Traceability | LangSmith optional; no-op when keys are absent |
| CI | compile/import/config/dry-run tests without private keys or solver binaries |

## Public surface policy

The main branch is a reproducible scaffold, not a dump of exploratory research notebooks and one-off drivers. Long-form scripts, archived experiment controllers, interactive HITL demos, full solver logs, generated matrices, and stale manuscript source are excluded from the public package. If table-level reproduction requires those assets, they should be published as release assets, Zenodo files, or a raw artifact archive with checksums listed in `artifacts/manifest.json`.

## Required before paper artifact link

- Publish a release asset or external archive containing raw matrices/logs needed for table-level reproduction.
- Add checksums for release artifacts to `artifacts/manifest.json`.
- For every table in the manuscript, list the exact command, inputs, expected outputs, number of runs, timeout, solver version, and compute environment.
- Keep claims in README synchronized with the revised manuscript. Avoid "first", "novel", "fully automated", or model-robustness claims unless the artifact provides direct evidence.

## Recommended next pass

- Add a release asset manifest with checksums for one small binary-code SAT run when `KISSAT_PATH` exists.
- Add a Dockerfile or devcontainer after the Python package path is stable.
- Add deterministic seed support to solver orchestration where the backend exposes seed controls.
- Add `docs/artifact_manifest_schema.md` once the external archive format is finalized.
