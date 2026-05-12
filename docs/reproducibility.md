# Reproducibility Checklist

This checklist aligns SolEvolve with current research-code release expectations for LLM-guided search systems and ML reproducibility checklists.

## External release patterns used as references

- OpenEvolve: package install, development install, examples, deterministic seeds, Docker option, cost/provider notes, and citation metadata.
- FunSearch: explicit per-domain artifact folders and a clear statement about which production components are not included.
- ML reproducibility checklist: dependency specification, source-code link, data/artifact availability, exact run counts, metrics, error bars, and compute environment.

## Current compliance target

| Area | SolEvolve status |
| --- | --- |
| Installable code | `pyproject.toml`, `requirements.txt`, and console scripts |
| Reviewer command | `solevolve-reviewer-reproduce --paper-claim-id all_reviewer_core --cadical-path /path/to/cadical` |
| Dependencies | pinned LangChain/LangGraph stack plus NumPy, LangSmith, dotenv, and requests |
| Artifact manifest | `artifacts/manifest.json` |
| Raw artifact policy | large CNF/model/matrix/log files excluded from git and referenced via `SOLEVOLVE_ARTIFACT_DIR` |
| Deterministic metadata | git SHA, thread id, artifact dir, preferred solver, solver timing, tags, and paper claim id recorded in run summaries/traces |
| Traceability | LangSmith optional; no-op when keys are absent |
| Public verification | deterministic CLI reports under `solevolve-reviewer-reproduce` |

The CaDiCaL path may be either a binary or a source/build directory. When the path is omitted, the resolver checks the repo-local `cadical/build/cadical` before common environment/default locations.

The reviewer command is the acceptance path. It checks the artifact manifest, hashes the single bundled reviewer artifact, verifies solver availability where needed, queries codetables.de with sanitized cache fallback where applicable, and runs deterministic paper-claim reproduction for the self-orthogonal table rows, Lucas perfect partitions, and binary `[43,10,16]` `A_16` progression. The demo graph remains useful for LangSmith/LLM tracing: it first resolves a paper-style `paper_input` from `--paper-claim-id` or `--target-json`, records deterministic repro checks, and ends with a reviewer-facing `paper_output`. Central coordinator logic owns tool policy, routing, fallback verdicts, and threshold decisions. LLM roles do not freely call tools in the default graph. The LLM Evolver emits a validated `EvolverAction`, and the coordinator executes only whitelisted codetables/proof actions. Missing SAT solvers are represented as `SKIPPED`, missing raw table-level evidence as `INSUFFICIENT_ARTIFACT`, and optional unbundled backends as `UNAVAILABLE`. The public code directly verifies the explicit GF(3)/GF(4)/GF(5) matrices and weight distributions, the Lucas `Lambda_7(1^4)`, `Lambda_15(1^11)`, and `Lambda_15(1^12)` center-set partitions, and the binary `A_16=91,86,80` progression.

The canonical LangGraph run I/O is:

```json
{
  "paper_input": {
    "human_goal": "Goal text",
    "target": {"claim_id": "ternary_bch_d9", "q": 3, "n": 20, "k": 7, "d": 9},
    "codetables_query": {"q": 3, "n": 20, "k": 7},
    "solver_budget_sec": 300
  },
  "paper_output": {
    "solver_runs": [{"claim_id": "ternary_bch_d9", "status": "SAT", "elapsed_ms": 0.0}],
    "diagnostics": [{"claim_id": "ternary_bch_d9", "d_min": 9, "self_orthogonal": true}],
    "verdict": "PASS",
    "missing_obligations": []
  }
}
```

Hybrid SAT-GA fields are represented in the same contracts as optional population/verifier state. The public graph does not claim a fresh live GA population search unless archived population artifacts are supplied.

## Public surface policy

The main branch is a reproducible scaffold, not a dump of exploratory research notebooks and one-off drivers. Long-form scripts, archived experiment controllers, interactive HITL demos, full solver logs, generated matrices, and stale manuscript source are excluded from the public package. If table-level reproduction requires those assets, they should be published as release assets, Zenodo files, or a raw artifact archive with checksums listed in `artifacts/manifest.json`.

## Required before paper artifact link

- Run focused reviewer commands for `all_so_table`, `binary_ad_43_10_16`, and `lucas_cubes`, then archive their JSON/Markdown reports.
- Publish any large generated CNFs/logs only as external release assets if journal policy requires raw solver logs.
- For every table beyond the self-orthogonal rows, list the exact command, inputs, expected outputs, number of runs, timeout, solver version, and compute environment.
- Keep claims in README synchronized with the revised manuscript. Avoid "first", "novel", "fully automated", or model-robustness claims unless the artifact provides direct evidence.

## Recommended next pass

- Add a release asset manifest with checksums for one small binary-code SAT run when `KISSAT_PATH` exists.
- Add a Dockerfile or devcontainer after the Python package path is stable.
- Add deterministic seed support to solver orchestration where the backend exposes seed controls.
- Add `docs/artifact_manifest_schema.md` once the external archive format is finalized.
