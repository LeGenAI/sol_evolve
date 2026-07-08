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

The reviewer command is the acceptance path. It checks the artifact manifest, hashes the bundled reviewer artifacts, verifies solver availability where needed, queries codetables.de with sanitized cache fallback where applicable, and runs deterministic paper-claim reproduction for the self-orthogonal table rows, Lucas perfect partitions, binary `[43,10,16]` `A_16` progression, and the archived Hybrid SAT-GA `[22,11,7]` witness when selected. The optional `cegar_eager_scaling` claim group runs the heavier verifier-feedback CEGAR versus eager-encoding suite for SO `[52,26]` and binary `[43,10]`; it is intentionally outside `all_reviewer_core`. The demo graph remains useful for LangSmith/LLM tracing: it first resolves a paper-style `paper_input` from `--paper-claim-id` or `--target-json`, records deterministic repro checks, and ends with a reviewer-facing `paper_output`. Central coordinator logic owns tool policy, routing, fallback verdicts, and threshold decisions. LLM roles do not freely call tools in the default graph. The LLM Evolver emits a validated `EvolverAction`, and the coordinator executes only whitelisted codetables/proof actions or the opt-in Hybrid SAT-GA branch. Missing SAT solvers are represented as `SKIPPED`, missing raw table-level evidence as `INSUFFICIENT_ARTIFACT`, and optional unbundled backends as `UNAVAILABLE`. The public code directly verifies the explicit GF(3)/GF(4)/GF(5) matrices and weight distributions, the Lucas `Lambda_7(1^4)`, `Lambda_15(1^11)`, and `Lambda_15(1^12)` center-set partitions, the binary `A_16=91,86,80` progression, and the Hybrid SAT-GA `[22,11,7]` archived witness with `A_7=176`. Compact JSON/CSV experiment summaries also cover the `[22,11,7]` repair ablation, `[35,10,12]` `A_12` diversity reanalysis, `[32,14,8]` construction context, the 2026-05-12 Magma BKLC snapshot, and Lucas L7 distribution audit.

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
    "solver_runs": [{"claim_id": "ternary_bch_d9", "status": "SAT", "elapsed_ms": 1234.5}],
    "diagnostics": [{"claim_id": "ternary_bch_d9", "d_min": 9, "self_orthogonal": true}],
    "hybrid_ga_reports": [],
    "verdict": "PASS",
    "missing_obligations": []
  }
}
```

Hybrid SAT-GA fields are represented in the same contracts as optional population/verifier state. The public reviewer command verifies the archived `[22,11,7]` witness; LangGraph executes the branch only when the CLI enables it and the Evolver emits a validated `construction_search` action with `method="hybrid_sat_ga"`. The Evolver JSON `mode` must be null or exactly match the CLI-owned `--hybrid-ga-mode`; mismatches are recorded as repair evidence and are not executed. The `frontier_repair` mode is the general workflow for target-distance SAT failures: it tries the target distance, stores a `d-1` seed bank, initializes GA from those seeds, and attempts low-weight-support SAT repair.

```bash
solevolve-reviewer-reproduce --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode archived --require-pass
solevolve-demo --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode replay --max-turns 4
solevolve-demo --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode frontier_repair --hybrid-ga-frontier-distance 6 --hybrid-ga-frontier-seed-count 20 --max-turns 4
solevolve-reviewer-reproduce --paper-claim-id cegar_eager_scaling --cadical-path /path/to/cadical --timeout 300
SOLEVOLVE_CEGAR_EAGER_SMOKE=1 solevolve-demo --paper-claim-id cegar_eager_scaling --claim-timeout-sec 1 --max-turns 4
```

## Public surface policy

The main branch is a reproducible scaffold, not a dump of exploratory research notebooks and one-off drivers. Long-form scripts, archived experiment controllers, interactive HITL demos, full solver logs, generated matrices, and stale manuscript source are excluded from the public package. If table-level reproduction requires those assets, they should be published as release assets, Zenodo files, or a raw artifact archive with checksums listed in `artifacts/manifest.json`.

## Required before paper artifact link

- Run focused reviewer commands for `all_so_table`, `binary_ad_43_10_16`, and `lucas_cubes`, then archive their JSON/Markdown reports.
- Publish any large generated CNFs/logs only as external release assets if journal policy requires raw solver logs.
- For every table beyond the self-orthogonal rows, list the exact command, inputs, expected outputs, number of runs, timeout, solver version, and compute environment.
- Keep claims in README synchronized with the revised manuscript. Avoid "first", "novel", "fully automated", or model-robustness claims unless the artifact provides direct evidence.

## Recommended next pass

- If a raw artifact archive is released later, add checksums for full binary-code SAT logs and matrix dumps without moving those large files into git.
- Add a Dockerfile or devcontainer after the Python package path is stable.
- Add deterministic seed support to solver orchestration where the backend exposes seed controls.
- Add `docs/artifact_manifest_schema.md` once the external archive format is finalized.
