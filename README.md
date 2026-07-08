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
python -m pip install -e .
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

No API keys are needed for reviewer reproduction.

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
SOLEVOLVE_STRICT_TRACING=false
```

Solver preference defaults to CaDiCaL:

```bash
SOLEVOLVE_SOLVER=cadical
CADICAL_PATH=/path/to/cadical
```

## Reviewer Reproduction

The reviewer-facing command is:

```bash
export CADICAL_PATH=/path/to/cadical
solevolve-reviewer-reproduce \
  --paper-claim-id all_reviewer_core \
  --cadical-path "$CADICAL_PATH" \
  --timeout 300
```

`--cadical-path` can be either a binary or a directory. If it is omitted, the resolver checks the repo-local `cadical/build/cadical` before common environment/default locations.

It writes `artifacts/reviewer_reproduction/reviewer_reproduction_summary.json` and `artifacts/reviewer_reproduction/REPRODUCTION_REPORT.md`. The command verifies the public manifest, hashes the bundled reviewer artifact, queries codetables.de where applicable, and runs deterministic verifiers for the SO GF(k), Lucas-cube, and binary `A_d` claims.

For strict acceptance of only fully bundled claims, use:

```bash
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes --require-pass
solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path "$CADICAL_PATH" --require-pass
```

`lucas_cubes` now verifies the bundled `Lambda_7(1^4)`, `Lambda_15(1^11)`, and `Lambda_15(1^12)` center sets directly from `artifacts/reviewer_core_claims.json`.

See `docs/reviewer_reproduction.md` for the reviewer acceptance contract.

## Supported Commands

The public package exposes these supported commands:

```bash
solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path /path/to/cadical --require-pass
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode archived --require-pass
solevolve-reviewer-reproduce --paper-claim-id cegar_eager_scaling --cadical-path /path/to/cadical
solevolve-reviewer-reproduce --paper-claim-id all_reviewer_core --cadical-path /path/to/cadical
solevolve-reproduce-ternary-bch --cadical-path /path/to/cadical
solevolve-reproduce-paper-claims --paper-claim-id ternary_bch_d9
solevolve-demo --paper-claim-id ternary_bch_d9 --max-turns 4 "Goal: assess the ternary BCH d=9 table claim from deterministic SAT evidence."
```

- `solevolve-reviewer-reproduce`: runs the deterministic reviewer reproduction bundle and writes JSON/Markdown acceptance reports for SO, Lucas, binary, and the archived Hybrid SAT-GA `[22,11,7]` claim.
- `cegar_eager_scaling`: optional long-running reviewer claim group for the verifier-feedback CEGAR versus eager-encoding scaling experiment on SO `[52,26]` and binary `[43,10]`; it is not part of `all_reviewer_core`.
- `solevolve-reproduce-ternary-bch`: deterministically reproduces the ternary BCH `[13,7,5]_3` self-orthogonal embedding claim with CaDiCaL SAT feasibility plus direct GF(3) witness verification.
- `solevolve-reproduce-paper-claims`: uses the paper claim registry for `ternary_bch_d9`, `gf4_hermitian`, `gf5_so`, or `all_so_table`; it verifies explicit matrices, runs SAT feasibility when CaDiCaL is available, and attempts the d+1 obligation under the configured timeout.
- `solevolve-demo`: runs the traceable LLM graph. It requires `OPENROUTER_API_KEY`; reviewer acceptance should use `solevolve-reviewer-reproduce`.

`solevolve-demo` starts by resolving the paper-style input contract (`paper_input`) from `--paper-claim-id` or `--target-json`, then records artifact manifest status and optional solver availability before any LLM role runs. `--max-turns` is the strict maximum number of agent invocations after setup; reaching the cap stops the graph before the next role is called. The default graph does not let LLMs freely call tools. The LLM Evolver emits a validated `EvolverAction`; the coordinator then executes whitelisted codetables/SAT proof actions or, when enabled by `--hybrid-ga-mode`, the gated Hybrid SAT-GA branch, and writes the reviewer-facing `paper_output`.

Coordinator verdicts use these statuses:

- `PARTIAL`: evidence was collected, but table-level reproduction is incomplete.
- `SKIPPED`: an optional environmental dependency, such as a SAT solver, is unavailable.
- `INSUFFICIENT_ARTIFACT`: raw CNF/model/matrix/checksum evidence is missing.
- `UNAVAILABLE`: an optional backend, such as the SO embedding backend, is not bundled.

## Reproducibility Layout

- `src/solevolve/agents`: Generator, Evolver, Verifier, and Reflector agent wrappers.
- `src/solevolve/engines`: SAT/CNF encoding and solver interfaces for binary code search.
- `src/solevolve/claim_registry.py`: machine-readable paper claim data for the ternary BCH, GF(4), and GF(5) self-orthogonal rows.
- `src/solevolve/lucas_repro.py`, `src/solevolve/binary_repro.py`, and `src/solevolve/hybrid_ga.py`: deterministic verifiers for Lucas perfect partitions, binary `[43,10,16]` `A_16` progression, and the coordinator-gated Hybrid SAT-GA `[22,11,7]` public witness.
- `artifacts/reviewer_core_claims.json`: the minimal public reviewer evidence bundle for Lucas, binary-code, and archived Hybrid SAT-GA claims.
- `artifacts/reviewer_experiment_summaries.json` and `artifacts/reviewer_experiment_summaries.csv`: compact public summaries for the `[22,11,7]` repair ablation, `[35,10,12]` `A_12` diversity reanalysis, `[32,14,8]` construction/BKLC baseline context, the 2026-05-12 Magma BKLC snapshot, and the Lucas L7 ball-size audit.
- `artifacts/manifest.json`: public manifest describing which evidence is bundled and which raw logs should be released separately.

## Public Scope

The public release scope is deliberately narrow:

- included: package source, CLI entrypoints, docs, `artifacts/manifest.json`, `artifacts/reviewer_core_claims.json`, and `artifacts/reviewer_experiment_summaries.{json,csv}`
- excluded: tests, runtime reports, LangSmith exports, CNFs, solver models, generated NumPy files, private logs, and scratch experiment drivers

Large CNF files, NumPy matrices, solver outputs, private raw console logs, runtime traces, test folders, and long-form exploratory drivers are intentionally excluded from the public artifact scope. The public bundle is intentionally small: `artifacts/manifest.json`, `artifacts/reviewer_core_claims.json`, and `artifacts/reviewer_experiment_summaries.{json,csv}`. The Lucas `n=15` center sets are bundled as compact bitstring lists inside `reviewer_core_claims.json`; the binary-code experiment summaries keep only hashes, counts, weight-distribution summaries, timing provenance, and the snapshot-dated Magma BKLC values used for manuscript comparisons. Raw SAT logs and NumPy copies remain excluded. Use `SOLEVOLVE_ARTIFACT_DIR` to point runtime commands to local or downloaded artifacts.

Hybrid SAT-GA is opt-in in the LangGraph path. Reviewer-safe verification uses only the archived public witness:

```bash
solevolve-reviewer-reproduce --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode archived --require-pass
solevolve-demo --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode replay --max-turns 4 "Goal: reproduce the binary [22,11,7] Hybrid SAT-GA claim."
solevolve-demo --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode frontier_repair --hybrid-ga-frontier-distance 6 --hybrid-ga-frontier-seed-count 20 --max-turns 4 "Goal: run frontier seed-bank Hybrid SAT-GA repair for [22,11,7]."
SOLEVOLVE_CEGAR_EAGER_SMOKE=1 solevolve-demo --paper-claim-id cegar_eager_scaling --max-turns 4 "Goal: run the Agent-loop smoke path for the CEGAR/eager scaling claims."
```

`archived` verifies the bundled final matrix as a `[22,11,7]` code with `A_7=176`. `replay` reruns the deterministic archived `d=6 -> d=7` repair replay. `frontier_repair` first tries direct target-distance SAT, then falls back to a `d-1` seed bank, GA population search, and low-weight-support SAT repair. `live` is marked exploratory and may return `PARTIAL` if CaDiCaL repair/seed work times out.

See `docs/reproducibility.md` for the paper-code checklist, included/excluded artifact policy, and benchmarked open-source release standards used for this cleanup.

## What This Release Does Not Include

- It does not include private `.env` files or API keys.
- It does not include the stale manuscript source from the earlier paper draft.
- It does not include full raw solver logs, large CNF files, or generated matrix dumps in git.
- It does not claim model independence; multi-model robustness requires separate experiments.

## LangSmith Tracing

SolEvolve records deterministic run metadata without uploading raw API keys, client objects, large CNF payloads, solver models, or matrix dumps. When LangSmith is configured, trace nodes include:

- `solevolve.run`
- `solevolve.paper_input`, `solevolve.repro_check`, `solevolve.artifact_check`, `solevolve.solver_check`
- `solevolve.generator`, `solevolve.evolver`, `solevolve.verifier`, `solevolve.reflector`
- `solevolve.bounds_lookup`, codetables lookup, paper-claim SAT proof, solver, `solevolve.paper_output`, fetch, Python execution, and artifact-write tools
- `solevolve.hybrid_ga`, `hybrid_ga:archived_verify`, `hybrid_ga:frontier_target_sat`, `hybrid_ga:frontier_seed_bank`, `hybrid_ga:init_population`, `hybrid_ga:init_from_seed_bank`, `hybrid_ga:evolve`, `hybrid_ga:sat_repair`, `hybrid_ga:final_verify`, and `hybrid_ga:report` when `--hybrid-ga-mode` is enabled

`solevolve.run_python` and `solevolve.fetch_url` are manual-only tools. They remain traced and documented but are not used by the default repro graph.

Trace inputs, outputs, and metadata are sanitized before upload. Message content is recorded as a bounded preview plus length and SHA-256 hash. Set `SOLEVOLVE_STRICT_TRACING=true` for repro runs where missing credentials, disabled tracing, SDK setup failure, or flush failure should fail the command instead of falling back to no-op tracing.

Solver traces record the requested solver, selected solver, selected binary path, source (`CADICAL_PATH`, repo default, or `PATH`), cwd, argv/command, candidate checks, version-check latency, solver execution time, and total tool elapsed time. The default solver preference is `cadical`; set `SOLEVOLVE_SOLVER=kissat` only when you want Kissat checked first.

LLM cost is added as estimated metadata for OpenRouter-backed runs. Set `SOLEVOLVE_PROMPT_COST_PER_1M` and `SOLEVOLVE_COMPLETION_COST_PER_1M` to override pricing; otherwise SolEvolve tries OpenRouter's public model pricing metadata. Root traces include `estimated_total_cost_usd`, and each `solevolve.agent.invoke` includes `llm_cost`.

For a direct paper-result reproduction trace, run:

```bash
SOLEVOLVE_STRICT_TRACING=true \
CADICAL_PATH=/path/to/cadical \
solevolve-reproduce-paper-claims --paper-claim-id ternary_bch_d9 --timeout 300
```

This produces `artifacts/paper_claims/ternary_bch_d9/reproduction_report.json` and traces `solevolve.reproduce_paper_claim` with nested `solevolve.so_claim_sat` solver spans. The default run SAT-encodes self-orthogonality plus all GF(3)^7 nonzero-message weight constraints for d>=9, then checks d>=10 as UNSAT to support d=9 optimality at t=7.

To put the same proof under a full LLM graph trace, set:

```bash
export SOLEVOLVE_PAPER_CLAIM_ID=ternary_bch_d9
export SOLEVOLVE_PAPER_CLAIM_TIMEOUT=300
solevolve-demo --paper-claim-id ternary_bch_d9 --max-turns 4 "Goal: assess the ternary BCH d=9 table claim from deterministic SAT evidence."
```

That run records `solevolve.run -> LangGraph -> paper_input -> repro_check -> generator -> evolver -> evolver_action`, where the validated Evolver action triggers `solevolve.bounds_lookup`, `solevolve.codetables_lookup`, `solevolve.reproduce_paper_claims`, and `solevolve.so_claim_sat`, followed by Verifier/Reflector spans, `paper_output`, and final cost metadata.

The `last_run_summary.json` top-level fields `paper_input` and `paper_output` mirror the manuscript workflow: target specification, codetables query, solver budget, verifier diagnostics, solver runs, Reflector action, coordinator decision, verdict, missing obligations, and artifact paths. Raw CNF/model/matrix bodies are not uploaded to LangSmith; summaries include paths, hashes, counts, statuses, and timings.

See `docs/langsmith.md` for trace inspection commands.

## Optional SAT Solvers

SAT proof obligations run when a solver path exists.

```bash
export CADICAL_PATH=/path/to/cadical
export KISSAT_PATH=./kissat/build/kissat
export SOLEVOLVE_SOLVER=cadical
```

If these variables are not set, the code checks common local build paths and then falls back to `PATH`.

## Development

```bash
python -m pip install -e .
python -m py_compile $(find src/solevolve -name '*.py')
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes
```

The reviewer acceptance command remains `solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path /path/to/cadical --require-pass`.

## Citation

```bibtex
@article{solevolve2026,
  title={{SolEvolve: LLM-Driven Evolutionary Search for SAT-Verified Constructions in Combinatorics}},
  author={Kim, Jon-Lark and Baek, Jae-Hyun},
  journal={Neurocomputing},
  year={2026}
}
```
