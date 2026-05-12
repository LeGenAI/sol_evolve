# LangSmith Tracing

SolEvolve uses LangChain/LangGraph tracing when LangSmith is configured and a no-op fallback otherwise.

## Local setup

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=solevolve-repro
export LANGCHAIN_CALLBACKS_BACKGROUND=false
export SOLEVOLVE_STRICT_TRACING=false
export SOLEVOLVE_SOLVER=cadical
```

For reviewer acceptance without LLM or LangSmith, use the deterministic report command instead:

```bash
solevolve-reviewer-reproduce --paper-claim-id all_reviewer_core --cadical-path /path/to/cadical
```

Run an LLM-backed trace:

```bash
export OPENROUTER_API_KEY=...
solevolve-demo "Goal: outline CNF encoding steps for [32,14,8]."
```

Run a direct paper-result reproduction trace with CaDiCaL:

```bash
export CADICAL_PATH=/path/to/cadical
solevolve-reproduce-paper-claims --paper-claim-id ternary_bch_d9 --timeout 300
```

Every `solevolve-demo` trace starts with deterministic repro evidence:

- `solevolve.paper_input` resolves the manuscript-facing target specification from `--paper-claim-id` or `--target-json`.
- `solevolve.repro_check` coordinates public artifact and solver checks.
- `solevolve.artifact_check` validates the artifact manifest.
- `solevolve.solver_check` records solver availability as `OK`, `ERROR`, or `SKIPPED`, with CaDiCaL checked first by default.
- When a paper claim is set, the LLM Evolver emits a schema action and the deterministic `evolver_action` node runs `solevolve.bounds_lookup`, codetables lookup, and paper-claim proof tools.
- `solevolve.paper_output` emits the reviewer-facing target, codetables results, solver runs, verifier diagnostics, Reflector action, decision, verdict, missing obligations, and artifact paths.

The direct paper-claim reproduction command avoids LangGraph node spans and records a compact hierarchy:

- `solevolve.reproduce_paper_claims`
- `solevolve.reproduce_paper_claim`
- `solevolve.so_claim_sat`
- `solevolve.lucas_verify:<claim_id>`
- `solevolve.binary_verify:binary_ad_43_10_16`

It verifies the explicit matrix, Gram-rank lower bound, self-orthogonality, and weight distribution for selected SO claims. For `ternary_bch_d9`, it SAT-encodes all GF(3)^7 nonzero-message weight constraints for d>=9 and checks d>=10 as UNSAT, giving separate CaDiCaL spans for existence and optimality. For GF(4)/GF(5) rows, explicit matrix verification is mandatory and d+1 SAT/UNSAT evidence is attempted under the configured timeout. Lucas and binary reviewer suites use deterministic verifier spans that record compact counts, distributions, artifact paths, hashes, and elapsed time.

For a full LLM trace with the solver proof under the same root:

```bash
export SOLEVOLVE_STRICT_TRACING=true
export SOLEVOLVE_PAPER_CLAIM_ID=ternary_bch_d9
export SOLEVOLVE_PAPER_CLAIM_TIMEOUT=300
export CADICAL_PATH=/path/to/cadical
solevolve-demo --max-turns 4 "Goal: assess the ternary BCH d=9 table claim from deterministic SAT evidence."
```

The graph is centrally gated by deterministic coordinator logic. The default graph does not allow free LLM tool-calling; the Evolver emits a validated `EvolverAction`, then the coordinator runs whitelisted codetables/proof tools. LLM role calls receive evidence as a provider-compatible single user handoff. They should not report raw CNF, model, center-set, or matrix claims as reproduced unless deterministic evidence supports them.

For explicit paper-shaped input without using a registry id:

```bash
solevolve-demo \
  --target-json '{"q":3,"n":20,"k":7,"d":9,"field":"GF(3)","inner_product":"dot","objective":"reproduce ternary BCH d=9","required_checks":["explicit_matrix_verification","sat_feasibility_minimum_distance_target"],"solver_preference":"cadical"}' \
  --max-turns 4 \
  "Goal: assess a paper target from structured input."
```

Coordinator metadata includes `decision`, `mode`, `verdict`, `thresholds_applied`, and `evidence_used`:

- `PARTIAL`: evidence exists, but reproduction is incomplete.
- `SKIPPED`: an optional dependency such as a SAT solver was unavailable.
- `INSUFFICIENT_ARTIFACT`: raw artifact evidence is missing.
- `UNAVAILABLE`: an optional backend is not bundled.

`solevolve.run_python` and `solevolve.fetch_url` are manual-only tools. Their schemas document local execution and network-fetch risks, and they are not part of the default repro graph.

For LLM repro runs that must fail closed when traces are unavailable:

```bash
export SOLEVOLVE_STRICT_TRACING=true
solevolve-demo --max-turns 4 "Goal: assess the ternary BCH d=9 table claim from deterministic SAT evidence."
```

## Inspect traces

```bash
langsmith trace list --project solevolve-repro --limit 10 --include-metadata --api-key "$LANGSMITH_API_KEY"
langsmith trace list --project solevolve-repro --limit 5 --show-hierarchy --api-key "$LANGSMITH_API_KEY"
langsmith trace export artifacts/langsmith_traces --project solevolve-repro --limit 20 --full --api-key "$LANGSMITH_API_KEY"
```

## Metadata policy

Traces include run parameters, solver status, timing, artifact paths, hashes, and concise verifier summaries. Lucas and binary traces intentionally omit raw center sets and generator matrices; only counts, distributions, `A_d` values, and artifact hashes are uploaded. All `traceable_run` inputs, outputs, and metadata pass through a shared sanitizer before upload:

- API keys, bearer tokens, passwords, secrets, and known LangSmith/OpenRouter key patterns are redacted.
- LangChain messages are represented as bounded content previews plus length and SHA-256 hashes.
- LLM clients, runnable chains, NumPy-like arrays, large strings, CNF/model/matrix payloads, and unknown objects are summarized instead of uploaded raw.

Solver metadata is intentionally structured for filtering. `solevolve.solver_check` includes `preferred_solver`, `selected_solver`, `selected_path`, `selected_source`, `cwd`, `argv`, `command`, candidate `exists`/`executable` flags, `elapsed_ms`, and `version_elapsed_ms` when a binary is found. Root run metadata also mirrors `solver_backend`, `solver_path`, `solver_command`, `solver_cwd`, `solver_version`, and `solver_check_elapsed_ms`. `solevolve.run_solver` and `solevolve.run_sat_search` add run metadata under `solver`, plus flat keys `solver_backend`, `solver_path`, `solver_elapsed_ms`, and `solver_total_elapsed_ms`.

Paper-claim metadata is mirrored separately from solver availability metadata. Root summaries include `paper_input`, `paper_output`, `paper_claim_execution`, `claim_results`, `codetables_refs`, `paper_claim_verdict`, `paper_claim_solver_command`, `paper_claim_solver_elapsed_ms`, `paper_claim_optimality_command`, and `paper_claim_optimality_elapsed_ms`. These fields refer to actual CNF solves, while `solver_execution` remains only the availability/version check. codetables traces store URL, parsed bounds, timestamp/hash, and cache path, not raw HTML. `paper_output.solver_runs` records actual solver argv/cwd/path, CNF path/hash, SAT status, return code, elapsed wall time, timeout, and memory when available.

LLM cost is recorded as an estimate in metadata because LangSmith's native `costs` field may stay empty for OpenRouter model aliases. SolEvolve first uses `SOLEVOLVE_PROMPT_COST_PER_1M` and `SOLEVOLVE_COMPLETION_COST_PER_1M` when set; otherwise it tries OpenRouter's public `/models` pricing metadata. Agent invoke runs include `llm_cost` and `llm_estimated_cost_usd`; the root run includes `estimated_total_cost_usd`, `llm_prompt_tokens`, `llm_completion_tokens`, `llm_total_tokens`, `cost_currency`, and `cost_pricing_sources`.

By default, missing LangSmith configuration leaves tracing as a no-op for deterministic reviewer reproduction. With `SOLEVOLVE_STRICT_TRACING=true`, disabled tracing, missing `LANGSMITH_API_KEY`, LangSmith SDK setup failure, or flush failure raises an error.
