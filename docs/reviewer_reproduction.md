# Reviewer Reproduction

This is the reviewer-facing path for the revised manuscript. It is deterministic and does not require OpenRouter or LangSmith. The command is the public acceptance surface; tests are intentionally outside the public artifact scope.

## Full Command

```bash
export CADICAL_PATH=/path/to/cadical
solevolve-reviewer-reproduce \
  --paper-claim-id all_reviewer_core \
  --cadical-path "$CADICAL_PATH" \
  --timeout 300
```

`--cadical-path` may point either to a binary or to a CaDiCaL source/build directory. The resolver checks `build/cadical` and `cadical` inside that directory. If omitted, the command also checks the repo-local `cadical/build/cadical`, then common environment/default locations.

The command writes:

- `artifacts/reviewer_reproduction/reviewer_reproduction_summary.json`
- `artifacts/reviewer_reproduction/REPRODUCTION_REPORT.md`
- per-claim reports under `artifacts/reviewer_reproduction/paper_claims/`, `lucas_cubes/`, and `binary_codes/`
- sanitized codetables cache entries under `artifacts/reviewer_reproduction/codetables/`

The public input bundle is intentionally minimal:

- `artifacts/manifest.json`
- `artifacts/reviewer_core_claims.json`
- `artifacts/reviewer_experiment_summaries.json`
- `artifacts/reviewer_experiment_summaries.csv`

Runtime reports, LangSmith exports, CNFs, solver models, generated NumPy files, and tests are excluded from the release bundle.

## What Is Checked

The reviewer command checks the public artifact manifest, hashes bundled files, verifies the CaDiCaL binary when SAT-backed SO claims are selected, queries codetables.de where applicable, and reproduces the registered paper rows:

| Claim | Required deterministic checks |
| --- | --- |
| `ternary_bch_d9` | GF(3) explicit matrix, self-orthogonality, weight distribution, d=9 SAT, d>=10 UNSAT |
| `gf4_hermitian` | GF(4) Hermitian explicit matrix, self-orthogonality, weight distribution, d=6 SAT; d>=7 attempted under timeout |
| `gf5_so` | GF(5) explicit matrix, self-orthogonality, weight distribution, d=8 SAT; d>=9 attempted under timeout |
| `lucas_l7_s4` | Generate `Lambda_7(1^4)`, verify 15 centers, pairwise distance >= 3, exact closed-neighborhood cover of 99 vertices, and ball-size distribution `{6:7,7:7,8:1}` |
| `lucas_l15_s12` | Verify the bundled 2047-center artifact from the archived SAT experiment: pairwise distance >= 3, exact closed-neighborhood cover of 32707 vertices, and ball-size distribution `{14:11,15:23,16:2013}` |
| `lucas_l15_s11` | Verify the bundled 2047-center artifact from the archived SAT experiment: pairwise distance >= 3, exact closed-neighborhood cover of 32647 vertices, and ball-size distribution `{14:16,15:73,16:1958}` |
| `binary_ad_43_10_16` | Verify three archived generator matrices as binary `[43,10,16]` codes and recompute `A_16=91,86,80` by enumerating all `2^10` codewords |
| `binary_22_11_7_hybrid_ga` | Verify the archived Hybrid SAT-GA public witness as a binary `[22,11,7]` code and recompute `A_7=176` and the full weight distribution |

`PASS` means the selected claims passed required proof obligations and required codetables lookup succeeded live or from cache. `PARTIAL` means some deterministic checks pass but a selected suite contains an unavailable optional proof, timeout, or missing artifact. `INSUFFICIENT_ARTIFACT` on a claim means the release lacks the raw center/matrix artifact needed for a full proof. `FAIL` means a required manifest or mathematical proof obligation contradicted the manuscript claim.

Useful focused commands:

```bash
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode archived --require-pass
solevolve-demo --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode frontier_repair --hybrid-ga-frontier-distance 6 --hybrid-ga-frontier-seed-count 20 --max-turns 4
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes
solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path "$CADICAL_PATH" --require-pass
```

`all_reviewer_core --require-pass` is intentionally strict: it fails if any required SO, Lucas, or binary claim obligation fails. The bundled Lucas `n=15` center sets are now included in `artifacts/reviewer_core_claims.json`; no external Lucas center archive is needed for the reviewer command.

The Lucas `Lambda_7(1^4)` verifier recomputes ball sizes inside the induced Lucas cube. For the manuscript center list, the reproducible distribution is `{6:7,7:7,8:1}`; any paper text reporting a different distribution should be corrected to this verifier output.

Hybrid SAT-GA has four public CLI modes. `archived` is reviewer-safe and verifies the bundled final witness without running GA or SAT. `replay` reruns the archived deterministic `d=6 -> d=7` repair replay. `frontier_repair` tries target-distance SAT first, then builds a `d-1` seed bank, initializes GA from those seeds, and attempts low-weight-support SAT repair. `live` is exploratory and may return `PARTIAL` if CaDiCaL-backed seed/repair work is unavailable or times out. The LangGraph path only executes this branch when `--hybrid-ga-mode` is not `off` and the Evolver emits a validated `construction_search` action with `method="hybrid_sat_ga"`. The Evolver may not change the CLI-owned mode; a mismatched JSON `mode` is rejected as repair evidence.

## Compact Experiment Summaries

`artifacts/reviewer_experiment_summaries.json` and the flattened CSV companion expose the reviewer-facing numerical summaries without bundling raw private logs:

- `[22,11,7]` repair ablation: `1/50`, `2/50`, and `5/50` successes with Wilson intervals.
- `[35,10,12]` structural-diversity reanalysis: 68 verified `A_12` values with min/mean/median/max/population-std summary.
- `[32,14,8]` construction context: the printed manuscript matrix witness, an additional CaDiCaL archive witness, solver-log timing for that companion witness, and Magma/random-baseline provenance.
- 2026-05-12 Magma BKLC snapshot: `A_8=377` for `[32,14]`, with the previous `72` value identified as `A_22`.
- Lucas L7 distribution audit: selected induced-graph ball distribution `{6:7,7:7,8:1}` and the rejected older distribution whose ball-size sum is 107 rather than 99.

The `[32,14,8]` summary intentionally separates the manuscript matrix from the CaDiCaL archive matrix. Both verify as valid `[32,14,8]` witnesses, but the CaDiCaL solver elapsed time belongs to the companion archive witness, not to the exact printed manuscript matrix.

## Contributor Checks

The reviewer command above is the acceptance path. Contributor checks should stay focused on package integrity and deterministic proof code:

```bash
python -m py_compile $(find src/solevolve -name '*.py')
solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path /path/to/cadical --require-pass
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id binary_22_11_7_hybrid_ga --hybrid-ga-mode archived --require-pass
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes
```
