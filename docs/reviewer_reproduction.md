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

Runtime reports, LangSmith exports, CNFs, solver models, generated NumPy files, and tests are excluded from the release bundle.

## What Is Checked

The reviewer command checks the public artifact manifest, hashes bundled files, verifies the CaDiCaL binary when SAT-backed SO claims are selected, queries codetables.de where applicable, and reproduces the registered paper rows:

| Claim | Required deterministic checks |
| --- | --- |
| `ternary_bch_d9` | GF(3) explicit matrix, self-orthogonality, weight distribution, d=9 SAT, d>=10 UNSAT |
| `gf4_hermitian` | GF(4) Hermitian explicit matrix, self-orthogonality, weight distribution, d=6 SAT; d>=7 attempted under timeout |
| `gf5_so` | GF(5) explicit matrix, self-orthogonality, weight distribution, d=8 SAT; d>=9 attempted under timeout |
| `lucas_l7_s4` | Generate `Lambda_7(1^4)`, verify 15 centers, pairwise distance >= 3, exact closed-neighborhood cover of 99 vertices, and ball-size distribution `{6:7,7:7,8:1}` |
| `lucas_l15_s12` | Requires public 2047-center artifact or deterministic construction source; missing artifact reports `INSUFFICIENT_ARTIFACT` |
| `lucas_l15_s11` | Requires public 2047-center artifact or deterministic construction source; missing artifact reports `INSUFFICIENT_ARTIFACT` |
| `binary_ad_43_10_16` | Verify three archived generator matrices as binary `[43,10,16]` codes and recompute `A_16=91,86,80` by enumerating all `2^10` codewords |

`PASS` means the selected claims passed required proof obligations and required codetables lookup succeeded live or from cache. `PARTIAL` means some deterministic checks pass but a selected suite contains an unavailable optional proof, timeout, or missing artifact. `INSUFFICIENT_ARTIFACT` on a claim means the release lacks the raw center/matrix artifact needed for a full proof. `FAIL` means a required manifest or mathematical proof obligation contradicted the manuscript claim.

Useful focused commands:

```bash
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes
solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path "$CADICAL_PATH" --require-pass
```

`all_reviewer_core --require-pass` is intentionally strict: it will fail until the Lucas `n=15` center artifacts are included.

The Lucas `Lambda_7(1^4)` verifier recomputes ball sizes inside the induced Lucas cube. For the manuscript center list, the reproducible distribution is `{6:7,7:7,8:1}`; any paper text reporting a different distribution should be corrected to this verifier output.

## Contributor Checks

The reviewer command above is the acceptance path. Contributor checks should stay focused on package integrity and deterministic proof code:

```bash
python -m py_compile $(find src/solevolve -name '*.py')
solevolve-reviewer-reproduce --paper-claim-id all_so_table --cadical-path /path/to/cadical --require-pass
solevolve-reviewer-reproduce --paper-claim-id binary_ad_43_10_16 --require-pass
solevolve-reviewer-reproduce --paper-claim-id lucas_cubes
```
