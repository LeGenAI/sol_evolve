# Common-Interface FunSearch/EoH Benchmark Reproduction

This document is the reviewer-facing path for the common-interface repair-policy
comparison reported in the revised manuscript (Section "Common-interface comparison
with FunSearch and EoH"). It has two tiers:

1. **Deterministic tier (recommended, no LLM access required).** Rebuild the
   benchmark, re-evaluate every fixed policy, the two frozen LLM-evolved policies,
   the oracle ceiling, and the paired statistics. All numbers in the manuscript
   table reproduce from this tier alone.
2. **LLM tier (optional).** Re-run the EoH/FunSearch evolution itself. This requires
   an OpenRouter API key and is *not* deterministic (LLM sampling); the frozen best
   policies from our runs are committed so tier 1 never depends on it.

The compact inputs and outputs are bundled under
`artifacts/external_baselines/release/` with SHA-256 checksums in
`release/release_manifest.json` (assembled by
`scripts/package_external_baselines_release.py`).

## Environment

- Python: `sol_evolve/.venv` (see `requirements.txt` / `pyproject.toml`)
- SAT solver: CaDiCaL at `cadical/build/cadical` (or pass `--cadical-path`)
- LLM (tier 2 only): OpenRouter, model pinned to `anthropic/claude-sonnet-4.6`,
  key in env var `OPENROUTER_API_KEY` (loaded from `.env`)

All commands below run from the repository root.

## Tier 1: Deterministic reproduction

### 1. Rebuild the benchmark dataset (bit-identical)

The 222 candidates are seeded corruptions (1--4 parity columns, 1 bit each,
`--seed-start 1000`) of the verified `[22,11,7]` witness from
`artifacts/reviewer_core_claims.json`:

```bash
.venv/bin/python scripts/build_repair_policy_dataset.py \
  --source-mode corrupt \
  --runs 240 \
  --seed-start 1000 \
  --corrupt-column-counts 1,2,3,4 \
  --corrupt-bits-per-column 1 \
  --min-d 5 \
  --max-d 6 \
  --train-frac 0.4 \
  --validation-frac 0.2 \
  --out-dir artifacts/external_baselines/repair_policy_dataset_corrupt_main
```

Alternatively, use the committed copy at `release/dataset/candidates.jsonl`
(train/validation/test = 88/44/90).

### 2. Fixed-policy controls on the held-out test split

```bash
.venv/bin/python scripts/evaluate_repair_policy_baselines.py \
  --dataset artifacts/external_baselines/repair_policy_dataset_corrupt_main/candidates.jsonl \
  --split test \
  --mutable-budget 4 \
  --timeout-sec 10 \
  --policies random,low_weight_support,all_parity \
  --out-dir artifacts/external_baselines/corrupt_main_test_budget4_t10
```

Expected: random 7/90, low_weight_support 0/90, all_parity (B=11 control) 90/90.

### 3. Frozen LLM-evolved policies on the held-out test split

The committed best policies are plain Python `score_columns` functions selected by
training objective; evaluating them requires no LLM access:

```bash
.venv/bin/python scripts/evaluate_generated_repair_policy.py \
  --dataset artifacts/external_baselines/repair_policy_dataset_corrupt_main/candidates.jsonl \
  --split test --mutable-budget 4 --timeout-sec 10 \
  --policy-json artifacts/external_baselines/release/policies/eoh_samples_best.json \
  --policy-name eoh_best \
  --out-dir artifacts/external_baselines/eoh_corrupt_main_test_budget4_t10

.venv/bin/python scripts/evaluate_generated_repair_policy.py \
  --dataset artifacts/external_baselines/repair_policy_dataset_corrupt_main/candidates.jsonl \
  --split test --mutable-budget 4 --timeout-sec 10 \
  --policy-json artifacts/external_baselines/release/policies/funsearch_s20_samples_best.json \
  --policy-name funsearch_best \
  --out-dir artifacts/external_baselines/funsearch_s20_corrupt_main_test_budget4_t10
```

Expected: EoH 9/90; FunSearch equals its seed template (see manuscript table).

### 4. Oracle ceiling (B=4 achievability)

The oracle reads each candidate's `metadata.corrupted_columns` and routes selection
through the identical top-B pathway:

```bash
.venv/bin/python scripts/evaluate_oracle_repair_policy.py \
  --dataset artifacts/external_baselines/repair_policy_dataset_corrupt_main/candidates.jsonl \
  --split test --mutable-budget 4 --timeout-sec 10 \
  --out-dir artifacts/external_baselines/oracle_corrupt_main_test_budget4_t10
```

Expected: 90/90 (all SAT). This certifies the bounded task admits a perfect policy,
so bounded-policy gaps reflect policy quality, not the budget.

### 5. Paired statistics (Wilson CIs, exact McNemar, Holm)

```bash
.venv/bin/python scripts/summarize_external_baseline_comparisons.py \
  --out-dir artifacts/external_baselines/corrupt_main_test_paired_stats
```

Pairs per-case outcomes by `candidate_id` across all runs above and writes
`paired_stats.json` with per-policy Wilson 95% intervals and all pairwise exact
McNemar tests with Holm-adjusted p-values.

Note on determinism: SAT repair outcomes are deterministic given the CNF encoding
and mutable set; per-case wall-clock times vary with hardware, so a 10 s timeout can
in principle flip a borderline case on much slower machines. On the reported runs,
bounded-policy queries decide in ~2--3 s on average and the control in ~9 s.

### 6. Constrained repair-localization sweeps (deterministic)

Budget axis (T=10 s): re-run steps 2 and 4 with `--mutable-budget` in {2, 6, 8}
(out-dirs `corrupt_main_test_budget{B}_t10`, `oracle_corrupt_main_test_budget{B}_t10`).
Timeout axis: re-run step 2 with `--policies all_parity` and step 4 at
`--timeout-sec` in {0.5, 2, 5, 30} (out-dirs `..._allparity_t{T}`, `oracle_..._t{T}`).

### 7. A_d objective ablation (deterministic, matched-seed GA)

```bash
.venv/bin/python scripts/run_ad_objective_ablation.py \
  --instance 43_10_16 --runs 50 --population 100 --generations 100 \
  --out-dir artifacts/external_baselines/ad_objective_43_10_16_n50_p100_g100
.venv/bin/python scripts/run_ad_objective_ablation.py \
  --instance 22_11_7 --runs 50 --population 100 --generations 100 \
  --out-dir artifacts/external_baselines/ad_objective_22_11_7_n50_p100_g100
```

Seeds come from archived witnesses in `artifacts/reviewer_core_claims.json`
([43,10,16]: first SAT witness with A16=91; [22,11,7]: verified final witness with
A7=176). Expected: d_min-only never improves A_d (0/50); d_min+A_d improves A16 in
13/50 runs (min 81, never worse; exact sign test p=2.4e-4); [22,11,7] is a
saturation control (50/50 ties at the reference A7=176).

### 8. Encoding-choice ablation (deterministic)

```bash
.venv/bin/python scripts/run_encoding_ablation.py \
  --repeats 3 --out-dir artifacts/external_baselines/encoding_ablation_main
```

Toggles the encoder options (systematic form, symmetry breaking, all-one
constraint) on direct [n,k,d] feasibility queries and records variables, clauses,
and CaDiCaL solve time/status.

### 9. Reflector decision-policy replay (deterministic)

```bash
.venv/bin/python scripts/run_reflector_replay_ablation.py \
  --runs 30 --horizon 150 \
  --out-dir artifacts/external_baselines/reflector_replay_n30_h150
```

Re-runs the deterministic [22,11,7] GA proxy with per-generation logging and
replays alternative repair-trigger policies (production threshold chain with and
without the diversity guard, d_min-only stagnation, fixed schedules, random
control) on identical trajectories. The candidate at each trigger generation is
actually repaired with all-parity SAT (30 s), so end-to-end costs are measured.
This is a decision-level replay: post-intervention trajectory changes are out of
scope by construction.

## Tier 2: Re-running the evolution (optional, requires OPENROUTER_API_KEY)

The adapters import the *public* upstream implementations from a sibling directory
one level above this repository:

```bash
mkdir -p ../external_baselines && cd ../external_baselines
git clone https://github.com/FeiLiu36/EOH.git EOH
(cd EOH && git checkout 36d10d49e9b80777fa544ac4e457b43ac6c2f9d0)
git clone https://github.com/google-deepmind/funsearch.git funsearch
(cd funsearch && git checkout cc53f274237d7ab05c19df939edbc1f9616a7c19)
```

Note the public FunSearch release omits its production LLM sampler, sandbox, and
distributed infrastructure; `scripts/run_funsearch_repair_policy.py` documents the
bounded local driver, OpenRouter completion, and AST-based sanitization we added.

```bash
# EoH: population 4, 3 generations (20 LLM samples consumed in our run)
.venv/bin/python scripts/run_eoh_repair_policy.py \
  --dataset artifacts/external_baselines/repair_policy_dataset_corrupt_main/candidates.jsonl \
  --split train --max-cases 24 --mutable-budget 4 \
  --repair-timeout-sec 5 --eval-timeout-sec 300 \
  --pop-size 4 --n-pop 3 --max-sample-nums 12 \
  --out-dir artifacts/external_baselines/eoh_repair_policy_corrupt_main

# FunSearch: 20 samples, matched to EoH's consumption
.venv/bin/python scripts/run_funsearch_repair_policy.py \
  --dataset artifacts/external_baselines/repair_policy_dataset_corrupt_main/candidates.jsonl \
  --split train --max-cases 24 --mutable-budget 4 \
  --repair-timeout-sec 5 --samples 20 --samples-per-prompt 1 \
  --out-dir artifacts/external_baselines/funsearch_repair_policy_corrupt_main_s20
```

LLM sampling is stochastic, so tier-2 reruns will generally produce different
policies; the manuscript reports single-run outcomes conditional on the frozen
policies committed in `release/policies/`.

## Artifact map

| Release path | Content |
| --- | --- |
| `release/dataset/` | Benchmark manifest + 222 candidates (train/val/test split) |
| `release/policies/` | Frozen best `score_columns` code from EoH and FunSearch |
| `release/runs/` | Evolution run manifests (config, sample budgets, LLM pin) |
| `release/results/` | Held-out test summaries per policy + paired statistics |
| `release/release_manifest.json` | SHA-256 checksums for every bundled file |
