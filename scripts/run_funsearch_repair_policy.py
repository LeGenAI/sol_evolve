#!/usr/bin/env python3
"""Adapt the public FunSearch implementation to SolEvolve's repair-policy verifier."""

from __future__ import annotations

import argparse
import ast
import http.client
import json
import os
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
FUNSEARCH_PARENT = WORKSPACE_ROOT / "external_baselines"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FUNSEARCH_PARENT) not in sys.path:
    sys.path.insert(0, str(FUNSEARCH_PARENT))

from funsearch.implementation import code_manipulation  # noqa: E402
from funsearch.implementation import config as fun_config  # noqa: E402
from funsearch.implementation import evaluator as fun_evaluator  # noqa: E402
from funsearch.implementation import funsearch  # noqa: E402
from funsearch.implementation import programs_database  # noqa: E402

from solevolve.baselines.repair_policy_eval import (  # noqa: E402
    RepairPolicyConfig,
    RepairPolicyEvaluator,
    json_ready,
    load_jsonl,
)
from solevolve.hybrid_ga import TARGET_D, TARGET_K, TARGET_N  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


SPECIFICATION = '''\
import numpy as np


@funsearch.run
def evaluate(case_id: int) -> float:
  """Returns a verifier-backed score for the current repair policy."""
  return run_policy_score(case_id, score_columns)


@funsearch.evolve
def score_columns(column_features: np.ndarray, global_features: np.ndarray) -> np.ndarray:
  """Score parity columns for bounded SAT repair.

  Args:
    column_features: shape (m, 12). Columns are:
      0 column_index,
      1 parity_position,
      2 normalized_parity_position,
      3 matrix_column_weight,
      4 low_support_count,
      5 low_support_fraction,
      6 min_low_weight_touching_column,
      7 mean_low_weight_touching_column,
      8 weight_deficit_sum_touching_column,
      9 column_pattern_fraction,
      10 target_gap,
      11 low_weight_count.
    global_features: [n, k, target_d, d_min, target_gap, low_weight_count,
      below_target_count, parity_column_count].
  Returns:
    One score per parity column. Higher-scoring columns are made mutable.
  """
  return column_features[:, 4] + 0.1 * column_features[:, 8]
'''


class _FunsearchDecorators:
    @staticmethod
    def run(func):
        return func

    @staticmethod
    def evolve(func):
        return func


class VerifierSandbox(fun_evaluator.Sandbox):
    def __init__(
        self,
        *,
        candidates: list[dict[str, Any]],
        repair_config: RepairPolicyConfig,
        cadical_path: str,
        artifact_dir: str | Path,
    ) -> None:
        self.candidates = candidates
        self.repair_config = repair_config
        self.cadical_path = cadical_path
        self.artifact_dir = Path(artifact_dir)
        self.run_count = 0

    def run(self, program: str, function_to_run: str, test_input: str, timeout_seconds: int) -> tuple[Any, bool]:
        del timeout_seconds
        self.run_count += 1

        def run_policy_score(case_id: int, score_columns):
            del case_id
            evaluator = RepairPolicyEvaluator(
                self.candidates,
                config=self.repair_config,
                cadical_path=self.cadical_path,
                artifact_dir=self.artifact_dir / f"eval_{self.run_count:05d}",
            )
            result = evaluator.evaluate_policy(
                policy_name="funsearch_candidate_policy",
                policy=score_columns,
                seed=2026,
            )
            summary = result["summary"]
            failures = int(summary["total"] - summary["successes"])
            timeouts = int(summary["solver_status_counts"].get("TIMEOUT", 0))
            errors = int(summary["solver_status_counts"].get("ERROR", 0))
            objective = (
                1000.0 * failures
                + 100.0 * timeouts
                + 100.0 * errors
                + 0.01 * float(summary["mean_elapsed_ms"])
                + 0.1 * float(summary["mean_mutable_columns"])
            )
            return -float(objective)

        namespace = {
            "np": np,
            "funsearch": _FunsearchDecorators,
            "run_policy_score": run_policy_score,
            "__builtins__": __builtins__,
        }
        try:
            exec(compile(program, "<funsearch-repair-policy>", "exec"), namespace)
            output = namespace[function_to_run](test_input)
            return output, True
        except Exception:
            return None, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset_stratified_pilot" / "candidates.jsonl"),
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--mutable-budget", type=int, default=4)
    parser.add_argument("--repair-timeout-sec", type=float, default=0.5)
    parser.add_argument("--evaluator-timeout-sec", type=int, default=120)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--functions-per-prompt", type=int, default=2)
    parser.add_argument("--num-islands", type=int, default=4)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--api-endpoint", default=os.getenv("OPENROUTER_API_ENDPOINT_HOST", "openrouter.ai"))
    parser.add_argument("--model", default=os.getenv("FUNSEARCH_OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--dry-run-initial", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "funsearch_repair_policy"),
    )
    return parser.parse_args()


def load_candidates(path: str | Path, split: str, max_cases: int | None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    return rows[: max_cases or len(rows)]


def openrouter_completion(*, prompt: str, api_endpoint: str, api_key: str, model: str, timeout: int) -> str | None:
    instruction = """Complete only the body of the final Python function in the following FunSearch prompt.
Return indented Python statements only, no markdown fences and no function header.

The evolved function is:
    score_columns(column_features: np.ndarray, global_features: np.ndarray) -> np.ndarray

Use this exact feature map:
    column_features[:, 0] = column_index
    column_features[:, 1] = parity_position
    column_features[:, 2] = normalized_parity_position
    column_features[:, 3] = matrix_column_weight
    column_features[:, 4] = low_support_count
    column_features[:, 5] = low_support_fraction
    column_features[:, 6] = min_low_weight_touching_column
    column_features[:, 7] = mean_low_weight_touching_column
    column_features[:, 8] = weight_deficit_sum_touching_column
    column_features[:, 9] = column_pattern_fraction
    column_features[:, 10] = target_gap
    column_features[:, 11] = low_weight_count
    global_features = [n, k, target_d, d_min, target_gap, low_weight_count,
                       below_target_count, parity_column_count]

Higher scores select mutable parity columns for the verifier-backed SAT repair.
Keep the function deterministic and vectorized with numpy.
"""
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.7,
            "max_tokens": 1200,
            "messages": [
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{prompt}",
                }
            ],
        }
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/solevolve",
        "X-Title": "SolEvolve FunSearch repair-policy baseline",
    }
    path = "/api/v1/chat/completions" if api_endpoint == "openrouter.ai" else "/v1/chat/completions"
    conn = http.client.HTTPSConnection(api_endpoint, timeout=timeout)
    try:
        conn.request("POST", path, payload, headers)
        data = conn.getresponse().read()
        parsed = json.loads(data)
        choices = parsed.get("choices")
        if not choices:
            return None
        return choices[0]["message"]["content"]
    finally:
        conn.close()


def _fenced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        block = parts[idx]
        if block.lstrip().startswith("python"):
            block = block.lstrip()[len("python") :]
        blocks.append(block.strip("\n"))
    return blocks


def _valid_function_body(body: str) -> str | None:
    dedented = textwrap.dedent(body).strip("\n")
    if not dedented or "return" not in dedented:
        return None
    source = "def _candidate_policy(column_features, global_features):\n"
    source += "\n".join(f"    {line}" if line.strip() else "" for line in dedented.splitlines())
    try:
        ast.parse(source)
    except SyntaxError:
        return None
    return "\n".join(f"  {line}" if line.strip() else "" for line in dedented.splitlines())


def sanitize_completion(completion: str | None) -> str | None:
    if not completion:
        return None
    candidates = _fenced_blocks(completion)
    candidates.append(completion)
    lines = completion.splitlines()
    candidates.extend("\n".join(lines[start:]) for start in range(len(lines)) if lines[start].strip())
    for candidate in candidates:
        valid = _valid_function_body(candidate)
        if valid is not None:
            return valid
    return None


def best_from_database(database: programs_database.ProgramsDatabase) -> dict[str, Any]:
    best_score = None
    best_program = None
    for score, program in zip(database._best_score_per_island, database._best_program_per_island, strict=True):
        if program is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_program = program
    return {
        "score": best_score,
        "objective": -best_score if best_score is not None else None,
        "code": str(best_program) if best_program is not None else None,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.dataset, args.split, args.max_cases)
    repair_config = RepairPolicyConfig(
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        mutable_budget=args.mutable_budget,
        timeout_sec=args.repair_timeout_sec,
        expected_a_d=None,
    )
    function_to_evolve, function_to_run = funsearch._extract_function_names(SPECIFICATION)
    template = code_manipulation.text_to_program(SPECIFICATION)
    cfg = fun_config.Config(
        programs_database=fun_config.ProgramsDatabaseConfig(
            functions_per_prompt=args.functions_per_prompt,
            num_islands=args.num_islands,
        ),
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=args.samples_per_prompt,
    )
    database = programs_database.ProgramsDatabase(cfg.programs_database, template, function_to_evolve)
    sandbox = VerifierSandbox(
        candidates=candidates,
        repair_config=repair_config,
        cadical_path=args.cadical_path,
        artifact_dir=out_dir / "evaluations",
    )
    evaluator = fun_evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs=[0],
        timeout_seconds=args.evaluator_timeout_sec,
    )
    evaluator._sandbox = sandbox
    initial_body = template.get_function(function_to_evolve).body
    evaluator.analyse(initial_body, island_id=None, version_generated=None)

    samples = []
    if not args.dry_run_initial:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise SystemExit(f"{args.api_key_env} is not set; cannot run OpenRouter-backed FunSearch.")
        for sample_idx in range(args.samples):
            prompt = database.get_prompt()
            completion = openrouter_completion(
                prompt=prompt.code,
                api_endpoint=args.api_endpoint,
                api_key=api_key,
                model=args.model,
                timeout=args.llm_timeout_sec,
            )
            sanitized_completion = sanitize_completion(completion)
            if sanitized_completion:
                evaluator.analyse(sanitized_completion, prompt.island_id, prompt.version_generated)
            samples.append(
                {
                    "sample_index": sample_idx,
                    "island_id": prompt.island_id,
                    "version_generated": prompt.version_generated,
                    "raw_completion": completion,
                    "completion": sanitized_completion,
                    "sanitized": sanitized_completion is not None,
                }
            )

    best = best_from_database(database)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Bounded adaptation of the public FunSearch implementation to SolEvolve's repair-policy verifier.",
        "dataset": args.dataset,
        "split": args.split,
        "candidate_count": len(candidates),
        "repair_config": repair_config.__dict__,
        "llm": {
            "provider": "openrouter",
            "api_endpoint": args.api_endpoint,
            "model": args.model,
            "api_key_env": args.api_key_env,
        },
        "funsearch": {
            "samples": args.samples,
            "samples_per_prompt": args.samples_per_prompt,
            "functions_per_prompt": args.functions_per_prompt,
            "num_islands": args.num_islands,
            "used_public_components": ["ProgramsDatabase", "Evaluator", "code_manipulation"],
            "adapted_components": ["OpenRouter completion", "verifier sandbox", "bounded driver"],
        },
        "best": best,
        "samples": samples,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if best.get("code"):
        (out_dir / "samples_best.json").write_text(
            json.dumps(
                {
                    "algorithm": "FunSearch bounded repair-policy program",
                    "code": best["code"],
                    "objective": best["objective"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(json_ready(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
