#!/usr/bin/env python3
"""Run EoH on the shared verifier-facing repair-policy interface."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
EOH_SRC = WORKSPACE_ROOT / "external_baselines" / "EOH" / "eoh" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EOH_SRC) not in sys.path:
    sys.path.insert(0, str(EOH_SRC))

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from eoh import BaseProblem, EoH, LLMConfig  # noqa: E402

from solevolve.baselines.repair_policy_eval import (  # noqa: E402
    RepairPolicyConfig,
    RepairPolicyEvaluator,
    json_ready,
    load_jsonl,
)
from solevolve.hybrid_ga import TARGET_D, TARGET_K, TARGET_N  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def patch_eoh_openrouter_api() -> None:
    """Make EoH's generic OpenAI-style wrapper use OpenRouter's /api/v1 path."""
    from eoh.llm.api_general import InterfaceAPI

    def get_response(self, prompt_content: str, max_retries: int = 5) -> str | None:
        payload = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [{"role": "user", "content": prompt_content}],
            }
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/solevolve",
            "X-Title": "SolEvolve EoH repair-policy baseline",
        }
        path = "/api/v1/chat/completions" if self.api_endpoint == "openrouter.ai" else "/v1/chat/completions"
        for attempt in range(max_retries):
            conn = None
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint, timeout=self.timeout)
                conn.request("POST", path, payload, headers)
                data = conn.getresponse().read()
                parsed = json.loads(data)
                choices = parsed.get("choices")
                if not choices:
                    raise ValueError(parsed.get("error", {}).get("message", str(parsed))[:300])
                return choices[0]["message"]["content"]
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
            finally:
                if conn is not None:
                    conn.close()
        return None

    InterfaceAPI.get_response = get_response


class BinaryRepairPolicyProblem(BaseProblem):
    """EoH task: generate a column-scoring policy for SAT repair."""

    template_program = '''
import numpy as np

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
        global_features: [n, k, target_d, d_min, target_gap,
            low_weight_count, below_target_count, parity_column_count].
    Returns:
        A one-dimensional numpy array of length m. Higher scores are selected
        as mutable parity columns.
    """
    return column_features[:, 4] + 0.1 * column_features[:, 8]
'''

    task_description = (
        "Design a deterministic numerical scoring function for choosing mutable "
        "parity columns in a SAT repair query for binary linear codes. The evaluator "
        "selects the top-scoring columns, runs the same CaDiCaL verifier-backed SAT "
        "repair for every policy, and rewards policies that repair more candidates "
        "with fewer timeouts and lower runtime. Return only valid Python code."
    )

    def __init__(
        self,
        *,
        candidates: list[dict[str, Any]],
        repair_config: RepairPolicyConfig,
        cadical_path: str,
        artifact_dir: str | Path,
        timeout: int,
        n_processes: int = 1,
        max_cases: int | None = None,
    ) -> None:
        super().__init__(timeout=timeout, n_processes=n_processes)
        self.candidates = candidates[: max_cases or len(candidates)]
        self.repair_config = repair_config
        self.cadical_path = cadical_path
        self.artifact_dir = Path(artifact_dir)
        self.eval_counter = 0

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        del program_str
        self.eval_counter += 1
        evaluator = RepairPolicyEvaluator(
            self.candidates,
            config=self.repair_config,
            cadical_path=self.cadical_path,
            artifact_dir=self.artifact_dir / f"eval_{self.eval_counter:05d}",
        )
        result = evaluator.evaluate_policy(
            policy_name="eoh_candidate_policy",
            policy=callable_func,
            seed=2026,
        )
        summary = result["summary"]
        failures = int(summary["total"] - summary["successes"])
        timeouts = int(summary["solver_status_counts"].get("TIMEOUT", 0))
        errors = int(summary["solver_status_counts"].get("ERROR", 0))
        # EoH minimizes objective. Success dominates, then timeout/error burden,
        # then runtime and selected-column parsimony.
        return float(
            1000.0 * failures
            + 100.0 * timeouts
            + 100.0 * errors
            + 0.01 * float(summary["mean_elapsed_ms"])
            + 0.1 * float(summary["mean_mutable_columns"])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "repair_policy_dataset_stratified_pilot" / "candidates.jsonl"),
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--mutable-budget", type=int, default=6)
    parser.add_argument("--repair-timeout-sec", type=float, default=5.0)
    parser.add_argument("--eval-timeout-sec", type=int, default=120)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--k", type=int, default=TARGET_K)
    parser.add_argument("--target-d", type=int, default=TARGET_D)
    parser.add_argument("--cadical-path", default=str(REPO_ROOT / "cadical" / "build" / "cadical"))
    parser.add_argument("--api-endpoint", default=os.getenv("OPENROUTER_API_ENDPOINT_HOST", "openrouter.ai"))
    parser.add_argument("--model", default=os.getenv("EOH_OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--n-pop", type=int, default=3)
    parser.add_argument("--max-sample-nums", type=int, default=8)
    parser.add_argument("--num-samplers", type=int, default=1)
    parser.add_argument("--num-evaluators", type=int, default=1)
    parser.add_argument("--dry-run-template", action="store_true", help="Evaluate the template policy without calling an LLM.")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "artifacts" / "external_baselines" / "eoh_repair_policy"),
    )
    return parser.parse_args()


def load_candidates(path: str | Path, split: str) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if split == "all":
        return rows
    return [row for row in rows if row.get("split") == split]


def main() -> None:
    args = parse_args()
    if args.api_endpoint == "openrouter.ai":
        patch_eoh_openrouter_api()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.dataset, args.split)
    repair_config = RepairPolicyConfig(
        n=args.n,
        k=args.k,
        target_d=args.target_d,
        mutable_budget=args.mutable_budget,
        timeout_sec=args.repair_timeout_sec,
        expected_a_d=None,
    )
    problem = BinaryRepairPolicyProblem(
        candidates=candidates,
        repair_config=repair_config,
        cadical_path=args.cadical_path,
        artifact_dir=out_dir / "evaluations",
        timeout=args.eval_timeout_sec,
        max_cases=args.max_cases,
    )

    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "EoH direct comparison on SolEvolve's shared verifier-facing repair-policy interface.",
        "dataset": args.dataset,
        "split": args.split,
        "candidate_count": len(candidates),
        "max_cases": args.max_cases,
        "repair_config": repair_config.__dict__,
        "llm": {
            "provider": "openrouter",
            "api_endpoint": args.api_endpoint,
            "model": args.model,
            "api_key_env": args.api_key_env,
        },
        "eoh": {
            "pop_size": args.pop_size,
            "n_pop": args.n_pop,
            "max_sample_nums": args.max_sample_nums,
            "num_samplers": args.num_samplers,
            "num_evaluators": args.num_evaluators,
        },
        "output_dir": str(out_dir),
    }

    if args.dry_run_template:
        objective = problem.evaluate(problem.template_program)
        run_manifest["dry_run_template_objective"] = objective
        (out_dir / "run_manifest.json").write_text(
            json.dumps(json_ready(run_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(json_ready(run_manifest), indent=2, sort_keys=True))
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set; cannot run OpenRouter-backed EoH.")

    llm = LLMConfig(
        api_endpoint=args.api_endpoint,
        api_key=api_key,
        model=args.model,
        timeout=args.llm_timeout_sec,
    )
    eoh = EoH(
        llm=llm,
        problem=problem,
        pop_size=args.pop_size,
        n_pop=args.n_pop,
        max_sample_nums=args.max_sample_nums,
        num_samplers=args.num_samplers,
        num_evaluators=args.num_evaluators,
        output_dir=str(out_dir),
        debug=False,
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(json_ready(run_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eoh.run()


if __name__ == "__main__":
    main()
