#!/usr/bin/env python3
"""End-to-end baseline: public FunSearch evolves full [43,10] generator-matrix programs.

Unlike the repair-policy interface, this is the discovery task itself: the
evolved function emits a complete generator matrix and is scored by SolEvolve's
deterministic verifier (exact d_min and A16 over all 1023 codewords). Uses the
same public-FunSearch driver, OpenRouter completion path, and AST sanitization
as run_funsearch_repair_policy.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
FUNSEARCH_PARENT = WORKSPACE_ROOT / "external_baselines"
for path in (str(SRC_ROOT), str(FUNSEARCH_PARENT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from funsearch.implementation import code_manipulation  # noqa: E402
from funsearch.implementation import config as fun_config  # noqa: E402
from funsearch.implementation import evaluator as fun_evaluator  # noqa: E402
from funsearch.implementation import funsearch  # noqa: E402
from funsearch.implementation import programs_database  # noqa: E402

import http.client  # noqa: E402

from endtoend_matrix_eval import SEED_BODY_LINES, TASK_DESCRIPTION, run_candidate_program  # noqa: E402
from run_funsearch_repair_policy import sanitize_completion, best_from_database  # noqa: E402
from solevolve.baselines.repair_policy_eval import json_ready  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

SEED_BODY = "\n".join("  " + line for line in SEED_BODY_LINES)

SPECIFICATION = f'''\
import numpy as np


@funsearch.run
def evaluate(dummy: int) -> float:
  """Returns the verifier-backed score of the current construction."""
  return run_matrix_score(build_generator)


@funsearch.evolve
def build_generator():
  """{TASK_DESCRIPTION}"""
{SEED_BODY}
'''


MATRIX_INSTRUCTION = f"""Complete only the body of the final Python function in the following FunSearch prompt.
Return indented Python statements only, no markdown fences and no function header.

The evolved function is:
    build_generator() -> a 10 x 43 matrix (list of lists of 0/1 integers)

Task: {TASK_DESCRIPTION}

The function takes NO arguments. It must be deterministic, finish within 90
seconds, and return the matrix (rows of length 43). numpy is imported as np.
"""


def matrix_completion(*, prompt: str, api_endpoint: str, api_key: str, model: str, timeout: int) -> str | None:
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": f"{MATRIX_INSTRUCTION}\n\n{prompt}"}],
        }
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/solevolve",
        "X-Title": "SolEvolve end-to-end FunSearch baseline",
    }
    path = "/api/v1/chat/completions" if api_endpoint == "openrouter.ai" else "/v1/chat/completions"
    conn = http.client.HTTPSConnection(api_endpoint, timeout=timeout)
    try:
        conn.request("POST", path, payload, headers)
        parsed = json.loads(conn.getresponse().read())
        choices = parsed.get("choices")
        return choices[0]["message"]["content"] if choices else None
    finally:
        conn.close()


class MatrixSandbox(fun_evaluator.Sandbox):
    def __init__(self, *, log_dir: Path, timeout_sec: int) -> None:
        self.log_dir = log_dir
        self.timeout_sec = timeout_sec
        self.run_count = 0
        self.records: list[dict[str, Any]] = []

    def run(self, program: str, function_to_run: str, test_input: str, timeout_seconds: int) -> tuple[Any, bool]:
        del function_to_run, test_input, timeout_seconds
        self.run_count += 1
        record = run_candidate_program(program, timeout_sec=self.timeout_sec)
        matrix = record.pop("matrix", None)
        if matrix is not None and record["valid"]:
            (self.log_dir / f"matrix_{self.run_count:05d}.json").write_text(json.dumps(matrix) + "\n")
        record["run"] = self.run_count
        if not record["valid"]:
            (self.log_dir / f"failed_{self.run_count:05d}.py").write_text(program)
        self.records.append(record)
        print(f"[fs-matrix] eval {self.run_count}: valid={record['valid']} "
              f"d_min={record.get('d_min')} A16={record.get('A16')} obj={record['objective']}",
              file=sys.stderr, flush=True)
        return -float(record["objective"]), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--functions-per-prompt", type=int, default=2)
    parser.add_argument("--num-islands", type=int, default=4)
    parser.add_argument("--program-timeout-sec", type=int, default=90)
    parser.add_argument("--api-endpoint", default=os.getenv("OPENROUTER_API_ENDPOINT_HOST", "openrouter.ai"))
    parser.add_argument("--model", default=os.getenv("FUNSEARCH_OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "matrices").mkdir(parents=True, exist_ok=True)

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
    sandbox = MatrixSandbox(log_dir=out_dir / "matrices", timeout_sec=args.program_timeout_sec)
    evaluator = fun_evaluator.Evaluator(
        database, template, function_to_evolve, function_to_run,
        inputs=[0], timeout_seconds=args.program_timeout_sec + 30,
    )
    evaluator._sandbox = sandbox
    evaluator.analyse(template.get_function(function_to_evolve).body, island_id=None, version_generated=None)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set.")
    samples = []
    for sample_idx in range(args.samples):
        prompt = database.get_prompt()
        completion = matrix_completion(
            prompt=prompt.code,
            api_endpoint=args.api_endpoint,
            api_key=api_key,
            model=args.model,
            timeout=args.llm_timeout_sec,
        )
        body = sanitize_completion(completion)
        entry = {"sample": sample_idx, "sanitized": body is not None}
        if body is not None:
            evaluator.analyse(body, island_id=prompt.island_id, version_generated=prompt.version_generated)
        samples.append(entry)
        best = best_from_database(database)
        print(f"[fs-matrix] sample {sample_idx + 1}/{args.samples}: best objective={best['objective']}",
              file=sys.stderr, flush=True)
        (out_dir / "run_manifest.json").write_text(json.dumps(json_ready({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": "End-to-end FunSearch: evolve [43,10] generator-matrix programs against the SolEvolve verifier.",
            "args": vars(args),
            "llm": {"provider": "openrouter", "model": args.model},
            "samples": samples,
            "best": best,
            "evaluations": sandbox.records,
        }), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best = best_from_database(database)
    if best.get("code"):
        (out_dir / "samples_best.json").write_text(json.dumps(json_ready(best), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_ready({"best": best, "evaluations": len(sandbox.records)}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
