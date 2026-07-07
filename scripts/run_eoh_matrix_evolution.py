#!/usr/bin/env python3
"""End-to-end baseline: public EoH evolves full [43,10] generator-matrix programs.

The discovery task itself (not the repair-policy sub-task): each evolved
heuristic must emit a complete generator matrix, scored by SolEvolve's
deterministic verifier (exact d_min and A16 over all 1023 codewords). Candidate
programs run in an isolated subprocess with a wall-clock limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
EOH_SRC = WORKSPACE_ROOT / "external_baselines" / "EOH" / "eoh" / "src"
for path in (str(SRC_ROOT), str(EOH_SRC), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv  # noqa: E402
from eoh import BaseProblem, EoH, LLMConfig  # noqa: E402

from endtoend_matrix_eval import SEED_BODY_LINES, TASK_DESCRIPTION, run_candidate_program  # noqa: E402
from run_eoh_repair_policy import patch_eoh_openrouter_api  # noqa: E402
from solevolve.baselines.repair_policy_eval import json_ready  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

SEED_BODY = "\n".join("    " + line for line in SEED_BODY_LINES)

TEMPLATE_PROGRAM = f'''
import numpy as np

def build_generator():
    """Return a 10x43 binary generator matrix (list of lists of 0/1 ints)."""
{SEED_BODY}
'''


class MatrixConstructionProblem(BaseProblem):
    """EoH task: construct a binary [43,10,>=16] code minimizing A16."""

    template_program = TEMPLATE_PROGRAM
    task_description = TASK_DESCRIPTION + " Return only valid Python code defining build_generator()."

    def __init__(self, *, log_dir: str | Path, timeout: int, program_timeout_sec: int) -> None:
        super().__init__(timeout=timeout, n_processes=1)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.program_timeout_sec = program_timeout_sec
        self.eval_counter = 0
        self.records: list[dict[str, Any]] = []

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        # EoH copies the problem object per evaluation, so all bookkeeping goes
        # to disk rather than instance state.
        del callable_func
        import hashlib
        tag = hashlib.sha1(program_str.encode()).hexdigest()[:10]
        record = run_candidate_program(program_str, timeout_sec=self.program_timeout_sec)
        matrix = record.pop("matrix", None)
        if matrix is not None and record["valid"]:
            (self.log_dir / f"matrix_{tag}.json").write_text(json.dumps(matrix) + "\n")
        elif not record["valid"]:
            (self.log_dir / f"failed_{tag}.py").write_text(program_str)
        record["tag"] = tag
        with (self.log_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[eoh-matrix] eval {tag}: valid={record['valid']} "
              f"d_min={record.get('d_min')} A16={record.get('A16')} obj={record['objective']}",
              file=sys.stderr, flush=True)
        return float(record["objective"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--n-pop", type=int, default=8)
    parser.add_argument("--max-sample-nums", type=int, default=100)
    parser.add_argument("--program-timeout-sec", type=int, default=90)
    parser.add_argument("--eval-timeout-sec", type=int, default=300)
    parser.add_argument("--api-endpoint", default=os.getenv("OPENROUTER_API_ENDPOINT_HOST", "openrouter.ai"))
    parser.add_argument("--model", default=os.getenv("EOH_OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=int, default=180)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.api_endpoint == "openrouter.ai":
        patch_eoh_openrouter_api()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = MatrixConstructionProblem(
        log_dir=out_dir / "matrices",
        timeout=args.eval_timeout_sec,
        program_timeout_sec=args.program_timeout_sec,
    )
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set.")
    llm = LLMConfig(
        api_endpoint=args.api_endpoint,
        api_key=api_key,
        model=args.model,
        timeout=args.llm_timeout_sec,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "End-to-end EoH: evolve [43,10] generator-matrix programs against the SolEvolve verifier.",
        "args": {k: v for k, v in vars(args).items()},
        "llm": {"provider": "openrouter", "model": args.model},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n")
    eoh = EoH(
        llm=llm,
        problem=problem,
        pop_size=args.pop_size,
        n_pop=args.n_pop,
        max_sample_nums=args.max_sample_nums,
        num_samplers=1,
        num_evaluators=1,
        output_dir=str(out_dir),
        debug=False,
    )
    eoh.run()
    records = [json.loads(line) for line in (out_dir / "matrices" / "evaluations.jsonl").read_text().splitlines() if line.strip()]
    manifest["evaluations"] = records
    valid = [r for r in records if r["valid"]]
    manifest["summary"] = {
        "evaluations": len(records),
        "valid": len(valid),
        "best_objective": min((r["objective"] for r in valid), default=None),
        "best_d_min": max((r["d_min"] for r in valid), default=None),
        "best_A16_at_d16": min((r["A16"] for r in valid if r.get("d_min", 0) >= 16), default=None),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_ready(manifest["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
