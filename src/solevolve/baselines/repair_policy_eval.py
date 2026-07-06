from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from solevolve.engines.cnf_encoder import CNFEncoder
from solevolve.engines.result_parser import ResultParser
from solevolve.engines.sat_solver_interface import SATSolver
from solevolve.hybrid_ga import (
    TARGET_D,
    TARGET_K,
    TARGET_N,
    _candidate_id,
    _diagnostics,
)
from solevolve.so_claim_repro import resolve_cadical
from solevolve.tracing import file_sha256

ColumnPolicy = Callable[[np.ndarray, np.ndarray], Any]


@dataclass(frozen=True)
class RepairPolicyConfig:
    n: int = TARGET_N
    k: int = TARGET_K
    target_d: int = TARGET_D
    mutable_budget: int = 4
    timeout_sec: float = 1.0
    solver_type: str = "cadical"
    expected_a_d: int | None = None
    keep_cnf: bool = False
    quiet: bool = True


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def matrix_sha256(matrix: list[list[int]]) -> str:
    payload = json.dumps(matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (centre - spread) / denom, (centre + spread) / denom


def exact_mcnemar_p(variant_only: int, reference_only: int) -> float:
    trials = variant_only + reference_only
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, idx) for idx in range(0, min(variant_only, reference_only) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def _codeword_from_message(arr: np.ndarray, message: int, *, n: int, k: int) -> np.ndarray:
    codeword = np.zeros(n, dtype=np.uint8)
    for bit_idx in range(k):
        if message & (1 << bit_idx):
            codeword ^= arr[bit_idx]
    return codeword


def low_weight_codewords(
    matrix: list[list[int]],
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
) -> list[dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for message in range(1, 1 << k):
        codeword = _codeword_from_message(arr, message, n=n, k=k)
        weight = int(codeword.sum())
        if weight < target_d:
            support = [idx for idx, value in enumerate(codeword.tolist()) if value]
            rows.append(
                {
                    "message": message,
                    "weight": weight,
                    "support": support,
                    "parity_support": [idx for idx in support if idx >= k],
                }
            )
    return rows


def below_target_count(diagnostics: dict[str, Any], target_d: int) -> int:
    distribution = diagnostics.get("weight_distribution") or {}
    total = 0
    for weight, count in distribution.items():
        weight_int = int(weight)
        if 0 < weight_int < target_d:
            total += int(count)
    return total


def candidate_record(
    matrix: list[list[int]],
    *,
    source: str,
    seed: int | None = None,
    split: str | None = None,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = [[int(value) & 1 for value in row] for row in matrix]
    diagnostics = _diagnostics(normalized, n=n, k=k, d=target_d, expected_a_d=None)
    lows = low_weight_codewords(normalized, n=n, k=k, target_d=target_d)
    return {
        "candidate_id": _candidate_id(normalized),
        "matrix_sha256": matrix_sha256(normalized),
        "source": source,
        "seed": seed,
        "split": split,
        "n": n,
        "k": k,
        "target_d": target_d,
        "matrix": normalized,
        "diagnostics": diagnostics,
        "d_min": int(diagnostics.get("minimum_distance") or 0),
        "below_target_count": below_target_count(diagnostics, target_d),
        "low_weight_count": len(lows),
        "low_weight_supports": [
            {key: item[key] for key in ("message", "weight", "support", "parity_support")}
            for item in lows[:64]
        ],
        "metadata": metadata or {},
    }


def column_feature_table(
    matrix: list[list[int]],
    *,
    n: int = TARGET_N,
    k: int = TARGET_K,
    target_d: int = TARGET_D,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.uint8)
    lows = low_weight_codewords(matrix, n=n, k=k, target_d=target_d)
    parity_columns = list(range(k, n))
    low_count = max(1, len(lows))
    diagnostics = _diagnostics(matrix, n=n, k=k, d=target_d, expected_a_d=None)
    d_min = int(diagnostics.get("minimum_distance") or 0)
    target_gap = max(0, target_d - d_min)

    rows = []
    for col in parity_columns:
        containing = [item for item in lows if col in item["parity_support"]]
        support_count = len(containing)
        weights = [int(item["weight"]) for item in containing]
        deficits = [target_d - weight for weight in weights]
        column_values = arr[:, col]
        pattern_int = int(sum(int(value) << idx for idx, value in enumerate(column_values.tolist())))
        rows.append(
            [
                float(col),
                float(col - k),
                float((col - k) / max(1, n - k - 1)),
                float(int(column_values.sum())),
                float(support_count),
                float(support_count / low_count),
                float(min(weights) if weights else target_d),
                float(statistics.fmean(weights) if weights else target_d),
                float(sum(deficits)),
                float(pattern_int / max(1, (1 << k) - 1)),
                float(target_gap),
                float(len(lows)),
            ]
        )
    global_features = np.array(
        [
            float(n),
            float(k),
            float(target_d),
            float(d_min),
            float(target_gap),
            float(len(lows)),
            float(below_target_count(diagnostics, target_d)),
            float(n - k),
        ],
        dtype=float,
    )
    meta = {
        "feature_names": [
            "column_index",
            "parity_position",
            "normalized_parity_position",
            "matrix_column_weight",
            "low_support_count",
            "low_support_fraction",
            "min_low_weight_touching_column",
            "mean_low_weight_touching_column",
            "weight_deficit_sum_touching_column",
            "column_pattern_fraction",
            "target_gap",
            "low_weight_count",
        ],
        "global_feature_names": [
            "n",
            "k",
            "target_d",
            "d_min",
            "target_gap",
            "low_weight_count",
            "below_target_count",
            "parity_column_count",
        ],
        "low_weight_count": len(lows),
        "d_min": d_min,
    }
    return np.asarray(rows, dtype=float), global_features, parity_columns, meta


def select_top_budget(scores: Any, parity_columns: list[int], budget: int) -> list[int]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != len(parity_columns) or not np.all(np.isfinite(values)):
        raise ValueError("policy returned invalid column scores")
    budget = max(1, min(int(budget), len(parity_columns)))
    order = np.argsort(-values, kind="mergesort")[:budget]
    return sorted(int(parity_columns[idx]) for idx in order)


def low_weight_support_policy(column_features: np.ndarray, global_features: np.ndarray) -> np.ndarray:
    del global_features
    return column_features[:, 4] * 100.0 + column_features[:, 8] - column_features[:, 2] * 1e-3


def random_policy_factory(seed: int) -> ColumnPolicy:
    rng = np.random.default_rng(seed)

    def _policy(column_features: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        del global_features
        return rng.random(column_features.shape[0])

    return _policy


def all_parity_columns(columns: Iterable[int]) -> list[int]:
    return sorted(int(column) for column in columns)


def repair_with_columns(
    matrix: list[list[int]],
    mutable_columns: list[int],
    *,
    config: RepairPolicyConfig,
    cadical_path: str | None,
    workdir: str | Path,
    case_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = resolve_cadical(cadical_path)
    if not resolved:
        return {
            "case_id": case_id,
            "solver_status": "UNAVAILABLE",
            "target_achieved": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "mutable_columns": sorted(mutable_columns),
            "post_diagnostics": None,
        }

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mutable_set = {int(column) for column in mutable_columns if config.k <= int(column) < config.n}
    fixed_columns = [col for col in range(config.k, config.n) if col not in mutable_set]
    cnf_path = workdir / f"{case_id}.cnf"

    stream = io.StringIO()
    stdout_cm = contextlib.redirect_stdout(stream) if config.quiet else contextlib.nullcontext()
    with stdout_cm:
        encoder = CNFEncoder(config.n, config.k, config.target_d, systematic=True)
        encoder.encode_all_constraints()
        for row_idx in range(config.k):
            for col in fixed_columns:
                var = encoder.var_map[("P", row_idx, col - config.k)]
                encoder.clauses.append([var if int(matrix[row_idx][col]) else -var])
        encoder.export_to_dimacs(str(cnf_path))
        solver = SATSolver(solver_type="cadical", solver_path=resolved)
        result = solver.solve(str(cnf_path), timeout=config.timeout_sec, verbose=False)

    post_diagnostics = None
    repaired_matrix = None
    solver_status = str(result.get("status") or "UNKNOWN")
    if solver_status == "SAT" and result.get("model"):
        parser = ResultParser(config.n, config.k, config.target_d, systematic=True)
        repaired = parser.parse_sat_model(result["model"], encoder.var_map)
        repaired_matrix = repaired.astype(int).tolist()
        post_diagnostics = _diagnostics(
            repaired_matrix,
            n=config.n,
            k=config.k,
            d=config.target_d,
            expected_a_d=config.expected_a_d,
        )
    elif solver_status == "SAT":
        solver_status = "SAT_NO_MODEL"

    if not config.keep_cnf:
        with contextlib.suppress(OSError):
            cnf_path.unlink()
        with contextlib.suppress(OSError):
            Path(str(cnf_path) + ".cadical.out").unlink()

    return {
        "case_id": case_id,
        "solver_status": solver_status,
        "target_achieved": bool((post_diagnostics or {}).get("target_achieved")),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "solver_elapsed_ms": round(float(result.get("time") or 0.0) * 1000, 3),
        "mutable_columns": sorted(mutable_set),
        "mutable_column_count": len(mutable_set),
        "fixed_column_count": len(fixed_columns),
        "cnf_path": str(cnf_path) if config.keep_cnf else None,
        "cnf_sha256": file_sha256(cnf_path) if config.keep_cnf and cnf_path.exists() else None,
        "solver_path": result.get("solver_path") or resolved,
        "solver_command": result.get("command"),
        "post_diagnostics": post_diagnostics,
        "post_matrix": repaired_matrix,
        "stdout_preview": stream.getvalue()[-1000:] if not config.quiet else None,
    }


class RepairPolicyEvaluator:
    def __init__(
        self,
        candidates: list[dict[str, Any]],
        *,
        config: RepairPolicyConfig,
        cadical_path: str | None,
        artifact_dir: str | Path,
    ) -> None:
        self.candidates = candidates
        self.config = config
        self.cadical_path = cadical_path
        self.artifact_dir = Path(artifact_dir)

    def select_columns(
        self,
        candidate: dict[str, Any],
        *,
        policy: ColumnPolicy | None,
        policy_name: str,
        seed: int,
    ) -> tuple[list[int], dict[str, Any]]:
        features, global_features, parity_columns, feature_meta = column_feature_table(
            candidate["matrix"],
            n=self.config.n,
            k=self.config.k,
            target_d=self.config.target_d,
        )
        if policy_name == "all_parity":
            return all_parity_columns(parity_columns), feature_meta
        if policy_name == "random":
            policy = random_policy_factory(seed)
        if policy_name in {"low_weight_support", "solevolve_rule"}:
            policy = low_weight_support_policy
        if policy is None:
            raise ValueError(f"policy function is required for {policy_name}")
        scores = policy(features.copy(), global_features.copy())
        columns = select_top_budget(scores, parity_columns, self.config.mutable_budget)
        return columns, feature_meta

    def evaluate_policy(
        self,
        *,
        policy_name: str,
        policy: ColumnPolicy | None = None,
        seed: int = 0,
        max_cases: int | None = None,
    ) -> dict[str, Any]:
        records = []
        selected = self.candidates[: max_cases or len(self.candidates)]
        for idx, candidate in enumerate(selected):
            case_seed = seed + idx * 9973
            try:
                columns, feature_meta = self.select_columns(
                    candidate,
                    policy=policy,
                    policy_name=policy_name,
                    seed=case_seed,
                )
                repair = repair_with_columns(
                    candidate["matrix"],
                    columns,
                    config=self.config,
                    cadical_path=self.cadical_path,
                    workdir=self.artifact_dir / "scratch" / policy_name,
                    case_id=f"{policy_name}_{idx:05d}_{candidate['candidate_id'][:10]}",
                )
                error = None
            except Exception as exc:
                feature_meta = {}
                repair = {
                    "solver_status": "ERROR",
                    "target_achieved": False,
                    "elapsed_ms": 0.0,
                    "mutable_columns": [],
                    "post_diagnostics": None,
                }
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "policy": policy_name,
                    "case_index": idx,
                    "candidate_id": candidate.get("candidate_id"),
                    "seed": candidate.get("seed"),
                    "split": candidate.get("split"),
                    "pre_d_min": candidate.get("d_min"),
                    "pre_below_target_count": candidate.get("below_target_count"),
                    "pre_low_weight_count": candidate.get("low_weight_count"),
                    "feature_meta": feature_meta,
                    "repair": repair,
                    "target_achieved": bool(repair.get("target_achieved")),
                    "solver_status": repair.get("solver_status"),
                    "elapsed_ms": repair.get("elapsed_ms"),
                    "mutable_column_count": repair.get("mutable_column_count", len(repair.get("mutable_columns") or [])),
                    "error": error,
                }
            )
        return {"policy": policy_name, "records": records, "summary": summarize_records(records)}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successes = sum(1 for record in records if record.get("target_achieved"))
    low, high = wilson_interval(successes, total)
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("solver_status") or "UNKNOWN")
        statuses[status] = statuses.get(status, 0) + 1
    elapsed = [float(record.get("elapsed_ms") or 0.0) for record in records]
    mutable_counts = [int(record.get("mutable_column_count") or 0) for record in records]
    return {
        "successes": successes,
        "total": total,
        "success_rate": successes / total if total else 0.0,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "solver_status_counts": statuses,
        "mean_elapsed_ms": statistics.fmean(elapsed) if elapsed else 0.0,
        "median_elapsed_ms": statistics.median(elapsed) if elapsed else 0.0,
        "mean_mutable_columns": statistics.fmean(mutable_counts) if mutable_counts else 0.0,
        "median_mutable_columns": statistics.median(mutable_counts) if mutable_counts else 0.0,
    }


def paired_comparisons(policy_results: dict[str, list[dict[str, Any]]], reference: str) -> list[dict[str, Any]]:
    rows = []
    ref_records = {record["case_index"]: record for record in policy_results.get(reference, [])}
    for policy_name, records in sorted(policy_results.items()):
        if policy_name == reference:
            continue
        variant_only = 0
        reference_only = 0
        paired = 0
        ref_successes = 0
        var_successes = 0
        for record in records:
            ref = ref_records.get(record["case_index"])
            if ref is None:
                continue
            paired += 1
            ref_ok = bool(ref.get("target_achieved"))
            var_ok = bool(record.get("target_achieved"))
            ref_successes += int(ref_ok)
            var_successes += int(var_ok)
            if var_ok and not ref_ok:
                variant_only += 1
            if ref_ok and not var_ok:
                reference_only += 1
        rows.append(
            {
                "reference": reference,
                "policy": policy_name,
                "paired_total": paired,
                "reference_successes": ref_successes,
                "policy_successes": var_successes,
                "risk_difference": (var_successes - ref_successes) / paired if paired else 0.0,
                "policy_only_successes": variant_only,
                "reference_only_successes": reference_only,
                "mcnemar_exact_p": exact_mcnemar_p(variant_only, reference_only),
            }
        )
    return rows


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), sort_keys=True) + "\n")


def write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "successes",
        "total",
        "success_rate",
        "wilson_95_low",
        "wilson_95_high",
        "mean_elapsed_ms",
        "median_elapsed_ms",
        "mean_mutable_columns",
        "median_mutable_columns",
        "solver_status_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: json.dumps(row.get(key)) if key == "solver_status_counts" else row.get(key) for key in fieldnames})
