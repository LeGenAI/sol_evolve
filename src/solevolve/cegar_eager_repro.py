from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .so_claim_repro import resolve_cadical
from .tracing import file_sha256, traceable_run

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "scripts"

SO_CEGAR_EAGER_CLAIM_ID = "so_52_26_d8_cegar_eager"
BINARY_CEGAR_EAGER_CLAIM_ID = "binary_43_10_cegar_eager"
CEGAR_EAGER_CLAIM_IDS = [SO_CEGAR_EAGER_CLAIM_ID, BINARY_CEGAR_EAGER_CLAIM_ID]

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_thresholds() -> list[int]:
    raw = os.getenv("SOLEVOLVE_CEGAR_EAGER_SO_THRESHOLDS")
    if not raw:
        return [3, 4, 5]
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values or [3, 4, 5]


def _tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
    }


def _run_script_arm(
    *,
    arm_id: str,
    script: Path,
    args: list[str],
    out_dir: Path,
    timeout_sec: int,
    max_rounds: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script), *args, "--out-dir", str(out_dir)]
    env = os.environ.copy()
    pythonpath_parts = [str(ROOT_DIR / "src"), str(SCRIPTS_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    wall_timeout = max(120, int(timeout_sec) * max(1, int(max_rounds) + 1) + 600)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=wall_timeout,
            check=False,
        )
        runner_status = "OK" if result.returncode == 0 else "ERROR"
        error = None if result.returncode == 0 else f"returncode={result.returncode}"
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired as exc:
        runner_status = "TIMEOUT"
        error = f"wrapper timeout after {wall_timeout}s"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = None
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    summary_path = out_dir / "summary.json"
    summary = _safe_load_json(summary_path) or {}
    final_status = summary.get("final_status")
    rounds = summary.get("rounds") if isinstance(summary.get("rounds"), list) else []
    last_round = rounds[-1] if rounds and isinstance(rounds[-1], dict) else {}
    solver_status = final_status or last_round.get("status")
    witness_found = bool(summary.get("witness_found"))
    if witness_found:
        status = "PASS"
    elif solver_status == "SAT":
        status = "SAT_WITH_VIOLATIONS"
    else:
        status = str(solver_status or runner_status)
    artifact_paths = [summary_path]
    matrix_path = summary.get("matrix_path")
    if matrix_path:
        resolved_matrix_path = Path(matrix_path)
        if not resolved_matrix_path.is_absolute():
            resolved_matrix_path = ROOT_DIR / resolved_matrix_path
        artifact_paths.append(resolved_matrix_path)
    return {
        "arm_id": arm_id,
        "command": command,
        "timeout_sec": timeout_sec,
        "max_rounds": max_rounds,
        "returncode": returncode,
        "runner_status": runner_status,
        "status": status,
        "solver_status": solver_status,
        "witness_found": witness_found,
        "elapsed_ms": elapsed_ms,
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path) if summary_path.exists() else None,
        "artifacts": [_artifact_record(path) for path in artifact_paths],
        "rounds": rounds,
        "verification": summary.get("verification"),
        "error": error,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _skipped_arm(arm_id: str, reason: str) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "runner_status": "SKIPPED",
        "status": "SKIPPED",
        "solver_status": "SKIPPED",
        "witness_found": False,
        "elapsed_ms": 0.0,
        "artifacts": [],
        "rounds": [],
        "verification": None,
        "error": reason,
    }


def _best_d_min(arms: list[dict[str, Any]]) -> int | None:
    values: list[int] = []
    for arm in arms:
        verification = arm.get("verification") if isinstance(arm.get("verification"), dict) else {}
        d_min = verification.get("d_min")
        if d_min is not None:
            values.append(int(d_min))
    return max(values) if values else None


def _weight_distribution_from_best(arms: list[dict[str, Any]], d_min: int | None) -> dict[str, int] | None:
    if d_min is None:
        return None
    for arm in arms:
        verification = arm.get("verification") if isinstance(arm.get("verification"), dict) else {}
        if verification.get("d_min") == d_min and isinstance(verification.get("weight_distribution"), dict):
            return {str(k): int(v) for k, v in verification["weight_distribution"].items()}
    return None


def _arm_metric(arm: dict[str, Any], key: str) -> Any:
    rounds = arm.get("rounds") if isinstance(arm.get("rounds"), list) else []
    last_round = rounds[-1] if rounds and isinstance(rounds[-1], dict) else {}
    return last_round.get(key)


def _source_artifacts(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in arms:
        for artifact in arm.get("artifacts") or []:
            if isinstance(artifact, dict) and artifact.get("path"):
                path = Path(str(artifact["path"]))
                records.append(_artifact_record(path))
    seen: set[str] = set()
    unique = []
    for record in records:
        path = record["path"]
        if path not in seen:
            unique.append(record)
            seen.add(path)
    return unique


def _run_so_claim(*, artifact_dir: Path, cadical: str, timeout: int, smoke: bool) -> dict[str, Any]:
    max_rounds = 1 if smoke else _env_int("SOLEVOLVE_CEGAR_EAGER_SO_MAX_ROUNDS", 12)
    thresholds = [3] if smoke else _env_thresholds()
    out_root = artifact_dir / "cegar_eager" / SO_CEGAR_EAGER_CLAIM_ID
    arms: list[dict[str, Any]] = []
    for threshold in thresholds:
        arms.append(
            _run_script_arm(
                arm_id=f"cegar_w{threshold}",
                script=SCRIPTS_DIR / "sat_52_26_d8_so_embedding.py",
                args=[
                    "--mode",
                    "cegar",
                    "--initial-max-base-weight",
                    str(threshold),
                    "--max-rounds",
                    str(max_rounds),
                    "--timeout-sec",
                    str(timeout),
                    "--cadical-path",
                    cadical,
                ],
                out_dir=out_root / f"cegar_w{threshold}",
                timeout_sec=timeout,
                max_rounds=max_rounds,
            )
        )
    if smoke:
        arms.append(_skipped_arm("eager_all", "SOLEVOLVE_CEGAR_EAGER_SMOKE=1 skips the 29,016-constraint eager SO arm."))
    else:
        arms.append(
            _run_script_arm(
                arm_id="eager_all",
                script=SCRIPTS_DIR / "sat_52_26_d8_so_embedding.py",
                args=[
                    "--mode",
                    "eager",
                    "--max-rounds",
                    "1",
                    "--timeout-sec",
                    str(timeout),
                    "--cadical-path",
                    cadical,
                ],
                out_dir=out_root / "eager_all",
                timeout_sec=timeout,
                max_rounds=1,
            )
        )
    cegar_arms = [arm for arm in arms if str(arm["arm_id"]).startswith("cegar")]
    eager_arm = next((arm for arm in arms if arm["arm_id"] == "eager_all"), None)
    cegar_success = any(arm.get("witness_found") for arm in cegar_arms)
    eager_recorded = eager_arm is not None and eager_arm.get("status") not in {None, "ERROR"}
    eager_timeout_or_skip = eager_arm is not None and eager_arm.get("status") in {"TIMEOUT", "SKIPPED"}
    main_text_candidate = bool(cegar_success and eager_timeout_or_skip and not smoke)
    d_min = _best_d_min(arms)
    distribution = _weight_distribution_from_best(arms, d_min)
    missing = []
    if not cegar_success:
        missing.append("cegar_witness")
    if not eager_recorded:
        missing.append("eager_arm_record")
    if smoke:
        missing.append("full_eager_arm_skipped_in_smoke")
    verdict = "PASS" if cegar_success and eager_recorded and not smoke else "PARTIAL"
    report = {
        "schema_version": 1,
        "result_id": SO_CEGAR_EAGER_CLAIM_ID,
        "claim_id": SO_CEGAR_EAGER_CLAIM_ID,
        "claim_family": "cegar_eager_scaling",
        "domain": "self_orthogonal_embedding",
        "verdict": verdict,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {"source": "[31,26,3] Hamming H5", "target": "[52,26]", "target_d": 8},
        "target_minimum_distance": 8,
        "arms": arms,
        "cegar_eager_summary": {
            "main_text_candidate": main_text_candidate,
            "cegar_success": cegar_success,
            "eager_recorded": eager_recorded,
            "eager_status": eager_arm.get("status") if eager_arm else None,
            "best_d_min": d_min,
            "cegar_active_constraints": [
                _arm_metric(arm, "active_constraints") for arm in cegar_arms if _arm_metric(arm, "active_constraints") is not None
            ],
            "eager_active_constraints": _arm_metric(eager_arm, "active_constraints") if eager_arm else None,
        },
        "diagnostics": {
            "d_min": d_min,
            "weight_distribution": distribution,
            "checks": {
                "cegar_success": cegar_success,
                "eager_recorded": eager_recorded,
                "main_text_candidate": main_text_candidate,
            },
        },
        "missing_obligations": missing,
        "smoke_mode": smoke,
    }
    report_path = out_root / "claim_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    report["source_artifacts"] = _source_artifacts(arms)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_binary_claim(*, artifact_dir: Path, cadical: str, timeout: int, smoke: bool) -> dict[str, Any]:
    max_rounds = 1 if smoke else _env_int("SOLEVOLVE_CEGAR_EAGER_BINARY_MAX_ROUNDS", 60)
    out_root = artifact_dir / "cegar_eager" / BINARY_CEGAR_EAGER_CLAIM_ID
    arms: list[dict[str, Any]] = []
    for target_d in (16, 17):
        if smoke and target_d == 17 and _truthy_env("SOLEVOLVE_CEGAR_EAGER_SKIP_FRONTIER_SMOKE"):
            arms.extend(
                [
                    _skipped_arm("d17_cegar", "frontier d=17 skipped by smoke setting"),
                    _skipped_arm("d17_eager", "frontier d=17 skipped by smoke setting"),
                ]
            )
            continue
        for mode in ("cegar", "eager"):
            arm_rounds = max_rounds if mode == "cegar" else 1
            args = [
                "--target-d",
                str(target_d),
                "--mode",
                mode,
                "--max-rounds",
                str(arm_rounds),
                "--timeout-sec",
                str(timeout),
                "--cadical-path",
                cadical,
            ]
            arms.append(
                _run_script_arm(
                    arm_id=f"d{target_d}_{mode}",
                    script=SCRIPTS_DIR / "sat_43_10_d17_campaign.py",
                    args=args,
                    out_dir=out_root / f"d{target_d}_{mode}",
                    timeout_sec=timeout,
                    max_rounds=arm_rounds,
                )
            )
    primary = [arm for arm in arms if str(arm["arm_id"]).startswith("d16_")]
    frontier = [arm for arm in arms if str(arm["arm_id"]).startswith("d17_")]
    primary_cegar = next((arm for arm in primary if arm.get("arm_id") == "d16_cegar"), None)
    primary_recorded = len(primary) == 2 and all(arm.get("status") not in {None, "ERROR"} for arm in primary)
    primary_decisive = any(arm.get("status") in {"PASS", "UNSAT"} for arm in primary)
    frontier_decisive = any(arm.get("status") in {"PASS", "UNSAT"} for arm in frontier)
    frontier_symmetric_timeout = bool(frontier) and all(arm.get("status") == "TIMEOUT" for arm in frontier)
    cegar_clean = primary_cegar is not None and primary_cegar.get("status") == "PASS"
    main_text_candidate = bool(primary_recorded and cegar_clean and not smoke)
    d_min = _best_d_min(arms)
    distribution = _weight_distribution_from_best(arms, d_min)
    missing = []
    if not primary_recorded:
        missing.append("binary_d16_primary_comparison")
    if not cegar_clean:
        missing.append("binary_d16_cegar_clean_witness")
    if not frontier_decisive:
        missing.append("binary_d17_frontier_decision")
    if smoke:
        missing.append("full_binary_campaign_not_run_in_smoke")
    verdict = "PASS" if primary_recorded and cegar_clean and not smoke else "PARTIAL"
    report = {
        "schema_version": 1,
        "result_id": BINARY_CEGAR_EAGER_CLAIM_ID,
        "claim_id": BINARY_CEGAR_EAGER_CLAIM_ID,
        "claim_family": "cegar_eager_scaling",
        "domain": "binary_linear_code",
        "verdict": verdict,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {"target": "[43,10]", "primary_d": 16, "frontier_d": 17},
        "target_minimum_distance": 16,
        "arms": arms,
        "cegar_eager_summary": {
            "main_text_candidate": main_text_candidate,
            "primary_recorded": primary_recorded,
            "primary_decisive": primary_decisive,
            "d16_cegar_clean": cegar_clean,
            "frontier_decisive": frontier_decisive,
            "frontier_symmetric_timeout": frontier_symmetric_timeout,
            "best_d_min": d_min,
            "d16_statuses": {arm["arm_id"]: arm.get("status") for arm in primary},
            "d17_statuses": {arm["arm_id"]: arm.get("status") for arm in frontier},
        },
        "diagnostics": {
            "d_min": d_min,
            "weight_distribution": distribution,
            "checks": {
                "primary_recorded": primary_recorded,
                "primary_decisive": primary_decisive,
                "frontier_decisive": frontier_decisive,
                "main_text_candidate": main_text_candidate,
            },
        },
        "missing_obligations": missing,
        "smoke_mode": smoke,
    }
    report_path = out_root / "claim_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    report["source_artifacts"] = _source_artifacts(arms)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


@traceable_run("solevolve.cegar_eager_repro", run_type="chain")
def reproduce_cegar_eager_claim(
    *,
    claim_id: str,
    artifact_dir: str | Path = "artifacts",
    cadical_path: str | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    cadical = resolve_cadical(cadical_path)
    if not cadical:
        return {
            "schema_version": 1,
            "result_id": claim_id,
            "claim_id": claim_id,
            "claim_family": "cegar_eager_scaling",
            "verdict": "SKIPPED",
            "parameters": {},
            "target_minimum_distance": None,
            "diagnostics": {"checks": {"cadical_available": False}},
            "missing_obligations": ["cadical_solver"],
            "error": "CaDiCaL binary not found.",
        }
    normalized = claim_id.strip().lower()
    artifact_root = Path(artifact_dir)
    smoke = _truthy_env("SOLEVOLVE_CEGAR_EAGER_SMOKE")
    started = time.perf_counter()
    if normalized == SO_CEGAR_EAGER_CLAIM_ID:
        report = _run_so_claim(artifact_dir=artifact_root, cadical=cadical, timeout=timeout, smoke=smoke)
    elif normalized == BINARY_CEGAR_EAGER_CLAIM_ID:
        report = _run_binary_claim(artifact_dir=artifact_root, cadical=cadical, timeout=timeout, smoke=smoke)
    else:
        raise KeyError(f"unknown CEGAR/eager claim id: {claim_id}")
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    report["cadical_path"] = cadical
    report_path = Path(str(report["report_path"]))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
