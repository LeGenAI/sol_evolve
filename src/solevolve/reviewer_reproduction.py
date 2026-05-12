from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact_check import check_manifest
from .codetables import lookup_codetables
from .reviewer_claims import codetables_query_for_claim, expand_reviewer_claim_ids, reproduce_reviewer_claims
from .so_claim_repro import resolve_cadical
from .solver_check import check_solver
from .tracing import file_sha256, flush_tracing, git_sha, traceable_run, update_current_run_context

ROOT_DIR = Path(__file__).resolve().parents[2]


def _status_symbol(status: str | None) -> str:
    if status in {"OK", "PASS", "SAT", "UNSAT", "LIVE", "CACHE"}:
        return "PASS"
    if status in {"STALE", "SKIPPED", "TIMEOUT", "PARTIAL", "UNKNOWN", None}:
        return "PARTIAL"
    return "FAIL"


def _overall_verdict(
    *,
    manifest: dict[str, Any],
    codetables_refs: list[dict[str, Any]],
    claim_summary: dict[str, Any],
) -> str:
    if claim_summary.get("verdict") == "FAIL" or manifest.get("status") == "ERROR":
        return "FAIL"
    codetables_ok = all(ref.get("fetch_status") in {"LIVE", "CACHE", "STALE"} for ref in codetables_refs)
    if manifest.get("status") == "OK" and claim_summary.get("verdict") == "PASS" and codetables_ok:
        return "PASS"
    return "PARTIAL"


def _manifest_file_hashes(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    root = manifest_path.parent.parent if manifest_path.parent.name == "artifacts" else Path.cwd()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    hashes = []
    for relpath in payload.get("bundled", []):
        path = root / relpath
        hashes.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    return hashes


def _claim_rows(claim_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for report in claim_summary.get("reports", []) or []:
        family = report.get("claim_family") or "self_orthogonal"
        matrix = report.get("matrix_verification") or {}
        solver = report.get("solver") or {}
        optimality = report.get("optimality_solver") or {}
        diagnostics = report.get("diagnostics") or {}
        binary_solutions = report.get("solution_reports") or []
        binary_first = binary_solutions[0].get("diagnostics", {}) if binary_solutions else {}
        binary_best = min((item.get("diagnostics", {}).get("A_d", 10**9) for item in binary_solutions), default=None)
        parameters = report.get("extended_parameters") or report.get("parameters")
        rows.append(
            {
                "claim_id": report.get("result_id"),
                "claim_family": family,
                "verdict": report.get("verdict"),
                "field_order": report.get("field_order") or ("2" if family == "binary_codes" else None),
                "parameters": parameters,
                "target_d": report.get("target_minimum_distance"),
                "matrix_ok": matrix.get("matrix_ok", binary_first.get("verification_ok")),
                "minimum_distance": matrix.get("minimum_distance", binary_first.get("minimum_distance")),
                "self_orthogonal": matrix.get("self_orthogonal"),
                "weight_distribution_ok": matrix.get("matches_expected_weight_distribution"),
                "coverage_ok": (diagnostics.get("checks") or {}).get("exact_cover"),
                "vertex_count": diagnostics.get("vertex_count"),
                "center_count": diagnostics.get("center_count"),
                "minimum_pairwise_distance": diagnostics.get("minimum_pairwise_distance"),
                "ball_size_distribution": diagnostics.get("ball_size_distribution"),
                "A_d_progression": report.get("A_d_progression"),
                "best_A_d": report.get("best_A_d", binary_best if binary_best != 10**9 else None),
                "reduction_percent": report.get("reduction_percent"),
                "solver_status": solver.get("status"),
                "solver_elapsed_ms": solver.get("solver_elapsed_ms"),
                "optimality_required": report.get("optimality_required"),
                "optimality_status": optimality.get("status"),
                "optimality_elapsed_ms": optimality.get("solver_elapsed_ms"),
                "missing_obligations": report.get("missing_obligations") or [],
                "report_path": report.get("report_path"),
            }
        )
    return rows


def _check_selected_cadical(cadical_path: str) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(cadical_path)
    resolved_path = str(path.resolve()) if path.exists() else str(path)
    argv = [str(path), "--version"]
    payload: dict[str, Any] = {
        "preferred_solver": "cadical",
        "selected_solver": "cadical",
        "solver": "cadical",
        "selected_path": str(path),
        "selected_resolved_path": resolved_path,
        "selected_source": "reviewer_argument_or_resolver",
        "path": str(path),
        "argv": argv,
        "command": " ".join(argv),
        "cwd": str(ROOT_DIR),
        "checked": [
            {
                "solver": "cadical",
                "path": str(path),
                "resolved_path": resolved_path,
                "source": "reviewer_argument_or_resolver",
                "exists": path.exists(),
                "executable": path.is_file() and path.exists(),
                "preferred": True,
            }
        ],
    }
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=str(ROOT_DIR),
        )
    except Exception as exc:
        payload.update(
            {
                "status": "ERROR",
                "error": str(exc),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return payload

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    payload.update(
        {
            "status": "OK" if result.returncode == 0 else "ERROR",
            "returncode": result.returncode,
            "version": (result.stdout or result.stderr).strip().splitlines()[:3],
            "version_stdout": result.stdout.strip().splitlines()[:3],
            "version_stderr": result.stderr.strip().splitlines()[:3],
            "version_elapsed_ms": elapsed_ms,
            "elapsed_ms": elapsed_ms,
        }
    )
    return payload


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SolEvolve Reviewer Reproduction Report",
        "",
        f"- Verdict: **{payload['verdict']}**",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Git SHA: `{payload.get('git_sha') or 'unknown'}`",
        f"- Paper claim id: `{payload['paper_claim_id']}`",
        f"- Output directory: `{payload['output_dir']}`",
        "",
        "## Environment",
        "",
        f"- Python: `{payload['environment']['python']}`",
        f"- Platform: `{payload['environment']['platform']}`",
        f"- CaDiCaL path: `{payload['environment'].get('cadical_path') or 'not found'}`",
        "",
        "## Artifact Manifest",
        "",
        f"- Status: `{payload['manifest']['status']}`",
        f"- Manifest: `{payload['manifest_path']}`",
        f"- Bundled count: `{payload['manifest'].get('bundled_count')}`",
        "",
        "| Bundled file | Exists | SHA-256 |",
        "| --- | --- | --- |",
    ]
    for item in payload.get("bundled_hashes", []):
        lines.append(f"| `{item['path']}` | `{item['exists']}` | `{item.get('sha256') or ''}` |")

    lines.extend(
        [
            "",
            "## Solver",
            "",
            f"- Solver check status: `{payload['solver_check']['status']}`",
            f"- Selected solver: `{payload['solver_check'].get('selected_solver') or ''}`",
            f"- Selected path: `{payload['solver_check'].get('selected_resolved_path') or payload['solver_check'].get('selected_path') or ''}`",
            "",
            "## codetables.de",
            "",
            "| Claim | Fetch | q | n | k | Bounds | Hash |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for ref in payload.get("codetables_refs", []):
        bounds = f"{ref.get('lower_bound') or ''}..{ref.get('upper_bound') or ''}"
        lines.append(
            f"| `{ref.get('claim_id')}` | `{ref.get('fetch_status')}` | `{ref.get('q')}` | `{ref.get('n')}` | `{ref.get('k')}` | `{bounds}` | `{ref.get('content_sha256') or ''}` |"
        )

    lines.extend(
        [
            "",
            "## Paper Claims",
            "",
            "| Claim | Family | Verdict | Parameters | Deterministic diagnostics | Missing obligations | Report |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload.get("claim_results", []):
        diagnostics = []
        if row.get("minimum_distance") is not None:
            diagnostics.append(f"d_min={row.get('minimum_distance')}")
        if row.get("best_A_d") is not None:
            diagnostics.append(f"best_A_d={row.get('best_A_d')}")
        if row.get("A_d_progression"):
            diagnostics.append(f"A_d={row.get('A_d_progression')}")
        if row.get("coverage_ok") is not None:
            diagnostics.append(f"exact_cover={row.get('coverage_ok')}")
        if row.get("vertex_count") is not None:
            diagnostics.append(f"vertices={row.get('vertex_count')}")
        if row.get("center_count") is not None:
            diagnostics.append(f"centers={row.get('center_count')}")
        if row.get("minimum_pairwise_distance") is not None:
            diagnostics.append(f"pairwise_d_min={row.get('minimum_pairwise_distance')}")
        if row.get("ball_size_distribution") is not None:
            diagnostics.append(f"balls={row.get('ball_size_distribution')}")
        if row.get("solver_status"):
            diagnostics.append(f"SAT={row.get('solver_status')}")
        if row.get("optimality_status"):
            diagnostics.append(f"d+1={row.get('optimality_status')}")
        lines.append(
            f"| `{row.get('claim_id')}` | `{row.get('claim_family')}` | `{row.get('verdict')}` | `{row.get('parameters')}` | `{'; '.join(diagnostics)}` | `{', '.join(row.get('missing_obligations') or [])}` | `{row.get('report_path') or ''}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS` means manifest files exist, required codetables lookups were live/cache-backed, and all selected claim obligations passed.",
            "- `PARTIAL` means deterministic matrix checks may pass, but a solver, network lookup, or optional proof obligation was unavailable or timed out.",
            "- `INSUFFICIENT_ARTIFACT` on a claim means the public release lacks the raw center/matrix artifact needed for a full proof.",
            "- `FAIL` means a required manifest or mathematical proof obligation contradicted the paper claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@traceable_run("solevolve.reviewer_reproduction", run_type="chain")
def run_reviewer_reproduction(
    *,
    paper_claim_id: str = "all_reviewer_core",
    output_dir: str | Path = "artifacts/reviewer_reproduction",
    manifest_path: str | Path | None = None,
    cadical_path: str | None = None,
    timeout: int = 300,
    codetables_timeout: int = 15,
    skip_codetables: bool = False,
    skip_optimality_check: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path) if manifest_path is not None else ROOT_DIR / "artifacts" / "manifest.json"

    manifest_payload = check_manifest(manifest)
    bundled_hashes = _manifest_file_hashes(manifest)
    selected_cadical = resolve_cadical(cadical_path)
    solver_payload = _check_selected_cadical(selected_cadical) if selected_cadical else check_solver(ROOT_DIR, preferred_solver="cadical")

    claim_ids = expand_reviewer_claim_ids(paper_claim_id)
    codetables_refs = []
    if skip_codetables:
        codetables_refs = [
            {
                "claim_id": claim_id,
                "fetch_status": "SKIPPED",
                "error": "codetables lookup skipped by CLI flag",
            }
            for claim_id in claim_ids
        ]
    else:
        for claim_id in claim_ids:
            query = codetables_query_for_claim(claim_id)
            if query is None:
                continue
            ref = lookup_codetables(
                q=query["q"],
                n=query["n"],
                k=query["k"],
                artifact_dir=output_root,
                timeout=codetables_timeout,
                use_cache=True,
            )
            ref["claim_id"] = claim_id
            codetables_refs.append(ref)

    claim_summary = reproduce_reviewer_claims(
        claim_id=paper_claim_id,
        artifact_dir=output_root,
        cadical_path=selected_cadical,
        timeout=timeout,
        prove_optimality=not skip_optimality_check,
    )
    claim_rows = _claim_rows(claim_summary)
    verdict = _overall_verdict(
        manifest=manifest_payload,
        codetables_refs=codetables_refs,
        claim_summary=claim_summary,
    )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "paper_claim_id": paper_claim_id,
        "output_dir": str(output_root),
        "manifest_path": str(manifest),
        "manifest": manifest_payload,
        "bundled_hashes": bundled_hashes,
        "solver_check": solver_payload,
        "codetables_refs": codetables_refs,
        "claim_summary": claim_summary,
        "claim_results": claim_rows,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "cadical_path": selected_cadical,
            "timeout": timeout,
            "codetables_timeout": codetables_timeout,
        },
        "git_sha": git_sha(ROOT_DIR),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    summary_path = output_root / "reviewer_reproduction_summary.json"
    report_path = output_root / "REPRODUCTION_REPORT.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown_report(report_path, payload)
    payload["summary_path"] = str(summary_path)
    payload["report_path"] = str(report_path)
    update_current_run_context(
        metadata={
            "run_kind": "reviewer_reproduction",
            "verdict": verdict,
            "paper_claim_id": paper_claim_id,
            "claim_ids": claim_ids,
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "selected_cadical": selected_cadical,
            "elapsed_ms": payload["elapsed_ms"],
        },
        tags=["reviewer-reproduction", paper_claim_id],
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-facing deterministic SolEvolve reproduction checks.")
    parser.add_argument("--paper-claim-id", default="all_reviewer_core", help="Claim id to reproduce: all_so_table, lucas_cubes, binary_ad_43_10_16, or all_reviewer_core.")
    parser.add_argument("--output-dir", default="artifacts/reviewer_reproduction", help="Directory for reviewer JSON/Markdown reports and generated claim artifacts.")
    parser.add_argument("--manifest", default=None, help="Path to artifacts/manifest.json. Defaults to the packaged manifest.")
    parser.add_argument("--cadical-path", default=None, help="Optional CaDiCaL binary or directory. Directories resolve via build/cadical then cadical.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each SAT solve.")
    parser.add_argument("--codetables-timeout", type=int, default=15, help="Timeout in seconds for each codetables.de live lookup.")
    parser.add_argument("--skip-codetables", action="store_true", help="Skip live/cache codetables lookup and mark those references SKIPPED.")
    parser.add_argument("--skip-optimality-check", action="store_true", help="Skip d+1 SAT checks; use only matrix verification and target-d SAT feasibility.")
    parser.add_argument("--require-pass", action="store_true", help="Exit non-zero unless the aggregate reviewer verdict is PASS.")
    args = parser.parse_args()
    try:
        payload = run_reviewer_reproduction(
            paper_claim_id=args.paper_claim_id,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            cadical_path=args.cadical_path,
            timeout=args.timeout,
            codetables_timeout=args.codetables_timeout,
            skip_codetables=args.skip_codetables,
            skip_optimality_check=args.skip_optimality_check,
        )
        print(
            json.dumps(
                {
                    "verdict": payload["verdict"],
                    "summary_path": payload["summary_path"],
                    "report_path": payload["report_path"],
                    "claim_results": payload["claim_results"],
                    "codetables_refs": payload["codetables_refs"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.require_pass and payload["verdict"] != "PASS":
            raise SystemExit(1)
    finally:
        flush_tracing()


if __name__ == "__main__":
    main()
