from __future__ import annotations

import json
from pathlib import Path

from vnntest.core.evaluator import EvaluationResult, Outcome, SkippedResult, summarize


def build_detail_message(result: EvaluationResult) -> str:
    """Human-readable explanation for any non-passing result."""
    tc = result.run_result.test_case
    description = tc.description or "(no description provided)"

    if result.outcome == Outcome.ERROR:
        if result.run_result.timed_out:
            return f"{description} -> solver timed out."
        if result.run_result.error:
            return f"{description} -> solver error: {result.run_result.error}"
        return f"{description} -> no known expected verdict for this test; nothing to compare against."

    if result.outcome == Outcome.SUSPECT:
        return (
            f"{description} -> solver claims to support this op and ran "
            f"without crashing or timing out, but returned 'unknown' instead "
            f"of a definite verdict (expected '{result.expected.value}')."
        )

    if result.outcome == Outcome.FAIL:
        if tc.unexpected_message:
            return (
                f"{description} -> {tc.unexpected_message} "
                f"(expected={result.expected.value}, actual={result.actual.value})"
            )
        return (
            f"{description} -> expected verdict '{result.expected.value}' "
            f"but solver returned '{result.actual.value}'."
        )

    return ""  # PASS never reaches this -- callers only invoke it on non-pass


def build_report(
    eval_results: list[EvaluationResult], skipped: list[SkippedResult]
) -> dict:
    """Assembles the full report dict (summary + per-test detail) from raw results."""
    return {
        "summary": summarize(eval_results, skipped),
        "results": [
            {
                "test_id": r.test_id,
                "category": r.run_result.test_case.category,
                "op_types": r.run_result.test_case.op_types,
                "description": r.run_result.test_case.description,
                "expected": r.expected.value,
                "actual": r.actual.value,
                "outcome": r.outcome.value,
                "returncode": r.run_result.returncode,
                "error": r.run_result.error,
                "unexpected_message": r.run_result.test_case.unexpected_message,
                "detail": build_detail_message(r) if r.outcome != Outcome.PASS else None,
                "onnx_path": str(r.run_result.test_case.onnx_path),
                "vnnlib_path": str(r.run_result.test_case.vnnlib_path),
            }
            for r in eval_results
        ],
        "skipped": [
            {
                "test_id": s.test_id,
                "category": s.test_case.category,
                "missing_op": s.missing_op,
            }
            for s in skipped
        ],
    }


def write_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2))