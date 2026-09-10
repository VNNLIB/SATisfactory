"""
Run the code by using:

    vnntest run mvp --solver vibecheck
"""

import json
from pathlib import Path

import typer

from vnntest.core.evaluator import (
    EvaluationResult,
    Outcome,
    evaluate,
    partition_by_support,
    summarize,
)
from vnntest.core.generator import generate_all_tests
from vnntest.core.runner import get_supported_ops, run_solver_on_test
from vnntest.models.test_case import TestCase

#Initialise typer CLI app instance to help with documentation
app = typer.Typer(help="Run solver test suites, one test at a time, with detailed output")

_SYMBOL = {"pass": "✅", "fail": "❌", "suspect": "🚩", "error": "⚠️", "skip": "⏭️"}


def _build_detail_message(result: EvaluationResult) -> str:

    #Code for explaining non-passable results for solver
    tc = result.run_result.test_case
    description = tc.description or "(no description provided)"

    #ERROR: Solver crashed/timed out OR missing known benchmark verdict.
    # Handles system-level errors, timeouts, or unmapped expected values 
    if result.outcome == Outcome.ERROR:
        if result.run_result.timed_out:
            return f"{description} -> solver timed out."
        if result.run_result.error:
            return f"{description} -> solver error: {result.run_result.error}"
        return f"{description} -> no known expected verdict for this test; nothing to compare against."

    #SUSPECT:Solver didn't crash but returned 'unknown' on supported ops.
    #Handle inconclusive results on supported operations 
    if result.outcome == Outcome.SUSPECT:
        return (
            f"{description} -> solver claims to support this op and ran "
            f"without crashing or timing out, but returned 'unknown' instead "
            f"of a definite verdict (expected '{result.expected.value}')."
        )

    #FAIL: FAIL: Disagreement between expected verdict and actual solver response.
    #Handle explicit answer mismatches, prioritizing custom failure messages if present
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


@app.command()
def mvp(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI (e.g. 'vibecheck' or './run_solver.sh')"),
    tests: str = typer.Option("basic_layers", help="Test suite to run"),
    timeout: float = typer.Option(300.0, help="Per-test timeout in seconds"),
):
    """
    Run the MVP test suite against a solver, one test at a time.
    """
    typer.echo("Generating test artifacts...")
    generate_all_tests() #Ensures local .vnntest/ directory and manifests exist

    #Execcution loop setup
    cache_dir = Path(".vnntest")
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    test_cases = [
        TestCase.from_manifest_entry(e, cache_dir / "tests" / e["category"])
        for e in manifest
    ]

    # Query solver capability to skip unsupported ONNX/VNNLIB operations
    typer.echo(f"\nChecking what {solver} claims to support...")
    supported_ops = get_supported_ops(solver)
    if supported_ops is None:
        typer.echo("  (solver does not implement `supports`; running every test)")
    runnable, skipped = partition_by_support(test_cases, supported_ops)
    if skipped:
        typer.echo(f"  Skipping {len(skipped)} test(s) for unsupported op types.")

    typer.echo(f"\nRunning {len(runnable)} test(s) one by one using solver at {solver}...\n")

    #Iterate through tests sequentially and stream status lines immediately
    eval_results: list[EvaluationResult] = []
    for i, tc in enumerate(runnable, start=1):
        run_result = run_solver_on_test(solver, tc, timeout=timeout)
        eval_result = evaluate(run_result)
        eval_results.append(eval_result)

        symbol = _SYMBOL[eval_result.outcome.value]
        typer.echo(f"Test {i} {symbol} {eval_result.test_id}")
        if eval_result.outcome != Outcome.PASS:
            typer.echo(f"    {_build_detail_message(eval_result)}")

    # Log skipped tests at bottom of console output
    for s in skipped:
        typer.echo(f"{_SYMBOL['skip']} {s.test_id:<24} (missing: {s.missing_op})")

    #Print total and format JSON report
    summary = summarize(eval_results, skipped)
    typer.echo(
        f"\n{summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['suspect']} suspect, "
        f"{summary['errored']} errored, {summary['skipped']} skipped"
    )

    report = {
        "summary": summary,
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
                "detail": _build_detail_message(r) if r.outcome != Outcome.PASS else None,
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
    (cache_dir / "report.json").write_text(json.dumps(report, indent=2))
    typer.echo(f"\nDetailed report written to {cache_dir / 'report.json'}")


if __name__ == "__main__":
    app()