"""
Run the code by using:

    vnntest run mvp --solver vibecheck
"""

import json
from pathlib import Path

import typer

from vnntest.core.evaluator import Outcome, evaluate, partition_by_support
from vnntest.core.generator import generate_all_tests
from vnntest.cli.report_builder import build_detail_message, build_report, write_report
from vnntest.core.runner import get_supported_ops, run_solver_on_test
from vnntest.models.test_case import TestCase

app = typer.Typer(help="Run solver test suites, one test at a time, with detailed output")

_SYMBOL = {"pass": "✅", "fail": "❌", "suspect": "🚩", "error": "⚠️", "skip": "⏭️"}


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
    generate_all_tests()  # Ensures local .vnntest/ directory and manifests exist

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

    # Iterate through tests sequentially and stream status lines immediately
    eval_results = []
    for i, tc in enumerate(runnable, start=1):
        run_result = run_solver_on_test(solver, tc, timeout=timeout)
        eval_result = evaluate(run_result)
        eval_results.append(eval_result)

        symbol = _SYMBOL[eval_result.outcome.value]
        typer.echo(f"Test {i} {symbol} {eval_result.test_id}")
        if eval_result.outcome != Outcome.PASS:
            typer.echo(f"    {build_detail_message(eval_result)}")

    # Log skipped tests at bottom of console output
    for s in skipped:
        typer.echo(f"{_SYMBOL['skip']} {s.test_id:<24} (missing: {s.missing_op})")

    # Build, print summary of, and persist the report
    report = build_report(eval_results, skipped)
    summary = report["summary"]
    typer.echo(
        f"\n{summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['suspect']} suspect, "
        f"{summary['errored']} errored, {summary['skipped']} skipped"
    )

    report_path = cache_dir / "report.json"
    write_report(report, report_path)
    typer.echo(f"\nDetailed report written to {report_path}")


if __name__ == "__main__":
    app()