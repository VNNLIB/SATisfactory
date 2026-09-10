import json
from pathlib import Path

import typer

from vnntest.core.evaluator import evaluate_all, partition_by_support, summarize
from vnntest.core.generator import generate_all_tests
from vnntest.core.runner import get_supported_ops, run_all
from vnntest.models.test_case import TestCase

app = typer.Typer(help="Run solver test suites")

_SYMBOL = {"pass": "✅", "fail": "❌", "suspect": "🚩", "error": "⚠️", "skip": "⏭️"}

@app.command()
def mvp(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI (e.g. 'vibecheck' or './run_solver.sh')"),
    tests: str = typer.Option("basic_layers", help="Test suite to run"),
    timeout: float = typer.Option(300.0, help="Per-test timeout in seconds"),
):
    """
    Run the MVP test suite against a solver.
    """
    typer.echo("Generating test artifacts...")
    generate_all_tests()

    cache_dir = Path(".vnntest")
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    test_cases = [
        TestCase.from_manifest_entry(e, cache_dir / "tests" / e["category"])
        for e in manifest
    ]

    typer.echo(f"\nChecking what {solver} claims to support...")
    supported_ops = get_supported_ops(solver)
    if supported_ops is None:
        typer.echo("  (solver does not implement `supports`; running every test)")
    runnable, skipped = partition_by_support(test_cases, supported_ops)
    if skipped:
        typer.echo(f"  Skipping {len(skipped)} test(s) for unsupported op types.")

    typer.echo(f"\nRunning tests using solver at {solver}...")
    run_results = run_all(solver, runnable, timeout=timeout)
    eval_results = evaluate_all(run_results)

    for r in eval_results:
        symbol = _SYMBOL[r.outcome.value]
        typer.echo(f"{symbol} {r.test_id:<24} expected={r.expected.value:<8} actual={r.actual.value:<8}")
        if r.run_result.error:
            typer.echo(f"    -> {r.run_result.error}")
    for s in skipped:
        typer.echo(f"{_SYMBOL['skip']} {s.test_id:<24} (missing: {s.missing_op})")

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

@app.command()
def precision():
    """
    Run precision tests.
    """
    typer.echo("Running precision tests...")
