import json
from pathlib import Path

import typer

from vnntest.core.evaluator import evaluate_all, summarize
from vnntest.core.generator import generate_all_tests
from vnntest.core.runner import run_all
from vnntest.models.test_case import TestCase

app = typer.Typer(help="Run solver test suites")

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

    typer.echo(f"\nRunning tests using solver at {solver}...")
    run_results = run_all(solver, test_cases, timeout=timeout)
    eval_results = evaluate_all(run_results)

    for r in eval_results:
        symbol = {"pass": "✅", "fail": "❌", "error": "⚠️"}[r.outcome.value]
        typer.echo(f"{symbol} {r.test_id:<20} expected={r.expected.value:<8} actual={r.actual.value:<8}")
        if r.run_result.error:
            typer.echo(f"    ↳ {r.run_result.error}")

    summary = summarize(eval_results)
    typer.echo(f"\n{summary['passed']}/{summary['total']} passed, "
               f"{summary['failed']} failed, {summary['errored']} errored")

    report = {
        "summary": summary,
        "results": [
            {
                "test_id": r.test_id,
                "expected": r.expected.value,
                "actual": r.actual.value,
                "outcome": r.outcome.value,
                "returncode": r.run_result.returncode,
                "error": r.run_result.error,
            }
            for r in eval_results
        ],
    }
    (cache_dir / "report.json").write_text(json.dumps(report, indent=2))

@app.command()
def precision():
    """
    Run precision tests.
    """
    typer.echo("Running precision tests...")
