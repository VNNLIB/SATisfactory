"""
Run the code by using:

    vnntest run mvp --solver vibecheck
    vnntest run mvp --solver vibecheck --node-type matmul
    vnntest run mvp --solver vibecheck --verbose
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
from vnntest.core.registry import get_node_types
from vnntest.core.runner import SolverRunResult, get_supported_ops, run_solver_on_test
from vnntest.models.test_case import TestCase

app = typer.Typer(help="Run solver test suites, one test at a time, with detailed output")

_SYMBOL = {"pass": "✅", "fail": "❌", "suspect": "🚩", "error": "⚠️", "skip": "⏭️"}


def _build_detail_message(result: EvaluationResult) -> str:
    tc = result.run_result.test_case
    description = tc.description or "(no description provided)"

    # ERROR: solver crashed/timed out, or there's no known verdict to compare against.
    if result.outcome == Outcome.ERROR:
        if result.run_result.timed_out:
            return f"{description} -> solver timed out."
        if result.run_result.error:
            return f"{description} -> solver error: {result.run_result.error}"
        return f"{description} -> no known expected verdict for this test; nothing to compare against."

    # SUSPECT: solver claims support, ran cleanly, still returned "unknown".
    if result.outcome == Outcome.SUSPECT:
        return (
            f"{description} -> solver claims to support this op and ran "
            f"without crashing or timing out, but returned 'unknown' instead "
            f"of a definite verdict (expected '{result.expected.value}')."
        )

    # FAIL: solver gave a definite, wrong verdict.
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


def _print_raw_output(run_result: SolverRunResult) -> None:
    """Dumps a test's full solver stdout/stderr, unabridged -- the same
    everything-it-said-for-every-test behaviour main's cli/run.py has
    always had, offered here as an opt-in alongside the terse/detailed
    modes rather than replacing them."""
    typer.echo("    --- raw solver output ---")
    for line in run_result.stdout.splitlines():
        typer.echo(f"    {line}")
    if run_result.stderr.strip():
        typer.echo("    --- stderr ---")
        for line in run_result.stderr.splitlines():
            typer.echo(f"    {line}")


def _validate_node_type(node_type: str | None) -> None:
    if node_type is None:
        return
    known = get_node_types()
    if node_type not in known:
        typer.echo(f"Unknown node type {node_type!r}. Known: {', '.join(known)}")
        raise typer.Exit(code=1)


def _run_suite(
    suite: str, solver: str, node_type: str | None, timeout: float, verbose: bool = False,
) -> None:
    _validate_node_type(node_type)

    typer.echo("Generating test artifacts...")
    manifest = generate_all_tests(suite=suite, node_type=node_type)
    if not manifest:
        typer.echo(f"No test cases registered for suite {suite!r}"
                    + (f" and node type {node_type!r}." if node_type else "."))
        return

    cache_dir = Path(".vnntest")
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

    typer.echo(f"\nRunning {len(runnable)} test(s) one by one using solver at {solver}...\n")

    eval_results: list[EvaluationResult] = []
    for i, tc in enumerate(runnable, start=1):
        run_result = run_solver_on_test(solver, tc, timeout=timeout)
        eval_result = evaluate(run_result)
        eval_results.append(eval_result)

        symbol = _SYMBOL[eval_result.outcome.value]
        typer.echo(f"Test {i} {symbol} {eval_result.test_id}")
        if eval_result.outcome != Outcome.PASS:
            typer.echo(f"    {_build_detail_message(eval_result)}")
        if verbose:
            _print_raw_output(run_result)

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
                "detail": _build_detail_message(r) if r.outcome != Outcome.PASS else None,
                "stdout": r.run_result.stdout,
                "stderr": r.run_result.stderr,
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


@app.command()
def mvp(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI (e.g. 'vibecheck' or './run_solver.sh')"),
    node_type: str | None = typer.Option(None, "--node-type", help="Run only this node type (e.g. add, relu, matmul)"),
    timeout: float = typer.Option(300.0, help="Per-test timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", help="Also print each test's full raw solver output, sat or unsat, like the original per-test dump"),
):
    """
    Run the MVP test suite against a solver, one test at a time.
    """
    _run_suite("mvp", solver, node_type, timeout, verbose)


@app.command()
def precision(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI"),
    node_type: str | None = typer.Option(None, "--node-type", help="Run only this node type (e.g. add, relu, matmul)"),
    timeout: float = typer.Option(300.0, help="Per-test timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", help="Also print each test's full raw solver output, sat or unsat, like the original per-test dump"),
):
    """
    Run the precision test suite against a solver, one test at a time.
    """
    _run_suite("precision", solver, node_type, timeout, verbose)


if __name__ == "__main__":
    app()
