"""
Run the code by using:

    vnnlib-satisfactory run --solver vibecheck
    vnnlib-satisfactory run --solver vibecheck --test matmul
    vnnlib-satisfactory run --solver vibecheck --verbose
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
from vnntest.core.runner import (
    SolverRunResult,
    get_supported_ops,
    get_supported_network_theories,
    get_supported_hidden_node_theories,
    run_solver_on_test,
)
from vnntest.models.test_case import TestCase

_SYMBOL = {"pass": "✅", "fail": "❌", "suspect": "🚩", "error": "⚠️", "skip": "⏭️"}


def _build_detail_message(result: EvaluationResult) -> str:
    tc = result.run_result.test_case
    description = tc.description or "(no description provided)"

    def _with_unexpected(base: str) -> str:
        """Appends the test's own unexpected_result_message(), when it
        defines one, on top of the outcome-specific explanation -- the
        whole point of that field is to explain what this specific test
        was checking, so every non-pass outcome should surface it, not
        just FAIL."""
        return f"{base} {tc.unexpected_message}" if tc.unexpected_message else base

    # ERROR: solver crashed/timed out, or there's no known verdict to compare against.
    if result.outcome == Outcome.ERROR:
        if result.run_result.timed_out:
            return _with_unexpected(f"{description} -> solver timed out.")
        if result.run_result.error:
            return _with_unexpected(f"{description} -> solver error: {result.run_result.error}")
        return _with_unexpected(f"{description} -> no known expected verdict for this test; nothing to compare against.")

    # SUSPECT: solver claims support, ran cleanly, still returned "unknown".
    if result.outcome == Outcome.SUSPECT:
        return _with_unexpected(
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


def _print_verbose_detail(run_result: SolverRunResult) -> None:
    """Shown for every test when --verbose is on, not just failures: the
    query that was run, the solver's full raw stdout/stderr, and the
    expected vs. actual verdict."""
    tc = run_result.test_case
    if tc.vnnlib_path and Path(tc.vnnlib_path).exists():
        typer.echo("    --- query ---")
        for line in Path(tc.vnnlib_path).read_text().strip().splitlines():
            typer.echo(f"    {line}")
    typer.echo("    --- raw solver output ---")
    for line in run_result.stdout.splitlines():
        typer.echo(f"    {line}")
    if run_result.stderr.strip():
        typer.echo("    --- stderr ---")
        for line in run_result.stderr.splitlines():
            typer.echo(f"    {line}")
    typer.echo(f"    expected={tc.expected.value}  actual={run_result.verdict.value}")


def _validate_test(test: str | None) -> None:
    if test is None:
        return
    known = get_node_types()
    if test not in known:
        typer.echo(f"Unknown test category {test!r}. Known: {', '.join(known)}")
        raise typer.Exit(code=1)


def _run_suite(
    suite: str, solver: str, test: str | None, timeout: float, verbose: bool | None = None,
) -> None:
    _validate_test(test)
    # No explicit --verbose/--no-verbose: default to on when drilling into a
    # single test category (that's a small, deliberate run -- see the output),
    # off for a full-suite run (that's already a lot of lines).
    if verbose is None:
        verbose = test is not None

    typer.echo("Generating test artifacts...")
    manifest = generate_all_tests(suite=suite, node_type=test)
    if not manifest:
        typer.echo(f"No test cases registered for suite {suite!r}"
                    + (f" and test category {test!r}." if test else "."))
        return

    cache_dir = Path(".vnnlib-satisfactory")
    test_cases = [
        TestCase.from_manifest_entry(e, cache_dir / "tests" / e["category"])
        for e in manifest
    ]

    typer.echo(f"\nChecking what {solver} claims to support...")
    supported_ops = get_supported_ops(solver)
    supported_network_theories = get_supported_network_theories(solver)
    supported_hidden_node_theories = get_supported_hidden_node_theories(solver)
    if supported_ops is None and supported_network_theories is None and supported_hidden_node_theories is None:
        typer.echo("  (solver does not implement `supports`; running every test)")
    runnable, skipped = partition_by_support(test_cases, supported_ops, supported_network_theories, supported_hidden_node_theories)
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
            _print_verbose_detail(run_result)

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


def run(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI (e.g. 'vibecheck' or './run_solver.sh')"),
    test: str | None = typer.Option(None, "--test", help="Run only this test category (e.g. add, relu, network_multiple)"),
    timeout: float = typer.Option(5.0, help="Per-test timeout in seconds"),
    verbose: bool | None = typer.Option(None, "--verbose/--no-verbose", help="Print each test's full raw solver output, sat or unsat. Defaults to on with --test, off for a full run"),
):
    """
    Run the test suite against a solver, one test at a time.
    """
    _run_suite("full", solver, test, timeout, verbose)
