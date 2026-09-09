import json
from collections import defaultdict
from pathlib import Path

import typer

app = typer.Typer(help="Generate and view test reports")

# Groups node types by what kind of thing can go wrong: arithmetic is real
# computation (wrong numbers = a real bug), activation is elementwise
# nonlinear functions (prone to strictness/boundary bugs), structural is
# value-preserving reshapes/no-ops (prone to elision/support bugs). Anything
# not listed falls back to "other" rather than erroring.
_CATEGORY_OF = {
    "add": "arithmetic", "sub": "arithmetic", "mul": "arithmetic", "div": "arithmetic",
    "matmul": "arithmetic", "gemm": "arithmetic", "batchnormalization": "arithmetic",
    "conv": "arithmetic", "pow": "arithmetic",
    "relu": "activation", "sigmoid": "activation", "softmax": "activation",
    "reshape": "structural", "unsqueeze": "structural", "shape": "structural",
    "flatten": "structural", "squeeze": "structural", "transpose": "structural",
    "concat": "structural", "gather": "structural", "slice": "structural",
    "constant": "structural",
}
_CATEGORY_ORDER = ["arithmetic", "activation", "structural", "other"]
_SYMBOL = {"pass": "✅", "fail": "❌", "suspect": "🚩", "error": "⚠️", "skip": "⏭️"}
_DETAIL_OUTCOMES = {"fail", "suspect", "error"}

@app.callback()
def report_callback():
    """Generate and view test reports."""
    pass

@app.command()
def view(
    format: str = typer.Option("text", help="Report output format (text, json)"),
    report_path: str = typer.Option(".vnntest/report.json", help="Path to the report file"),
):
    """
    View a generated test report.
    """
    path = Path(report_path)
    if not path.exists():
        typer.echo(f"No report found at {path}. Run `vnntest run mvp --solver <solver>` first.")
        raise typer.Exit(code=1)

    report = json.loads(path.read_text())

    if format == "json":
        typer.echo(json.dumps(report, indent=2))
        return
    if format != "text":
        typer.echo(f"Unknown format '{format}', falling back to text.")

    _print_text_report(report)

    summary = report["summary"]
    if summary["failed"] > 0 or summary["suspect"] > 0 or summary["errored"] > 0:
        raise typer.Exit(code=1)


def _print_text_report(report: dict) -> None:
    grouped = defaultdict(list)
    for r in report["results"]:
        grouped[_CATEGORY_OF.get(r["category"], "other")].append(r)

    for cat in _CATEGORY_ORDER:
        if cat not in grouped:
            continue
        typer.echo(f"\n{cat.capitalize()}")
        typer.echo("-" * len(cat))
        for r in grouped[cat]:
            symbol = _SYMBOL.get(r["outcome"], "?")
            typer.echo(f"{symbol} {r['test_id']:<24} expected={r['expected']:<8} actual={r['actual']:<8}")
            if r["outcome"] in _DETAIL_OUTCOMES:
                _print_failure_detail(r)

    _print_skipped_section(report.get("skipped", []))

    summary = report["summary"]
    typer.echo(
        f"\nTotal: {summary['total']}  Passed: {summary['passed']}  "
        f"Failed: {summary['failed']}  Suspect: {summary['suspect']}  "
        f"Errored: {summary['errored']}  Skipped: {summary['skipped']}"
    )


def _print_failure_detail(r: dict) -> None:
    if r.get("description"):
        typer.echo(f"    {r['description']}")
    if r.get("unexpected_message") and r["outcome"] in ("fail", "suspect"):
        typer.echo(f"    -> {r['unexpected_message']}")
    if r.get("error"):
        typer.echo(f"    -> error: {r['error']}")

    vnnlib_path = r.get("vnnlib_path")
    if vnnlib_path and Path(vnnlib_path).exists():
        typer.echo("    query:")
        for line in Path(vnnlib_path).read_text().strip().splitlines():
            typer.echo(f"      {line}")


def _print_skipped_section(skipped: list) -> None:
    if not skipped:
        return
    typer.echo(f"\nNot tested -- solver does not claim support ({len(skipped)})")
    typer.echo("-" * 50)
    for s in skipped:
        typer.echo(f"{_SYMBOL['skip']} {s['test_id']:<24} missing: {s['missing_op']}")
