import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List

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


class Category(str, Enum):
    ARITHMETIC = "arithmetic"
    ACTIVATION = "activation"
    STRUCTURAL = "structural"
    OTHER = "other"


class OutcomeFilter(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SUSPECT = "suspect"
    ERROR = "error"
    SKIP = "skip"


@app.callback()
def report_callback():
    """Generate and view test reports."""
    pass

@app.command()
def view(
    format: str = typer.Option("text", help="Report output format (text, json)"),
    report_path: str = typer.Option(".vnnlib-satisfactory/report.json", help="Path to the report file"),
    category: List[Category] = typer.Option(
        [], "--category", help="Only show these categories (repeatable), e.g. --category arithmetic"
    ),
    outcome: List[OutcomeFilter] = typer.Option(
        [], "--outcome", help="Only show these outcomes (repeatable), e.g. --outcome fail --outcome suspect"
    ),
):
    """
    View a generated test report. With no filters, shows everything.
    """
    path = Path(report_path)
    if not path.exists():
        typer.echo(f"No report found at {path}. Run `vnnlib-satisfactory run --solver <solver>` first.")
        raise typer.Exit(code=1)

    report = json.loads(path.read_text())
    categories = {c.value for c in category}
    outcomes = {o.value for o in outcome}
    filtered = _apply_filters(report, categories, outcomes)

    if format == "json":
        typer.echo(json.dumps(filtered, indent=2))
    else:
        if format != "text":
            typer.echo(f"Unknown format '{format}', falling back to text.")
        _print_text_report(filtered, categories, outcomes)

    # Exit status reflects the full, unfiltered run -- a display filter
    # should never mask a real failure from CI.
    summary = report["summary"]
    if summary["failed"] > 0 or summary["suspect"] > 0 or summary["errored"] > 0:
        raise typer.Exit(code=1)


def _apply_filters(report: dict, categories: set, outcomes: set) -> dict:
    results = report["results"]
    skipped = report.get("skipped", [])

    if categories:
        results = [r for r in results if _CATEGORY_OF.get(r["category"], "other") in categories]
        skipped = [s for s in skipped if _CATEGORY_OF.get(s["category"], "other") in categories]
    if outcomes:
        results = [r for r in results if r["outcome"] in outcomes]
        if "skip" not in outcomes:
            skipped = []

    return {**report, "results": results, "skipped": skipped}


def _print_text_report(report: dict, categories: set, outcomes: set) -> None:
    if categories or outcomes:
        parts = []
        if categories:
            parts.append(f"category in {{{', '.join(sorted(categories))}}}")
        if outcomes:
            parts.append(f"outcome in {{{', '.join(sorted(outcomes))}}}")
        typer.echo(f"Filters: {' and '.join(parts)}")

    grouped = defaultdict(list)
    for r in report["results"]:
        grouped[_CATEGORY_OF.get(r["category"], "other")].append(r)

    shown = 0
    for cat in _CATEGORY_ORDER:
        if cat not in grouped:
            continue
        shown += len(grouped[cat])
        typer.echo(f"\n{cat.capitalize()}")
        typer.echo("-" * len(cat))
        for r in grouped[cat]:
            symbol = _SYMBOL.get(r["outcome"], "?")
            typer.echo(f"{symbol} {r['test_id']:<24} expected={r['expected']:<8} actual={r['actual']:<8}")
            if r["outcome"] in _DETAIL_OUTCOMES:
                _print_failure_detail(r)

    _print_skipped_section(report.get("skipped", []))
    shown += len(report.get("skipped", []))

    if shown == 0:
        typer.echo("No tests match the given filters.")
        return

    _print_filtered_summary(report["results"], report.get("skipped", []))


def _print_filtered_summary(results: list, skipped: list) -> None:
    counts = {o.value: 0 for o in OutcomeFilter}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    counts["skip"] += len(skipped)

    typer.echo(
        f"\nShown: {len(results) + len(skipped)}  Passed: {counts['pass']}  "
        f"Failed: {counts['fail']}  Suspect: {counts['suspect']}  "
        f"Errored: {counts['error']}  Skipped: {counts['skip']}"
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
