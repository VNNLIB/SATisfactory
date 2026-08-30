import json
from pathlib import Path
import typer

app = typer.Typer(help="Generate and view test reports")

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
        typer.echo(f"No report found at {path}. Run `vnntest run mvp --exec <solver>` first.")
        raise typer.Exit(code=1)

    report = json.loads(path.read_text())

    if format == "json":
        typer.echo(json.dumps(report, indent=2))
        return

    if format != "text":
        typer.echo(f"Unknown format '{format}', falling back to text.")

    summary = report["summary"]
    results = report["results"]

    symbol = {"pass": "✅", "fail": "❌", "error": "⚠️"}

    typer.echo(f"{'Test':<20} {'Expected':<10} {'Actual':<10} {'Outcome':<8}")
    typer.echo("-" * 52)
    for r in results:
        s = symbol.get(r["outcome"], "?")
        typer.echo(f"{r['test_id']:<20} {r['expected']:<10} {r['actual']:<10} {s} {r['outcome']}")
        if r["error"]:
            typer.echo(f"    ↳ error: {r['error']}")

    typer.echo("-" * 52)
    typer.echo(
        f"Total: {summary['total']}  "
        f"Passed: {summary['passed']}  "
        f"Failed: {summary['failed']}  "
        f"Errored: {summary['errored']}"
    )

    if summary["failed"] > 0 or summary["errored"] > 0:
        raise typer.Exit(code=1)
