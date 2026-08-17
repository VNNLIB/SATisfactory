import typer
from vnntest.cli import run, report

app = typer.Typer(help="Generic Solver Testing Tool")

# Add subcommands
app.add_typer(run.app, name="run", help="Run solver test suites")
app.add_typer(report.app, name="report", help="Generate and view test reports")

if __name__ == "__main__":
    app()
