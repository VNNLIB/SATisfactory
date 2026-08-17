import typer

app = typer.Typer(help="Run solver test suites")

@app.command()
def mvp(
    exec_path: str = typer.Option(..., "--exec", help="Path to the solver executable"),
    tests: str = typer.Option("basic_layers", help="Test suite to run"),
):
    """
    Run the MVP test suite against a solver.
    """
    typer.echo(f"Running MVP tests ({tests}) using solver at {exec_path}")

@app.command()
def precision():
    """
    Run precision tests.
    """
    typer.echo("Running precision tests...")
