import typer
from vnntest.core.generator import generate_all_tests

app = typer.Typer(help="Run solver test suites")

@app.command()
def mvp(
    exec_path: str = typer.Option(..., "--exec", help="Path to the solver executable"),
    tests: str = typer.Option("basic_layers", help="Test suite to run"),
):
    """
    Run the MVP test suite against a solver.
    """
    typer.echo("Generating test artifacts...")
    generate_all_tests()
    
    typer.echo(f"\nRunning tests using solver at {exec_path}...")
    # TODO: Pass the generated files in .vnntest/tests to your SolverHarness

@app.command()
def precision():
    """
    Run precision tests.
    """
    typer.echo("Running precision tests...")