import typer
from vnntest.core.generator import discover_tests, build_test_artifacts
from vnntest.core.runner import run_solver_on_test

app = typer.Typer(help="Run solver test suites")

@app.command()
def mvp(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI (e.g. 'vibecheck' or './run_solver.sh')"),
    tests: str = typer.Option("basic_layers", help="Test suite to run"),
):
    """
    Run the MVP test suite against a solver.
    """
    typer.echo(f"Running tests using solver at {solver}...")
    
    for test_file in discover_tests():
        artifacts = build_test_artifacts(test_file)
        if artifacts is None:
            continue
            
        onnx_path, vnnlib_path = artifacts
        
        typer.echo(f"\nRunning solver on {onnx_path.stem}...")
        output = run_solver_on_test(solver, onnx_path, vnnlib_path)
        typer.echo(f"Output for {onnx_path.stem}:")
        typer.echo(output)

@app.command()
def precision():
    """
    Run precision tests.
    """
    typer.echo("Running precision tests...")