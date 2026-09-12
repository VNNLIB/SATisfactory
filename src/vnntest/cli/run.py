import typer
from vnntest.core.generator import build_test_artifacts
from vnntest.core.registry import get_cases, get_node_types
from vnntest.core.runner import run_solver_on_test

app = typer.Typer(help="Run solver test suites")


def _run_suite(suite: str, solver: str, node_type: str | None) -> None:
    if node_type is not None:
        known = get_node_types()
        if node_type not in known:
            typer.echo(f"Unknown node type {node_type!r}. Known: {', '.join(known)}")
            raise typer.Exit(code=1)

    cases = get_cases(suite=suite, node_type=node_type)
    if not cases:
        typer.echo(f"No test cases registered for suite {suite!r}.")
        return

    typer.echo(f"Running {len(cases)} {suite} test(s) using solver at {solver}...")

    for case in cases:
        artifacts = build_test_artifacts(case)
        if artifacts is None:
            continue

        onnx_path, vnnlib_path = artifacts
        typer.echo(f"\nRunning solver on {case.node_type}/{case.id}...")
        output = run_solver_on_test(solver, onnx_path, vnnlib_path)
        typer.echo(f"Output for {case.id}:")
        typer.echo(output)


@app.command()
def mvp(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI (e.g. 'vibecheck' or './run_solver.sh')"),
    tests: str = typer.Option("basic_layers", help="Legacy suite label; ignored. Use --node-type to filter."),
    node_type: str | None = typer.Option(None, "--node-type", help="Run only this node type (e.g. add, relu, matmul)"),
):
    """
    Run the MVP test suite against a solver.
    """
    _ = tests
    _run_suite("mvp", solver, node_type)


@app.command()
def precision(
    solver: str = typer.Option(..., "--solver", help="Command or path to invoke the solver's CLI"),
    node_type: str | None = typer.Option(None, "--node-type", help="Run only this node type (e.g. add, relu, matmul)"),
):
    """
    Run precision tests.
    """
    _run_suite("precision", solver, node_type)
