import typer

from vnntest.core.compliance import (
    check_opset_versions, check_vnnlib_versions, check_serialise_assignments,
)

app = typer.Typer(help="Check the solver command-line interface")


@app.command()
def opset(
    solver: str = typer.Option(..., "--solver", help="Solver command or executable path"),
    timeout: int = typer.Option(10, min=1, help="Capability query timeout in seconds"),
):
    """Check supports --onnx-opset-versions (VNN-LIB Section 5.4.1)."""
    passed, message = check_opset_versions(solver, timeout)
    typer.echo("Compliance: supports --onnx-opset-versions")
    typer.echo(f"{'PASS' if passed else 'FAIL'}: {message}")
    if not passed:
        raise typer.Exit(code=1)


@app.command("vnnlib")
def vnnlib_versions(
    solver: str = typer.Option(..., "--solver", help="Solver command or executable path"),
    timeout: int = typer.Option(10, min=1, help="Capability query timeout in seconds"),
):
    """Check supports --vnnlib-versions (VNN-LIB Section 5.4.2)."""
    passed, message = check_vnnlib_versions(solver, timeout)
    typer.echo("Compliance: supports --vnnlib-versions")
    typer.echo(f"{'PASS' if passed else 'FAIL'}: {message}")
    if not passed:
        raise typer.Exit(code=1)


@app.command("serialise-assignments")
def serialise_assignments(
    solver: str = typer.Option(..., "--solver", help="Solver command or executable path"),
    timeout: int = typer.Option(10, min=1, help="Capability query timeout in seconds"),
):
    """Check supports --serialise-assignments (VNN-LIB Section 5.4.3)."""
    passed, message = check_serialise_assignments(solver, timeout)
    typer.echo("Compliance: supports --serialise-assignments")
    typer.echo(f"{'PASS' if passed else 'FAIL'}: {message}")
    if not passed:
        raise typer.Exit(code=1)
