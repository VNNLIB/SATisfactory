import typer
from vnntest.cli import run, report, compliance

app = typer.Typer(help="VNN-LIB SATisfactory -- automated testing for VNN-LIB 2.0 neural network solvers")

# Add subcommands
app.add_typer(compliance.app, name="compliance")
app.command("run", help="Run the test suite against a solver, one test at a time")(run.run)
app.add_typer(report.app, name="report", help="Generate and view test reports")

if __name__ == "__main__":
    app()
