import typer

app = typer.Typer(help="Generate and view test reports")

@app.command()
def view(
    format: str = typer.Option("text", help="Report output format (text, json)"),
):
    """
    View a generated test report.
    """
    typer.echo(f"Viewing report in {format} format")
