# VNN-LIB SATisfactory

An automated, mathematically grounded testing suite for VNN-LIB 2.0 neural network solvers.

Dependency version ranges are declared in `pyproject.toml`.

## Installation

See the [installation guide](docs/installation.md) for PyPI installation,
solver setup, and your first test on Windows, macOS, or Linux.

## Local Setup

To set up the project locally for development, follow these steps:

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Upgrade pip**:
   Ensure you have the latest version of `pip` so that editable installs work correctly.
   ```bash
   pip install --upgrade pip
   ```

3. **Install the project in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Verify the installation**:
   You should now have access to the CLI, callable as `vnnlib-satisfactory`:
   ```bash
   vnnlib-satisfactory --help
   ```

## Basic usage

Run a test suite against a solver:

```bash
vnnlib-satisfactory run --solver vibecheck
```

Useful options for `run`:

| Flag | Purpose |
|---|---|
| `--solver` (required) | Command or path to invoke the solver's CLI |
| `--test` | Run only one test category, e.g. `--test matmul` |
| `--timeout` | Per-test timeout in seconds (default `5.0`) |
| `--verbose` / `--no-verbose` | For every test, print the VNN-LIB query that was run, the solver's full raw stdout/stderr, and the expected vs. actual verdict -- not just for failures. Defaults to on when `--test` is set, off for a full-suite run. Behaves identically regardless of how it's triggered (explicit flag or the test-category default) |

Example targeting a single test category with verbose output:

```bash
vnnlib-satisfactory run --solver vibecheck --test matmul --verbose
```

Each run:

1. Generates ONNX + VNNLIB artifacts for every registered test case and records them in `.vnnlib-satisfactory/manifest.json`.
2. Queries the solver (via `supports --onnx-operators`) to find out which ops it claims to support, and skips tests for unsupported ops rather than running them.
3. Runs the remaining tests one at a time, printing a pass/fail/suspect/error/skip symbol for each.
4. Writes a detailed report to `.vnnlib-satisfactory/report.json`.

## Viewing reports

```bash
vnnlib-satisfactory report view
```

Options:

| Flag | Purpose |
|---|---|
| `--format` | `text` (default) or `json` |
| `--report-path` | Path to the report file (default `.vnnlib-satisfactory/report.json`) |
| `--category` | Repeatable. Filter to `arithmetic`, `activation`, `structural`, or `other` |
| `--outcome` | Repeatable. Filter to `pass`, `fail`, `suspect`, `error`, or `skip` |

Examples:

```bash
# Only failures and suspect results
vnnlib-satisfactory report view --outcome fail --outcome suspect

# Only arithmetic-category results, as JSON
vnnlib-satisfactory report view --category arithmetic --format json
```

Categories group node types by failure mode: **arithmetic** (real computation — wrong numbers indicate a real bug), **activation** (elementwise nonlinear functions, prone to boundary/strictness bugs), **structural** (value-preserving reshapes/no-ops, prone to elision or support bugs), and **other** for anything unlisted.

The exit code always reflects the *full, unfiltered* run — a display filter never masks a real failure in CI, even if the printed output is filtered down to `--outcome pass`.

## Compliance checks (Story 7)

```bash
vnnlib-satisfactory compliance opset --solver vibecheck
vnnlib-satisfactory compliance vnnlib --solver vibecheck
vnnlib-satisfactory compliance serialise-assignments --solver vibecheck
```

- `opset`: validates the reported minimum/maximum ONNX opset versions.
- `vnnlib`: validates the reported minimum/maximum VNN-LIB versions.
- `serialise-assignments`: validates that the reported capability is `true` or `false`; either is a valid response.

Each command reports PASS or FAIL for the solver's reported CLI/interface
format and behaviour. These checks do not prove that the solver actually
supports or correctly implements the capabilities it reports. Full compliance
coverage and integration with the report remain future work.

## Adding new tests

The tool's core (`core/protocols.py`, `core/registry.py`, `core/generator.py`, `core/runner.py`) is not meant to be edited when adding coverage. Two extension points cover everything:

- **New node type** — drop a handwritten case folder under `src/vnntest/core/tests/<node>/`; the handwritten generator picks it up automatically on the next run.
- **New test generator** — add a module under `src/vnntest/generators/` that subclasses `TestGenerator` and calls `register_generator()` at import time.

See [`docs/modular_architecture.md`](docs/modular_architecture.md) for the full step-by-step guide, naming conventions, and the required interface for each case file (`build_onnx_model()`, `build_vnnlib_spec()`, and the optional `return_expected_output()` / `unexpected_result_message()`).
