# CITS3200---Team-28: Generic Solver Testing Tool (vnntest)

An automated, mathematically grounded testing suite for VNNLIB 2.0 neural network solvers.

Dependency versions are pinned in 'pyproject.toml' to keep installs reproducible.

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
   You should now have access to the `vnntest` CLI, callable as either `vnntest` or `solver-test-suite`:
   ```bash
   vnntest --help
   solver-test-suite --help
   ```

## Basic usage

Run a test suite against a solver:

```bash
vnntest run mvp --solver vibecheck
```

Useful options for `run mvp` / `run precision`:

| Flag | Purpose |
|---|---|
| `--solver` (required) | Command or path to invoke the solver's CLI |
| `--node-type` | Run only one node type, e.g. `--node-type matmul` |
| `--timeout` | Per-test timeout in seconds (default `5.0`) |
| `--verbose` / `--no-verbose` | Print each test's full raw solver stdout/stderr. Defaults to on when `--node-type` is set, off for a full-suite run |

Example targeting a single node type with verbose output:

```bash
vnntest run mvp --solver vibecheck --node-type matmul --verbose
```

Each run:

1. Generates ONNX + VNNLIB artifacts for every registered test case and records them in `.vnntest/manifest.json`.
2. Queries the solver (via `supports --onnx-operators`) to find out which ops it claims to support, and skips tests for unsupported ops rather than running them.
3. Runs the remaining tests one at a time, printing a pass/fail/suspect/error/skip symbol for each.
4. Writes a detailed report to `.vnntest/report.json`.

## Viewing reports

```bash
vnntest report view
```

Options:

| Flag | Purpose |
|---|---|
| `--format` | `text` (default) or `json` |
| `--report-path` | Path to the report file (default `.vnntest/report.json`) |
| `--category` | Repeatable. Filter to `arithmetic`, `activation`, `structural`, or `other` |
| `--outcome` | Repeatable. Filter to `pass`, `fail`, `suspect`, `error`, or `skip` |

Examples:

```bash
# Only failures and suspect results
vnntest report view --outcome fail --outcome suspect

# Only arithmetic-category results, as JSON
vnntest report view --category arithmetic --format json
```

Categories group node types by failure mode: **arithmetic** (real computation — wrong numbers indicate a real bug), **activation** (elementwise nonlinear functions, prone to boundary/strictness bugs), **structural** (value-preserving reshapes/no-ops, prone to elision or support bugs), and **other** for anything unlisted.

The exit code always reflects the *full, unfiltered* run — a display filter never masks a real failure in CI, even if the printed output is filtered down to `--outcome pass`.

## Compliance checks (Story 7)

```bash
vnntest compliance opset --solver vibecheck
vnntest compliance vnnlib --solver vibecheck
vnntest compliance serialise-assignments --solver vibecheck
```

- `opset`: validates the reported minimum/maximum ONNX opset versions.
- `vnnlib`: validates the reported minimum/maximum VNN-LIB versions.
- `serialise-assignments`: validates that the reported capability is `true` or `false`; either is a valid response.

Each command reports PASS or FAIL for the solver's reported CLI/interface
format and behaviour. These checks do not prove that the solver actually
supports or correctly implements the capabilities it reports. Full compliance
coverage and integration with the report remain future work.

## Adding new tests

vnntest's core (`core/protocols.py`, `core/registry.py`, `core/generator.py`, `core/runner.py`) is not meant to be edited when adding coverage. Two extension points cover everything:

- **New node type** — drop a handwritten case folder under `src/vnntest/core/tests/<node>/`; the handwritten generator picks it up automatically on the next run.
- **New test generator** — add a module under `src/vnntest/generators/` that subclasses `TestGenerator` and calls `register_generator()` at import time.

See [`docs/modular_architecture.md`](docs/modular_architecture.md) for the full step-by-step guide, naming conventions, and the required interface for each case file (`build_onnx_model()`, `build_vnnlib_spec()`, and the optional `return_expected_output()` / `unexpected_result_message()`).