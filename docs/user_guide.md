# VNN-LIB SATisfactory user guide

This guide is for **using** the tool after it is installed. It does not cover install steps. For those, use the installation guide (or **Local Setup** in the project `README.md`).

The command-line name is `vnnlib-satisfactory`.

Confirm the CLI is available:

```bash
vnnlib-satisfactory --help
```

You also need a VNN-LIB solver on your `PATH` (or a path to its executable). Examples below use `vibecheck`.

---

## Run the test suite

Full suite against a solver:

```bash
vnnlib-satisfactory run --solver vibecheck
```

`--solver` is required. Use the solver command name or a path such as `./run_solver.sh`.

### Options

| Flag | Purpose |
|---|---|
| `--solver` | Solver CLI command or path (required) |
| `--test` | Run only one node type / category, e.g. `--test matmul` |
| `--timeout` | Seconds allowed per test (default `5.0`) |
| `--verbose` / `--no-verbose` | Print the VNN-LIB query, full solver stdout/stderr, and expected vs actual verdict for **every** test. Default: on when `--test` is set, off for a full-suite run |

Examples:

```bash
# One node type (verbose is on by default)
vnnlib-satisfactory run --solver vibecheck --test matmul

# Same run, verbose forced on
vnnlib-satisfactory run --solver vibecheck --test matmul --verbose

# Full suite, still print every test in detail
vnnlib-satisfactory run --solver vibecheck --verbose

# Longer per-test budget
vnnlib-satisfactory run --solver vibecheck --timeout 30
```

Unknown `--test` values are rejected. The CLI lists the node types it knows.

### What a run does

1. Builds ONNX and VNN-LIB artifacts for the selected tests and records them in `.vnnlib-satisfactory/manifest.json`.
2. Asks the solver which ONNX operators it claims to support (`supports --onnx-operators`). Tests for unsupported ops are skipped, not failed.
3. Runs the remaining tests **one at a time** and prints a symbol for each result.
4. Writes `.vnnlib-satisfactory/report.json`.

If the solver has no `supports` command, every generated test is run.

---

## How to read results

Each test is compared to a known expected verdict (`sat` or `unsat`) when one exists.

| Outcome | Meaning |
|---|---|
| **pass** | Solver verdict matches the expected answer |
| **fail** | Solver returned a definite `sat` or `unsat` that disagrees with the expected answer |
| **suspect** | Solver claims to support the op, did not crash or time out, but returned `unknown` |
| **error** | Timeout, crash, or the test has no expected verdict to compare against |
| **skip** | Solver does not claim support for an operator the test needs |

A line looks like: `Test 3 ✅  add/add01` (or fail / suspect / error / skip). Non-pass results include a short explanation. With `--verbose`, every test also dumps the query and raw solver output.

The summary at the end is:

```text
passed/total passed, N failed, N suspect, N errored, N skipped
```

---

## View a report

After a run:

```bash
vnnlib-satisfactory report view
```

| Flag | Purpose |
|---|---|
| `--format` | `text` (default) or `json` |
| `--report-path` | Report file (default `.vnnlib-satisfactory/report.json`) |
| `--category` | Repeatable filter: `arithmetic`, `activation`, `structural`, `other` |
| `--outcome` | Repeatable filter: `pass`, `fail`, `suspect`, `error`, `skip` |

Examples:

```bash
vnnlib-satisfactory report view --outcome fail --outcome suspect
vnnlib-satisfactory report view --category arithmetic --format json
```

If there is no report yet, run `vnnlib-satisfactory run --solver <solver>` first.

**Categories** (for filtering, not for choosing `--test`):

- **arithmetic** — real computation (Add, MatMul, Conv, …). Wrong numbers usually mean a real bug.
- **activation** — elementwise nonlinear ops (Relu, Sigmoid, …). Often boundary or strictness bugs.
- **structural** — reshapes and similar (Reshape, Concat, Slice, …). Often elision or support bugs.
- **other** — anything not in those lists.

**Exit code:** `report view` uses the **full unfiltered** run. Filters only change what is printed. If any test failed, was suspect, or errored, the command still exits with status `1`, even if you filtered the display to `--outcome pass`. That keeps CI honest.

---

## Compliance checks

These commands check what the solver **reports** on its CLI. They do not prove the solver implements those capabilities correctly.

```bash
vnnlib-satisfactory compliance opset --solver vibecheck
vnnlib-satisfactory compliance vnnlib --solver vibecheck
vnnlib-satisfactory compliance serialise-assignments --solver vibecheck
```

| Command | What it checks |
|---|---|
| `opset` | Reported min/max ONNX opset (`supports --onnx-opset-versions`) |
| `vnnlib` | Reported min/max VNN-LIB versions (`supports --vnnlib-versions`) |
| `serialise-assignments` | Reported capability is `true` or `false` (`supports --serialise-assignments`) |

Each prints `PASS` or `FAIL`. Optional `--timeout` (default 10 seconds) is the capability-query timeout, not the test-run timeout.

---

## Files a run leaves behind

Under `.vnnlib-satisfactory/` in the working directory:

| Path | Contents |
|---|---|
| `manifest.json` | List of generated tests |
| `report.json` | Last run: summary, per-test outcomes, solver output |
| `tests/` | Generated `.onnx` and `.vnnlib` files |

---

## Typical workflow

1. Install the tool (installation guide / README).
2. `vnnlib-satisfactory run --solver <solver>`.
3. Scan the summary. For a failing op, rerun with `--test <node>` (verbose is on by default).
4. `vnnlib-satisfactory report view --outcome fail --outcome suspect`.
5. Optionally run the three `compliance` commands.

This guide does not cover adding new tests. That is in `docs/modular_architecture.md`.
