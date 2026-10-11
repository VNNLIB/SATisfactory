# Installation and first use

VNN-LIB SATisfactory runs test cases against a neural network solver and reports
whether its answers match the expected results.

## Requirements

- Python 3.11 or newer, with pip. If Python is not installed, get it from
  [python.org](https://www.python.org/downloads/).
- Internet access to download the packages.
- A separately installed VNN-LIB 2.0-compatible solver. The steps below use
  `vibecheck` as an example.

## Create a virtual environment

Open a terminal in a writable folder where you want to work. A virtual environment
keeps these packages separate from other Python projects.

### macOS/Linux

Check that the version is 3.11 or newer, then create and activate the environment:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

Check that the version is 3.11 or newer, then create and activate the environment:

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt in the same folder and run
`.venv\Scripts\activate.bat`, then continue with the commands below.

Keep the environment active for installation and use. When you open a new terminal,
return to this folder and run the activation command again.

## Install VNN-LIB SATisfactory

With the environment active, check pip and install the package from PyPI:

```bash
python -m pip --version
python -m pip install --upgrade pip
python -m pip install vnnlib-satisfactory
```

If pip is missing, run `python -m ensurepip --upgrade` and retry. On Linux, if
Python reports that venv or ensurepip is unavailable, install your distribution's
Python venv/pip packages and recreate the environment.

You do not need to clone the repository for a PyPI installation.

## Install the example solver

The solver is a separate package. Install it into the same active environment:

```bash
python -m pip install vibecheck-nn
vibecheck --version
```

The package name is `vibecheck-nn`; the command it installs is `vibecheck`.
`--solver vibecheck` finds that executable through PATH, which includes the active
virtual environment's commands. For solver-specific dependencies and platform
notes, see the [vibecheck installation instructions](https://pypi.org/project/vibecheck-nn/).

## Verify the installation

```bash
vnnlib-satisfactory --help
```

You should see help listing the `run`, `compliance`, and `report` commands.
This check does not require a solver.

## Run your first test

Run the small Relu test category, then view the report:

```bash
vnnlib-satisfactory run --solver vibecheck --test relu --timeout 30
vnnlib-satisfactory report view
```

The timeout is 30 seconds per test. Generated ONNX models, VNN-LIB specifications,
and reports are stored under `.vnnlib-satisfactory/` in the current working
directory. Run `report view` from that same directory. The report is saved as
`.vnnlib-satisfactory/report.json`; a later run replaces the default report.

## Install from source

For developers or users with repository access, obtain the repository and open a
terminal in its root folder (the folder containing `pyproject.toml`). With your
virtual environment active, use this instead of the PyPI installation step:

```bash
python -m pip install .
```

For development, use `python -m pip install -e .` instead. This editable installation
uses your working source files so code edits take effect without reinstalling.
The solver installation, verification, and first-run steps above still apply.

## Troubleshooting

- **`vnnlib-satisfactory: command not found`:** activate the environment used for
  installation. Check `python -m pip show vnnlib-satisfactory`; if it is missing,
  repeat the installation step. You can also try `python -m vnntest --help`.
- **`vibecheck: command not found` or solver executable not found:** activate the
  environment and check `python -m pip show vibecheck-nn`. Install it if missing.
  On macOS/Linux, `command -v vibecheck` shows its path; in PowerShell, use
  `Get-Command vibecheck`.
- **Wrong virtual environment:** run `python -c "import sys; print(sys.executable)"`.
  The path should point inside your `.venv`. Reactivate the correct environment
  and use `python -m pip` to install into that Python environment.
- **Solver installed elsewhere or not on PATH:** pass its executable path, for
  example `--solver "/absolute/path/to/vibecheck"` on macOS/Linux or
  `--solver "C:\path\to\Scripts\vibecheck.exe"` on Windows. Supply an executable
  path, not a command string such as `"python -m vibecheck"`.
- **Timeout, SKIP, or FAIL:** these do not necessarily mean installation failed.
  A timeout means the solver exceeded the limit; try increasing `--timeout`.
  SKIP means a required capability was reported as unsupported. FAIL means the
  solver's verdict disagreed with the expected result. Read the output and report
  for details before reinstalling the package.
