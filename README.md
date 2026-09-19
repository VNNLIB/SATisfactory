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

## How to add a new node type

Do **not** edit `src/vnntest/core/generator.py`, `registry.py`, `runner.py`, or `src/vnntest/cli/run.py`.

Agreed interfaces (see `src/vnntest/core/protocols.py`):

- `NodeType` — operator metadata. `name` is the folder name (`add`, `neg`).
- `TestCase` — one ONNX model + VNNLIB query (`build_onnx_model`, `build_vnnlib_spec`).
- `TestGenerator` — a plugin that yields `TestCase` objects.

### Node type (handwritten cases)

1. Create `src/vnntest/core/tests/<node>/`.
2. Add `<node>01.py` (UNSAT) and `<node>02.py` (SAT) when that split applies.
3. Each file must define `build_onnx_model()` and `build_vnnlib_spec()`. Optional: `return_expected_output()` and `unexpected_result_message()`.

The handwritten generator picks the folder up on the next run:

```bash
vnntest run mvp --solver vibecheck --node-type neg
```

Worked example on this branch: `src/vnntest/core/tests/neg/` (`neg01.py`, `neg02.py`). No core execution files were changed for that add.

### Test generator (new strategy)

1. Create `src/vnntest/generators/<name>.py`.
2. Subclass `TestGenerator`, implement `generate()`, set `name` and `suite`.
3. Call `register_generator(YourGenerator())` at module import.

More detail: `docs/modular_architecture.md`.
