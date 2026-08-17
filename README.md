# CITS3200---Team-28: Generic Solver Testing Tool (vnntest)

An automated, mathematically grounded testing suite for VNNLIB 2.0 neural network solvers.

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
   You should now have access to the `vnntest` CLI command.
   ```bash
   vnntest --help
   ```