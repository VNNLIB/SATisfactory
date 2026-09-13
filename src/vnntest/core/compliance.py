"""Small checks for the VNN-LIB Section 5 solver interface."""

import re
import subprocess

from vnntest.core.runner import run_solver_command


def validate_opset_versions(stdout: str) -> str | None:
    """Return a failure reason, or None for a minimum/maximum opset pair.

    Section 5.4.1 specifies two newline-separated versions. ONNX opsets
    are positive integers, not semantic versions. A final newline is allowed.
    """
    # Accept LF/CRLF and at most one optional final newline.
    lines = stdout.replace("\r\n", "\n").removesuffix("\n").split("\n")
    if len(lines) != 2:
        return "Expected exactly two lines: minimum and maximum ONNX opset versions."
    if any(re.fullmatch(r"[0-9]+", line) is None for line in lines):
        return "Each ONNX opset version must be a positive integer."
    # Compare as strings after removing leading zeroes to avoid integer limits.
    minimum, maximum = (line.lstrip("0") for line in lines)
    if not minimum or not maximum:
        return "Each ONNX opset version must be a positive integer."
    if (len(minimum), minimum) > (len(maximum), maximum):
        return "Minimum ONNX opset version must not exceed maximum."
    return None


def check_opset_versions(solver: str, timeout: int = 10) -> tuple[bool, str]:
    """Query one capability using the shared runner and check its output.

    A non-zero exit is an execution failure (Issue #17), not a format rule
    explicitly stated in Section 5.4.1. This does not test actual opset support.
    """
    try:
        result = run_solver_command(
            solver, ["supports", "--onnx-opset-versions"], timeout, preserve_newlines=True
        )
    except subprocess.TimeoutExpired:
        return False, f"Capability query timed out after {timeout} seconds."
    except (OSError, UnicodeError) as error:
        return False, f"Could not execute capability query: {error}"
    if result.returncode != 0:
        return False, f"Capability query failed with exit code {result.returncode}: {result.stderr.strip()}"
    error = validate_opset_versions(result.stdout)
    if error:
        return False, error
    return True, "Reported minimum/maximum ONNX opset versions have valid format."


def validate_vnnlib_versions(stdout: str) -> str | None:
    """Validate a min/max pair using VNN-LIB's numeric major.minor syntax."""
    lines = stdout.replace("\r\n", "\n").removesuffix("\n").split("\n")
    if len(lines) != 2:
        return "Expected exactly two lines: minimum and maximum VNN-LIB versions."
    if any(re.fullmatch(r"[0-9]+\.[0-9]+", line) is None for line in lines):
        return "Each VNN-LIB version must use numeric major.minor format (e.g. 2.0)."
    # Numeric component ordering, without float rounding or integer limits.
    versions = []
    for line in lines:
        parts = [part.lstrip("0") or "0" for part in line.split(".")]
        versions.append(tuple((len(part), part) for part in parts))
    if versions[0] > versions[1]:
        return "Minimum VNN-LIB version must not exceed maximum."
    return None


def check_vnnlib_versions(solver: str, timeout: int = 10) -> tuple[bool, str]:
    """Query VNN-LIB versions using the shared runner; check reported format only.

    Non-zero exit handling follows Issue #17, as in the opset check.
    """
    try:
        result = run_solver_command(
            solver, ["supports", "--vnnlib-versions"], timeout, preserve_newlines=True
        )
    except subprocess.TimeoutExpired:
        return False, f"Capability query timed out after {timeout} seconds."
    except (OSError, UnicodeError) as error:
        return False, f"Could not execute capability query: {error}"
    if result.returncode != 0:
        return False, f"Capability query failed with exit code {result.returncode}: {result.stderr.strip()}"
    error = validate_vnnlib_versions(result.stdout)
    if error:
        return False, error
    return True, "Reported minimum/maximum VNN-LIB versions have valid format."


def validate_serialise_assignments(stdout: str) -> str | None:
    """Accept exactly true or false, with one optional LF/CRLF ending."""
    value = stdout.removesuffix("\r\n") if stdout.endswith("\r\n") else stdout.removesuffix("\n")
    if value not in ("true", "false"):
        return "Expected exactly true or false, with at most one final LF or CRLF newline."
    return None


def check_serialise_assignments(solver: str, timeout: int = 10) -> tuple[bool, str]:
    """Check the reported boolean, not whether assignment serialisation works.

    Both true and false are valid capability responses. Non-zero exit
    handling follows Issue #17, as in the other compliance checks.
    """
    try:
        result = run_solver_command(
            solver, ["supports", "--serialise-assignments"], timeout, preserve_newlines=True
        )
    except subprocess.TimeoutExpired:
        return False, f"Capability query timed out after {timeout} seconds."
    except (OSError, UnicodeError) as error:
        return False, f"Could not execute capability query: {error}"
    if result.returncode != 0:
        return False, f"Capability query failed with exit code {result.returncode}: {result.stderr.strip()}"
    error = validate_serialise_assignments(result.stdout)
    if error:
        return False, error
    return True, "Reported assignment serialisation capability has valid boolean format."
