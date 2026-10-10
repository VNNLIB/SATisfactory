from __future__ import annotations

import locale
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from vnntest.models.test_case import TestCase, Verdict

# Matches a standalone sat/unsat/unknown token, case-insensitive.
# Word boundaries avoid "unsat" false-matching inside other words.
_VERDICT_RE = re.compile(r"\b(unsat|sat|unknown)\b", re.IGNORECASE)


@dataclass
class SolverRunResult:
    test_case: TestCase
    verdict: Verdict
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error: Optional[str] = None


def parse_verdict(stdout: str) -> Verdict:
    match = _VERDICT_RE.search(stdout)
    if not match:
        return Verdict.UNKNOWN
    return Verdict.from_str(match.group(1))


def run_solver_command(
    exec_path: str, arguments: list[str], timeout: int = 10, *, preserve_newlines: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run any solver subcommand, preserving stdout, stderr and the exit code.

    Set preserve_newlines for strict CLI format checks (e.g. compliance
    checks that care about trailing newlines). Errors reach the caller.
    """
    result = subprocess.run(
        [exec_path, *arguments], capture_output=True, text=not preserve_newlines, timeout=timeout
    )
    if preserve_newlines:
        encoding = locale.getpreferredencoding(False)
        return subprocess.CompletedProcess(
            result.args, result.returncode,
            result.stdout.decode(encoding), result.stderr.decode(encoding),
        )
    return result


def get_supported_ops(exec_path: str) -> Optional[set[str]]:
    """
    Returns the set of ONNX op types the solver claims to support, via its
    `supports --onnx-operators` command (VNN-LIB 2.0 Chapter 5).

    Returns None if the check itself is unavailable (the solver doesn't
    implement `supports`, times out, etc.) -- callers should treat that as
    "unknown", not as "supports nothing".
    """
    try:
        result = run_solver_command(exec_path, ["supports", "--onnx-operators"], timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    ops = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return ops or None


def get_supported_network_theories(exec_path: str) -> Optional[set[str]]:
    """
    Returns the set of network theories the solver claims to support, via its
    `supports --multiple-network-theories` command (VNN-LIB 2.0 Chapter 5).
    """
    try:
        result = run_solver_command(exec_path, ["supports", "--multiple-network-theories"], timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    theories = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return theories or None


def get_supported_hidden_node_theories(exec_path: str) -> Optional[set[str]]:
    """
    Returns the set of hidden node theories the solver claims to support, via its
    `supports --hidden-node-theories` command (VNN-LIB 2.0 Chapter 5).
    """
    try:
        result = run_solver_command(exec_path, ["supports", "--hidden-node-theories"], timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    theories = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return theories or None


def get_supported_io_theories(exec_path: str) -> Optional[set[str]]:
    """
    Returns the set of input/output theories the solver claims to support, via its
    `supports --multiple-input-output-theories` command (VNN-LIB 2.0 Chapter 5).

    Some solvers add a note after the theory name on the same line,
    e.g. "SIO * multi-input models run only through ...". We only keep
    the first word (e.g. "SIO"), since that's the actual theory name.
    """
    try:
        result = run_solver_command(exec_path, ["supports", "--multiple-input-output-theories"], timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    theories = {line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()}
    return theories or None


def get_supported_arithmetic_complexity_theories(exec_path: str) -> Optional[set[str]]:
    """
    Returns the set of arithmetic complexity theories the solver claims to
    support, via its `supports --arithmetic-complexity-theories` command
    (VNN-LIB 2.0 Chapter 4.1.5 / Chapter 5).

    A theory name may be followed by a free-text caveat on the same line,
    e.g. "POLY * polynomial constraints transpiled via nonlinear-augment" --
    only the first whitespace-delimited token on each line is the theory
    name itself.
    """
    try:
        result = run_solver_command(exec_path, ["supports", "--arithmetic-complexity-theories"], timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    theories = {line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()}
    return theories or None


def run_solver_on_test(exec_path: str, test_case: TestCase, timeout: int = 300) -> SolverRunResult:
    """
    Executes a VNNLIB 2.0 compatible solver on a single test case.

    The CLI standard (VNN-LIB 2.0 Chapter 5) is:
    <solver> verify <vnnlib_path> --network <name>=<onnx_path> [--timeout <seconds>]

    The verdict is parsed from stdout only, per the standard -- stderr is
    kept separate for error reporting, not verdict parsing.
    """
    cmd = [
        "verify",
        str(test_case.vnnlib_path),
    ]
    if hasattr(test_case, "onnx_paths") and test_case.onnx_paths:
        for name, path in test_case.onnx_paths.items():
            cmd.extend(["--network", f"{name}={path}"])
    else:
        cmd.extend(["--network", f"N={test_case.onnx_path}"])
    cmd.extend(["--timeout", str(timeout)])

    try:
        result = run_solver_command(exec_path, cmd, timeout)
    except subprocess.TimeoutExpired as e:
        return SolverRunResult(
            test_case=test_case, verdict=Verdict.UNKNOWN, returncode=-1,
            stdout=e.stdout or "", stderr=e.stderr or "",
            timed_out=True, error=f"Timed out after {timeout} seconds",
        )
    except FileNotFoundError as e:
        return SolverRunResult(
            test_case=test_case, verdict=Verdict.UNKNOWN, returncode=-1,
            stdout="", stderr="", error=f"Solver executable not found: {e}",
        )

    return SolverRunResult(
        test_case=test_case,
        verdict=parse_verdict(result.stdout),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_all(exec_path: str, test_cases: list[TestCase], timeout: int = 300) -> list[SolverRunResult]:
    return [run_solver_on_test(exec_path, tc, timeout=timeout) for tc in test_cases]
