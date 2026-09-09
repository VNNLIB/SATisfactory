from __future__ import annotations

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


def run_solver_on_test(exec_path: str, test_case: TestCase, timeout: int = 300) -> SolverRunResult:
    """
    Executes a VNNLIB 2.0 compatible solver on a single test case.

    The CLI standard (VNN-LIB 2.0 Chapter 5) is:
    <solver> verify <vnnlib_path> --network <name>=<onnx_path> [--timeout <seconds>]

    The verdict is parsed from stdout only, per the standard -- stderr is
    kept separate for error reporting, not verdict parsing.
    """
    cmd = [
        exec_path,
        "verify",
        str(test_case.vnnlib_path),
        "--network", f"N={test_case.onnx_path}",
        "--timeout", str(timeout),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
