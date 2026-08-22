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


class SolverHarness:
    """
    Runs an external verifier binary against (onnx, vnnlib) pairs and
    parses its stdout for a sat/unsat/unknown verdict.

    `command_template` controls invocation. Default is the common
    VNN-COMP convention: `<exec> <onnx_path> <vnnlib_path>`.
    Override for flag-based solvers, e.g.:
        "{exec} --onnx {onnx} --vnnlib {vnnlib}"
    """

    def __init__(
        self,
        exec_path: str,
        command_template: str = "{exec} {onnx} {vnnlib}",
        timeout: float = 300.0,
    ):
        self.exec_path = exec_path
        self.command_template = command_template
        self.timeout = timeout

    def _build_command(self, test_case: TestCase) -> list[str]:
        cmd_str = self.command_template.format(
            exec=self.exec_path,
            onnx=str(test_case.onnx_path),
            vnnlib=str(test_case.vnnlib_path),
        )
        return cmd_str.split()

    def _parse_verdict(self, stdout: str) -> Verdict:
        match = _VERDICT_RE.search(stdout)
        if not match:
            return Verdict.UNKNOWN
        return Verdict.from_str(match.group(1))

    def run(self, test_case: TestCase) -> SolverRunResult:
        cmd = self._build_command(test_case)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            return SolverRunResult(
                test_case=test_case, verdict=Verdict.UNKNOWN, returncode=-1,
                stdout=e.stdout or "", stderr=e.stderr or "",
                timed_out=True, error=f"Timed out after {self.timeout}s",
            )
        except FileNotFoundError as e:
            return SolverRunResult(
                test_case=test_case, verdict=Verdict.UNKNOWN, returncode=-1,
                stdout="", stderr="", error=f"Solver executable not found: {e}",
            )

        verdict = self._parse_verdict(proc.stdout)
        return SolverRunResult(
            test_case=test_case, verdict=verdict, returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr,
        )

    def run_all(self, test_cases: list[TestCase]) -> list[SolverRunResult]:
        return [self.run(tc) for tc in test_cases]