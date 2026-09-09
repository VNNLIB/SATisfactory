from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vnntest.core.runner import SolverRunResult
from vnntest.models.test_case import TestCase, Verdict


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"        # solver disagreed with the known answer
    SUSPECT = "suspect"  # solver claims support, ran cleanly, still returned "unknown"
    ERROR = "error"      # crashed, timed out, or no known answer to compare against
    SKIP = "skip"        # solver does not claim to support a required op type


@dataclass
class EvaluationResult:
    run_result: SolverRunResult
    outcome: Outcome

    @property
    def test_id(self) -> str:
        return self.run_result.test_case.id

    @property
    def expected(self) -> Verdict:
        return self.run_result.test_case.expected

    @property
    def actual(self) -> Verdict:
        return self.run_result.verdict


@dataclass
class SkippedResult:
    test_case: TestCase
    missing_op: str

    @property
    def test_id(self) -> str:
        return self.test_case.id


def evaluate(run_result: SolverRunResult) -> EvaluationResult:
    if run_result.error or run_result.timed_out:
        return EvaluationResult(run_result, Outcome.ERROR)

    expected = run_result.test_case.expected
    actual = run_result.verdict

    # No known answer for this test at all -- nothing to compare against.
    if expected == Verdict.UNKNOWN:
        return EvaluationResult(run_result, Outcome.ERROR)

    # Solver claims support, ran without crashing/timing out, and still came
    # back "unknown" -- per the client, this is a real signal worth flagging
    # separately, not routine incompleteness.
    if actual == Verdict.UNKNOWN:
        return EvaluationResult(run_result, Outcome.SUSPECT)

    return EvaluationResult(
        run_result, Outcome.PASS if actual == expected else Outcome.FAIL,
    )


def evaluate_all(run_results: list[SolverRunResult]) -> list[EvaluationResult]:
    return [evaluate(r) for r in run_results]


def partition_by_support(
    test_cases: list[TestCase], supported_ops: Optional[set[str]]
) -> tuple[list[TestCase], list[SkippedResult]]:
    """
    Splits test cases into those the solver claims to support and those it
    doesn't, so unsupported ones are never run at all.

    Gates only on each test's primary_op_type (the node type it's actually
    testing), not every op in the network -- most tests also contain a
    Constant node to carry fixed values, and solvers commonly fold that away
    without declaring it a supported capability, so requiring it too would
    wrongly skip almost everything.

    If supported_ops is None (the solver's `supports` command is
    unavailable) or a test has no determinable primary_op_type, that test
    is not skipped -- we have no solid basis to filter it out.
    """
    if supported_ops is None:
        return test_cases, []

    runnable: list[TestCase] = []
    skipped: list[SkippedResult] = []
    for tc in test_cases:
        if tc.primary_op_type and tc.primary_op_type not in supported_ops:
            skipped.append(SkippedResult(tc, tc.primary_op_type))
        else:
            runnable.append(tc)
    return runnable, skipped


def summarize(
    results: list[EvaluationResult], skipped: Optional[list[SkippedResult]] = None
) -> dict:
    skipped = skipped or []
    return {
        "total": len(results) + len(skipped),
        "passed": sum(1 for r in results if r.outcome == Outcome.PASS),
        "failed": sum(1 for r in results if r.outcome == Outcome.FAIL),
        "suspect": sum(1 for r in results if r.outcome == Outcome.SUSPECT),
        "errored": sum(1 for r in results if r.outcome == Outcome.ERROR),
        "skipped": len(skipped),
    }
