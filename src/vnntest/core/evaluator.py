from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vnntest.core.harness import SolverRunResult
from vnntest.models.test_case import Verdict


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"    # solver disagreed with known answer
    ERROR = "error"  # crashed, timed out, or unparseable output


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


def evaluate(run_result: SolverRunResult) -> EvaluationResult:
    if run_result.error or run_result.timed_out:
        return EvaluationResult(run_result, Outcome.ERROR)

    expected = run_result.test_case.expected
    actual = run_result.verdict

    if expected == Verdict.UNKNOWN or actual == Verdict.UNKNOWN:
        return EvaluationResult(run_result, Outcome.ERROR)

    return EvaluationResult(
        run_result, Outcome.PASS if actual == expected else Outcome.FAIL,
    )


def evaluate_all(run_results: list[SolverRunResult]) -> list[EvaluationResult]:
    return [evaluate(r) for r in run_results]


def summarize(results: list[EvaluationResult]) -> dict:
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.outcome == Outcome.PASS),
        "failed": sum(1 for r in results if r.outcome == Outcome.FAIL),
        "errored": sum(1 for r in results if r.outcome == Outcome.ERROR),
    }