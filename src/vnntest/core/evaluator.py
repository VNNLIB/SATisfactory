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
    test_cases: list[TestCase],
    supported_ops: Optional[set[str]],
    supported_network_theories: Optional[set[str]] = None,
    supported_hidden_node_theories: Optional[set[str]] = None,
    supported_io_theories: Optional[set[str]] = None,
    supported_arithmetic_complexity_theories: Optional[set[str]] = None,
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
    if (
        supported_ops is None
        and supported_network_theories is None
        and supported_hidden_node_theories is None
        and supported_arithmetic_complexity_theories is None
        and supported_io_theories is None
    ):
        return test_cases, []

    runnable: list[TestCase] = []
    skipped: list[SkippedResult] = []
    for tc in test_cases:
        if tc.category.startswith("network_"):
            theory_map = {
                "network_single": "SNET",
                "network_multiple": "MNET",
                "network_equal": "MENET",
                "network_isomorphic": "MINET",
            }
            req = theory_map.get(tc.category)
            if req and supported_network_theories is not None and req not in supported_network_theories:
                skipped.append(SkippedResult(tc, req))
            else:
                runnable.append(tc)
        elif tc.category == "hidden":
            if supported_hidden_node_theories is not None and "H" not in supported_hidden_node_theories:
                skipped.append(SkippedResult(tc, "H theory"))
            else:
                runnable.append(tc)
        elif tc.category == "multi_io":
            if supported_io_theories is not None and "MIO" not in supported_io_theories:
                skipped.append(SkippedResult(tc, "MIO theory"))
            else:
                runnable.append(tc)
        elif tc.category.startswith("arithmetic_"):
            theory_map = {
                "arithmetic_bnd": "BND",
                "arithmetic_outc": "OUTC",
                "arithmetic_lin": "LIN",
                "arithmetic_poly": "POLY",
            }
            req = theory_map.get(tc.category)
            if req and supported_arithmetic_complexity_theories is not None and req not in supported_arithmetic_complexity_theories:
                skipped.append(SkippedResult(tc, req))
            else:
                runnable.append(tc)
        elif tc.primary_op_type and supported_ops is not None and tc.primary_op_type not in supported_ops:
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
