import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import vnnlib

from vnntest.core.generator import build_test_artifacts
from vnntest.core.registry import get_cases, get_node_types, reset_registry


def setup_function():
    reset_registry()


def teardown_function():
    reset_registry()


@pytest.fixture(params=["neg01", "neg02"])
def neg_case(request):
    return next(
        case for case in get_cases(generator="handwritten", node_type="neg")
        if case.id == request.param
    )


@pytest.fixture
def neg_artifacts(neg_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return build_test_artifacts(neg_case)


def test_neg_cases_are_discovered():
    cases = get_cases(generator="handwritten", node_type="neg")
    assert "neg" in get_node_types()
    assert {case.id for case in cases} == {"neg01", "neg02"}
    assert len(cases) == 2
    assert all(case.node_type == "neg" for case in cases)
    assert all(case.suite == "full" for case in cases)


def test_neg_onnx_model_is_valid(neg_artifacts):
    onnx_path, _ = neg_artifacts
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Neg"]


def test_neg_flips_input_signs(neg_artifacts):
    onnx_path, _ = neg_artifacts
    model = onnx.load(str(onnx_path))
    evaluator = ReferenceEvaluator(model)
    # Exercise both ends of each input interval and an interior value.
    for values in ([1.0, -1.0], [2.0, -2.0], [1.5, -1.5]):
        inputs = np.array(values, dtype=np.float32)
        outputs = evaluator.run(None, {model.graph.input[0].name: inputs})
        assert len(outputs) == 1
        np.testing.assert_array_equal(outputs[0], -inputs)


def test_neg_vnnlib_spec_is_parseable(neg_artifacts):
    _, vnnlib_path = neg_artifacts
    assert vnnlib.parse_query_file(str(vnnlib_path)) is not None


def test_neg_expected_result(neg_case):
    expected = {"neg01": "SAT", "neg02": "UNSAT"}
    assert neg_case.expected_output() == expected[neg_case.id]
