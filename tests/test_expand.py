import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest
import vnnlib

from vnntest.core.generator import build_test_artifacts, generate_all_tests
from vnntest.core.registry import get_cases, get_node_types, reset_registry


def setup_function():
    reset_registry()


def teardown_function():
    reset_registry()


@pytest.fixture(params=["expand01", "expand02"])
def expand_case(request):
    return next(
        case for case in get_cases(generator="handwritten", node_type="expand")
        if case.id == request.param
    )


@pytest.fixture
def expand_artifacts(expand_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return build_test_artifacts(expand_case)


def test_expand_cases_are_discovered():
    cases = get_cases(generator="handwritten", node_type="expand")
    assert "expand" in get_node_types()
    assert {case.id for case in cases} == {"expand01", "expand02"}
    assert len(cases) == 2
    assert all(case.node_type == "expand" for case in cases)
    assert all(case.suite == "full" for case in cases)


def test_expand_onnx_model_is_valid(expand_artifacts):
    onnx_path, _ = expand_artifacts
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Constant", "Expand"]
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in dimensions] == [2, 3]


def test_expand_is_primary_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = generate_all_tests(node_type="expand")
    assert {entry["name"] for entry in manifest} == {"expand01", "expand02"}
    assert all(entry["primary_op_type"] == "Expand" for entry in manifest)


def test_expand_repeats_values_across_columns(expand_artifacts):
    onnx_path, _ = expand_artifacts
    model = onnx.load(str(onnx_path))
    evaluator = ReferenceEvaluator(model)
    # Check interval endpoints and an interior value, including unequal magnitudes.
    for a, b in [(1.0, -1.0), (2.0, -2.0), (1.5, -1.5), (1.0, -2.0), (2.0, -1.0)]:
        inputs = np.array([[a], [b]], dtype=np.float32)
        outputs = evaluator.run(None, {model.graph.input[0].name: inputs})
        assert len(outputs) == 1
        assert outputs[0].shape == (2, 3)
        expected = np.array([[a, a, a], [b, b, b]], dtype=np.float32)
        np.testing.assert_array_equal(outputs[0], expected)


def test_expand_vnnlib_spec_is_parseable(expand_artifacts):
    _, vnnlib_path = expand_artifacts
    assert vnnlib.parse_query_file(str(vnnlib_path)) is not None


def test_expand_expected_result(expand_case):
    expected = {"expand01": "SAT", "expand02": "UNSAT"}
    assert expand_case.expected_output() == expected[expand_case.id]
