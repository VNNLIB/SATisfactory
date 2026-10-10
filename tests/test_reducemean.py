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


@pytest.fixture(params=["reducemean01", "reducemean02"])
def reducemean_case(request):
    return next(
        case for case in get_cases(generator="handwritten", node_type="reducemean")
        if case.id == request.param
    )


@pytest.fixture
def reducemean_artifacts(reducemean_case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return build_test_artifacts(reducemean_case)


def test_reducemean_cases_are_discovered():
    cases = get_cases(generator="handwritten", node_type="reducemean")
    assert "reducemean" in get_node_types()
    assert {case.id for case in cases} == {"reducemean01", "reducemean02"}
    assert len(cases) == 2
    assert all(case.node_type == "reducemean" for case in cases)
    assert all(case.suite == "full" for case in cases)


def test_reducemean_onnx_model_is_valid(reducemean_artifacts):
    onnx_path, _ = reducemean_artifacts
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == ["Constant", "ReduceMean"]


def test_reducemean_is_primary_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = generate_all_tests(node_type="reducemean")
    assert {entry["name"] for entry in manifest} == {"reducemean01", "reducemean02"}
    assert all(entry["primary_op_type"] == "ReduceMean" for entry in manifest)


def test_reducemean_axes_and_keepdims(reducemean_case, reducemean_artifacts):
    onnx_path, _ = reducemean_artifacts
    model = onnx.load(str(onnx_path))
    reduction = next(node for node in model.graph.node if node.op_type == "ReduceMean")
    axes_node = next(node for node in model.graph.node if reduction.input[1] in node.output)
    assert axes_node.op_type == "Constant"
    axes = next(attribute.t for attribute in axes_node.attribute if attribute.name == "value")
    assert axes.data_type == onnx.TensorProto.INT64
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(axes), [1])
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in reduction.attribute
    }
    expected_keepdims = {"reducemean01": 1, "reducemean02": 0}
    assert attributes["keepdims"] == expected_keepdims[reducemean_case.id]


def test_reducemean_values_and_output_shape(reducemean_case, reducemean_artifacts):
    onnx_path, _ = reducemean_artifacts
    model = onnx.load(str(onnx_path))
    expected_shape = {"reducemean01": (2, 1), "reducemean02": (2,)}[reducemean_case.id]
    dimensions = model.graph.output[0].type.tensor_type.shape.dim
    assert tuple(dimension.dim_value for dimension in dimensions) == expected_shape
    inputs = np.array([[2.0, 6.0], [10.0, 14.0]], dtype=np.float32)
    outputs = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: inputs})
    assert len(outputs) == 1
    assert outputs[0].shape == expected_shape
    expected = np.array([4.0, 12.0], dtype=np.float32).reshape(expected_shape)
    np.testing.assert_array_equal(outputs[0], expected)


def test_reducemean_vnnlib_spec_is_parseable(reducemean_artifacts):
    _, vnnlib_path = reducemean_artifacts
    assert vnnlib.parse_query_file(str(vnnlib_path)) is not None


def test_reducemean_expected_result(reducemean_case):
    expected = {"reducemean01": "SAT", "reducemean02": "UNSAT"}
    assert reducemean_case.expected_output() == expected[reducemean_case.id]
