"""Unsqueeze node test case: unsat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    """Builds the ONNX graph: single input X, Unsqueeze adds a dimension, output Y."""
    # Opset 18 takes `axes` as a second input tensor, not an attribute
    # (earlier opsets did it differently) -- baked in as a constant, same
    # approach as the Add/Sub tests. Unsqueeze is value-preserving: it only
    # adds a dimension, it never changes the number itself, so the output
    # shape grows from [1] to [1,1] but the value stays identical.
    @script()
    def unsqueeze_x(x: FLOAT[1]) -> FLOAT[1, 1]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        return op.Unsqueeze(x, axes)

    return unsqueeze_x.to_model_proto()


def build_vnnlib_spec():
    """Builds the VNNLIB query: X in [0,1], asserts Y[0,0] >= 2.0 (unreachable, expect UNSAT)."""
    # Y is X unchanged, just wrapped in an extra dimension -> Y[0,0] in [0,1]
    # Max reachable Y[0,0] is 1.0, so Y[0,0] >= 2.0 is unreachable -> expect UNSAT
    spec = """(vnnlib-version 2.0)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1,1])
)

(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

(assert (>= Y[0,0] 2.0))"""
    return spec.splitlines()
