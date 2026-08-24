"""Unsqueeze node test case: sat direction. Paired with unsqueeze01.py so
the suite covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    """Builds the ONNX graph: single input X, Unsqueeze adds a dimension, output Y."""

    @script()
    def unsqueeze_x(x: FLOAT[1]) -> FLOAT[1, 1]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        return op.Unsqueeze(x, axes)

    return unsqueeze_x.to_model_proto()


def build_vnnlib_spec():
    """Builds the VNNLIB query: X in [0,1], asserts Y[0,0] >= 0.7 (reachable, expect SAT)."""
    # Witness X=1.0 -> Y[0,0]=1.0, satisfies Y[0,0] >= 0.7 -> expect SAT
    spec = """(vnnlib-version 2.0)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1,1])
)

(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

(assert (>= Y[0,0] 0.7))"""
    return spec.splitlines()
