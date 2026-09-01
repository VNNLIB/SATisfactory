"""Reshape node test case: sat direction. Paired with reshape01.py so the
suite covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reshape_x(x: FLOAT[4]) -> FLOAT[2, 2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2]))
        return op.Reshape(x, shape)

    return reshape_x.to_model_proto()


def build_vnnlib_spec():
    # X[0..3] each 0 to 1. Row-major reshape: Y[0,0]=X[0]. Y[0,0] >= 0.7
    # is reachable (witness X[0]=1.0 -> Y[0,0]=1.0) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 0.0))
(assert (<= X[2] 1.0))
(assert (>= X[3] 0.0))
(assert (<= X[3] 1.0))

; Output Constraints
(assert (>= Y[0,0] 0.7))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"
