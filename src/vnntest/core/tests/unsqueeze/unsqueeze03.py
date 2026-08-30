"""Unsqueeze node test case: axis=1 (insert after existing dim), unsat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def unsqueeze_x(x: FLOAT[2]) -> FLOAT[2, 1]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [1]))
        return op.Unsqueeze(x, axes)

    return unsqueeze_x.to_model_proto()


def build_vnnlib_spec():
    # X[0], X[1] each 0 to 1. Y[0,0]=X[0], Y[1,0]=X[1] unchanged, just
    # reshaped. Y[0,0] >= 2.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2,1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= Y[0,0] 2.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"
