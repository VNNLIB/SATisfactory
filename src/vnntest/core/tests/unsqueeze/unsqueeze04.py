"""Unsqueeze node test case: axis=1 (insert after existing dim), sat direction.
Paired with unsqueeze03.py."""

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
    # X[0], X[1] each 0 to 1. Y[0,0]=X[0], Y[1,0]=X[1] unchanged. Y[0,0]
    # >= 0.7 is reachable (witness X[0]=1.0) -> SAT.
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
(assert (>= Y[0,0] 0.7))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model reshapes [X[0], X[1]] into Y[0,0]=X[0], Y[1,0]=X[1] unchanged. With X[0] in [0.0, 1.0], finding Y[0,0] >= 0.7 is satisfiable (at X[0]=1.0, Y[0,0]=1.0). The solver failed to find this solution."
