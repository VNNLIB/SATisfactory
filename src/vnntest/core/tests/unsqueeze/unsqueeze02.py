"""Unsqueeze node test case: sat direction. Paired with unsqueeze01.py so
the suite covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def unsqueeze_x(x: FLOAT[1]) -> FLOAT[1, 1]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        return op.Unsqueeze(x, axes)

    return unsqueeze_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y[0,0] is X unchanged, so Y[0,0] is 0 to 1. Y[0,0] >=
    # 0.7 is reachable (witness X=1.0 -> Y[0,0]=1.0) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1,1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0,0] 0.7))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model reshapes X into Y[0,0] unchanged. With X[0] in [0.0, 1.0], finding Y[0,0] >= 0.7 is satisfiable (at X=1.0, Y[0,0]=1.0). The solver failed to find this solution."
