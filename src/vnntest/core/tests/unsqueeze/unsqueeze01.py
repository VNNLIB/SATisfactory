"""Unsqueeze node test case: unsat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # Opset 18 takes `axes` as an input tensor, not an attribute.
    # Unsqueeze only adds a dimension -- values are unchanged.
    @script()
    def unsqueeze_x(x: FLOAT[1]) -> FLOAT[1, 1]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        return op.Unsqueeze(x, axes)

    return unsqueeze_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y[0,0] is X unchanged, so Y[0,0] is 0 to 1. Y[0,0] >=
    # 2.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1,1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0,0] 2.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model reshapes X into Y[0,0] unchanged. With X[0] in [0.0, 1.0], Y[0,0] is at most 1.0, making the output condition Y[0,0] >= 2.0 impossible. The solver incorrectly claimed it is satisfiable."
