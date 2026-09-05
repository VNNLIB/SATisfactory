"""Sub node test case: unsat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def sub_bias(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [0.5]))
        return op.Sub(x, c)

    return sub_bias.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = X - 0.5, so Y is -0.5 to 0.5. Y >= 1.0 is
    # unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 1.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X - 0.5. With X in [0.0, 1.0], Y is at most 0.5, making the output condition Y[0] >= 1.0 impossible. The solver incorrectly claimed it is satisfiable."
