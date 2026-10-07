"""Equal node test case: unsat direction. Mixed vector: first element
equals 0.0, second does not."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def equal_zero(x: FLOAT[2]) -> FLOAT[2]:
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
        )
        eq = op.Equal(x, zero)
        return op.Cast(eq, to=TensorProto.FLOAT)

    return equal_zero.to_model_proto()


def build_vnnlib_spec():
    # X[0] is 0.0 -> Y[0] = 1. X[1] is 1 to 2 -> Y[1] = 0.
    # Y[1] >= 0.5 is unreachable. A solver that treated every element
    # as equal would wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 0.0))
(assert (>= X[1] 1.0))
(assert (<= X[1] 2.0))

; Output Constraints
(assert (>= Y[1] 0.5))
(assert (<= Y[1] 1.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = 1.0 where X == 0.0 else 0.0. X[1] is in [1.0, 2.0] so Y[1] is 0.0, making Y[1] >= 0.5 impossible. The solver incorrectly claimed it is satisfiable, likely because it treated every element as equal."
