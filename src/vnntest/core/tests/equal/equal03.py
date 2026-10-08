"""Equal node test case: unsat direction. X is fixed at 0.0, so Equal
is always true and Y is 1.0."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def equal_zero(x: FLOAT[1]) -> FLOAT[1]:
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
        )
        eq = op.Equal(x, zero)
        return op.Cast(eq, to=TensorProto.FLOAT)

    return equal_zero.to_model_proto()


def build_vnnlib_spec():
    # X is exactly 0.0, so Equal is true and Y = 1.0. Y[0] <= 0.5 is
    # unreachable. A solver that treated Equal as always false would
    # wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 0.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 0.5))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = 1.0 if X == 0.0 else 0.0. With X fixed at 0.0, Equal is always true, so Y is 1.0 and Y[0] <= 0.5 is impossible. The solver incorrectly claimed it is satisfiable, likely because it treated the values as not equal."
