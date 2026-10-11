"""Multiple input nodes test case: UNSAT direction.
Two separate ONNX inputs X1 and X2, one output Y = X1 - X2.
Sub is order sensitive, so a solver that mixes up which declared input
maps to which ONNX input will get the wrong answer."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def multi_input(x1: FLOAT[1], x2: FLOAT[1]) -> FLOAT[1]:
        return op.Sub(x1, x2)

    return multi_input.to_model_proto()


def build_vnnlib_spec():
    # X1 in [0.0, 1.0], X2 in [10.0, 11.0]. Y = X1 - X2, so Y is in [-11.0, -9.0].
    # Y >= -8.0 is unreachable -> UNSAT.
    # If the inputs were swapped, Y would be in [9.0, 11.0] and the solver would say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X1 float32 [1])
    (declare-input X2 float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X1[0] 0.0))
(assert (<= X1[0] 1.0))
(assert (>= X2[0] 10.0))
(assert (<= X2[0] 11.0))

; Output Constraints
(assert (>= Y[0] -8.0))
"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model has two inputs and computes Y = X1 - X2. With X1 in [0.0, 1.0] and X2 in [10.0, 11.0], Y is at most -9.0, so Y >= -8.0 is impossible. A SAT answer suggests the solver mapped the declared inputs to the wrong ONNX inputs."