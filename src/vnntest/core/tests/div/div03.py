"""Div node test case: unsat direction, negative divisor. Checks basic
sign handling -- a positive numerator divided by an always-negative
divisor must be negative, so asking for a non-negative output should
be unreachable. Paired with div04.py, which stresses the same
negative-divisor case for precision rather than just sign."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def div_x(x: FLOAT[2]) -> FLOAT[1]:
        return op.Div(x[0:1], x[1:2])

    return div_x.to_model_proto()


def build_vnnlib_spec():
    # X[0] (numerator) is 2 to 8, always positive. X[1] (divisor) is
    # -4 to -1, always negative. Y = X[0]/X[1] is therefore always
    # negative (true range [-8.0, -0.5]). Y >= 0.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 2.0))
(assert (<= X[0] 8.0))
(assert (>= X[1] -4.0))
(assert (<= X[1] -1.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 1000.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X[0] / X[1]. With X[0] always positive [2.0, 8.0] and X[1] always negative [-4.0, -1.0], Y must always be negative, making the output condition Y[0] >= 0.0 impossible. The solver incorrectly claimed it is satisfiable -- likely a sign-handling error when dividing by a negative range."
