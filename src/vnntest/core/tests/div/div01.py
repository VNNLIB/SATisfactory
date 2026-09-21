"""Div node test case: unsat direction, positive divisor. Sanity check
that basic division is computed at all -- paired with div02.py to cover
both polarities before div03/div04 stress the negative-divisor case."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def div_x(x: FLOAT[2]) -> FLOAT[1]:
        return op.Div(x[0:1], x[1:2])

    return div_x.to_model_proto()


def build_vnnlib_spec():
    # X[0] (numerator) is 4 to 8. X[1] (divisor) is 2 to 4, always
    # positive. Y = X[0]/X[1], so Y is 1.0 to 4.0. Y >= 5.0 is
    # unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 4.0))
(assert (<= X[0] 8.0))
(assert (>= X[1] 2.0))
(assert (<= X[1] 4.0))

; Output Constraints
(assert (>= Y[0] 5.0))
(assert (<= Y[0] 1000.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X[0] / X[1]. With X[0] in [4.0, 8.0] and X[1] in [2.0, 4.0] (always positive), Y is at most 4.0, making the output condition Y[0] >= 5.0 impossible. The solver incorrectly claimed it is satisfiable."
