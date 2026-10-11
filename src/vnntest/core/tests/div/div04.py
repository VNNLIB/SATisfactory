"""Div node test case: sat direction, negative divisor, tight window
against the true maximum. Paired with div03.py. This is the sharpest
test in the Div suite: it needs both correct sign handling (dividing by
a negative range flips which corner of the input box produces the
largest output) and a tight bound right at the true boundary."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def div_x(x: FLOAT[2]) -> FLOAT[1]:
        return op.Div(x[0:1], x[1:2])

    return div_x.to_model_proto()


def build_vnnlib_spec():
    # Same network as div03. X[0] in [2.0, 8.0], X[1] in [-4.0, -1.0].
    # Y = X[0]/X[1], true range [-8.0, -0.5], maximised (least negative)
    # at X[0]=2.0, X[1]=-4.0 -> Y=-0.5. Y in [-1.0, -0.5] is reachable
    # at exactly that witness -> SAT.
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
(assert (>= Y[0] -1.0))
(assert (<= Y[0] -0.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X[0] / X[1]. The true maximum of Y is -0.5, reached at X[0]=2.0, X[1]=-4.0, which satisfies Y[0] in [-1.0, -0.5]. The solver failed to find this solution -- likely a combination of the sign flip on a negative divisor and a looser-than-true bound near that boundary."
