"""Div node test case: sat direction, positive divisor, tight window
against the true maximum. Paired with div01.py. Unlike div01 (which is
trivially outside the true range regardless of precision), this one
requires the solver to actually find the tightest correct bound -- a
loose over-approximation here reports UNSAT when the true answer is
SAT, the same way our earlier Pow and BatchNormalization tests caught
vibecheck being confidently wrong rather than honestly unsure."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def div_x(x: FLOAT[2]) -> FLOAT[1]:
        return op.Div(x[0:1], x[1:2])

    return div_x.to_model_proto()


def build_vnnlib_spec():
    # Same network as div01. X[0] in [4.0, 8.0], X[1] in [2.0, 4.0].
    # Y = X[0]/X[1], true range [1.0, 4.0], maximised at X[0]=8, X[1]=2.
    # Y in [3.5, 4.0] is reachable at exactly that witness -> SAT.
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
(assert (>= Y[0] 3.5))
(assert (<= Y[0] 4.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X[0] / X[1]. The true maximum of Y is 4.0, reached at X[0]=8.0, X[1]=2.0, which satisfies Y[0] in [3.5, 4.0]. The solver failed to find this solution -- likely because its bound on the division was looser than the true range."
