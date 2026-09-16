"""Neg node test case: sat direction. Paired with neg01.py so the suite
covers both polarities for this node type."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def neg_x(x: FLOAT[1]) -> FLOAT[1]:
        return op.Neg(x)

    return neg_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = -X, so Y is -1.0 to 0.0. Y <= -0.5 is
    # reachable (witness X=1.0 -> Y=-1.0) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] -100.0))
(assert (<= Y[0] -0.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = -X. With X in [0.0, 1.0], finding Y[0] <= -0.5 is satisfiable (at X=1.0, Y=-1.0). The solver failed to find this solution."
