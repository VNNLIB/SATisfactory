"""Sigmoid node test case: sat direction."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def sigmoid_test(x: FLOAT[1]) -> FLOAT[1]:
        return op.Sigmoid(x)

    return sigmoid_test.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = Sigmoid(X), so Y is 0.5 to ~0.731.
    # Y >= 0.7 is reachable -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 0.7))
(assert (<= Y[0] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Sigmoid(X). With X in [0.0, 1.0], Y can be up to ~0.731, making the output condition Y[0] >= 0.7 possible. The solver incorrectly claimed it is unsatisfiable."
