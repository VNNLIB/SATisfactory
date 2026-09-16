"""Sigmoid node test case: bounds unsat direction."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def sigmoid_test(x: FLOAT[1]) -> FLOAT[1]:
        return op.Sigmoid(x)

    return sigmoid_test.to_model_proto()


def build_vnnlib_spec():
    # X is -1000 to 1000. Y = Sigmoid(X), so Y is strictly in (0, 1).
    # Y <= -0.01 or Y >= 1.01 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] -1000.0))
(assert (<= X[0] 1000.0))

; Output Constraints
(assert (or (<= Y[0] -0.01) (>= Y[0] 1.01)))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Sigmoid(X). The output of Sigmoid is always strictly between 0 and 1, regardless of the input. Therefore, it is impossible for the output Y[0] to be <= -0.01 or >= 1.01. The solver incorrectly claimed it is satisfiable."
