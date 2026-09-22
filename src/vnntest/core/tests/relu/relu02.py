"""Relu node test case: unsat direction. Paired with relu01.py to cover
both polarities for the Relu node."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def relu_layer(x: FLOAT[1]) -> FLOAT[1]:
        return op.Relu(x)

    return relu_layer.to_model_proto()


def build_vnnlib_spec():
    # X[0] is in [1.0, 1000.0]. Y = Relu(X), so Y[0] will be in [1.0, 1000.0].
    # Output constraint requires Y[0] <= 0.0, which is impossible.
    # Therefore, UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (<= X[0] 1000.0))
(assert (>= X[0] 1.0))

; Output Constraints
(assert (<= Y[0] 0.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Relu(X). With X[0] in [1.0, 1000.0], Y is in [1.0, 1000.0], making the condition Y[0] <= 0.0 impossible. The solver incorrectly claimed it is satisfiable."