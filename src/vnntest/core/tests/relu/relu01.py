"""Expected SAT: Model computes Y = Relu(X). With X[0] in [-1000.0, 0.0], Y is 0.0, satisfying Y[0] <= 0.0. The solver failed to find this solution."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def relu_layer(x: FLOAT[1]) -> FLOAT[1]:
        return op.Relu(x)

    return relu_layer.to_model_proto()


def build_vnnlib_spec():
    # X[0] is in [-1000.0, 0.0]. Y = Relu(X), so Y[0] will be 0.0.
    # Output constraint checks if Y[0] <= 0.0 or Y[0] >= 1.0. 
    # Since Y[0] == 0.0, it is SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (<= X[0] 0.0))
(assert (>= X[0] -1000.0))

; Output Constraints
(assert (or (<= Y[0] 0.0) (>= Y[0] 1.0)))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Relu(X). With X[0] in [-1000.0, 0.0], Y is 0.0, satisfying Y[0] <= 0.0. The solver failed to find this solution."