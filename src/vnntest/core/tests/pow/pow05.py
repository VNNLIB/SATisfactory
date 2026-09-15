"""Pow node test case: negative fractional indices."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def pow_model(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [-0.5]))
        return op.Pow(x, c)

    return pow_model.to_model_proto()


def build_vnnlib_spec():
    # X[0] is 1.0 to 4.0.
    # Y[0] = X[0]^(-0.5) = 1 / sqrt(X[0]), which is 0.5 to 1.0.
    # We want to check if Y[0] <= 0.6 is reachable. -> SAT
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 1.0))
(assert (<= X[0] 4.0))

; Output Constraints
(assert (<= Y[0] 0.6))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y[0] = X[0]^(-0.5). With X[0] in [1.0, 4.0], Y[0] can be 0.5 (when X[0] = 4.0). The output condition Y[0] <= 0.6 is reachable. The solver incorrectly claimed it is unsatisfiable."
