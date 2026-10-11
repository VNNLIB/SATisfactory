"""Arithmetic Complexity theory test: LIN level, output-combination axis,
unsat direction. Paired with the input-axis pair (arithmetic_lin01/02)
to cover the other half of what LIN unlocks over OUTC: OUTC can only
compare two outputs directly to each other (Y[0] >= Y[1]), it cannot sum
or weight them. This pair tests a genuine linear combination of outputs,
(+ Y[0] Y[1]), using inputs that stay simple bounds so only the output
side is under test."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def model(x: FLOAT[2]) -> FLOAT[2]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [2], [0.0, 0.0]))
        return op.Add(x, c)

    return model.to_model_proto()


def build_vnnlib_spec():
    # X[0], X[1] each 0 to 1. Y = X, so Y[0]+Y[1] has true range [0, 2].
    # Asserting it's >= 3.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= (+ Y[0] Y[1]) 3.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: With X[0], X[1] each in [0.0, 1.0] and Y=X, the true maximum of Y[0]+Y[1] is 2.0, making the assertion >= 3.0 impossible."
