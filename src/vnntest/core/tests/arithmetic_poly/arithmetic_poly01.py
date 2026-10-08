"""Arithmetic Complexity theory test: POLY level, input-combination axis,
unsat direction. POLY allows polynomial expressions -- multiplying two
variables together, (* X[0] X[1]) -- the level above LIN, which only
allows weighted sums. Checks that a solver claiming POLY support
correctly evaluates a product of two inputs, not just a linear
combination of them."""

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
    # X[0], X[1] each 0 to 2. X[0]*X[1] has true range [0, 4] (max at
    # X[0]=2, X[1]=2). Asserting it's >= 4.5 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 2.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 2.0))

; Input Constraints (combined)
(assert (>= (* X[0] X[1]) 4.5))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: With X[0], X[1] each in [0.0, 2.0], the true maximum of X[0]*X[1] is 4.0 (at X[0]=2.0, X[1]=2.0), making the assertion >= 4.5 impossible."
