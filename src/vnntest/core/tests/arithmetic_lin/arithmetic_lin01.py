"""Arithmetic Complexity theory test: LIN level, input-combination axis,
unsat direction. LIN allows genuine linear combinations -- coefficients
and sums -- over declared variables, not just a bare comparison like
OUTC. This pair tests the combination applied to inputs, using the
standard's own example coefficients (0.5, 0.75). Checks that a solver
claiming LIN support correctly evaluates a weighted sum of two inputs,
not just individual input bounds."""

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
    # X[0], X[1] each 0 to 2. 0.5*X[0] + 0.75*X[1] has true range [0, 2.5]
    # (max at X[0]=2, X[1]=2). Asserting it's >= 3.0 is unreachable -> UNSAT.
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

; Output Constraints
(assert (>= (+ (* 0.5 X[0]) (* 0.75 X[1])) 3.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: With X[0], X[1] each in [0.0, 2.0], the true maximum of 0.5*X[0] + 0.75*X[1] is 2.5 (at X[0]=2.0, X[1]=2.0), making the assertion >= 3.0 impossible."
