"""Arithmetic Complexity theory test: POLY level, input-combination axis,
sat direction. Paired with arithmetic_poly01.py -- same network and
input ranges, a tight threshold right at the true maximum instead of
beyond it, so a looser-than-true bound on the product would fail to
find this solution."""

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
    # Same network and ranges as arithmetic_poly01.py. X[0]*X[1] reaches
    # exactly 4.0 at X[0]=2.0, X[1]=2.0 -- the only point satisfying
    # >= 4.0 -> SAT.
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
(assert (>= (* X[0] X[1]) 4.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: With X[0], X[1] each in [0.0, 2.0], X[0]*X[1] reaches exactly 4.0 at X[0]=2.0, X[1]=2.0 -- the only point satisfying >= 4.0. The solver failed to find this tight witness."
