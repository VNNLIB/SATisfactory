"""Arithmetic Complexity theory test: LIN level, input-combination axis,
sat direction. Paired with arithmetic_lin01.py -- same network and input
ranges, a tight threshold right at the true maximum instead of beyond
it, so a solver with a looser-than-true bound on the weighted sum would
fail to find this solution."""

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
    # Same network and ranges as arithmetic_lin01.py. The true maximum of
    # 0.5*X[0] + 0.75*X[1] is exactly 2.5 (at X[0]=2.0, X[1]=2.0).
    # Asserting it's >= 2.5 is reachable only at that exact corner -> SAT.
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
(assert (>= (+ (* 0.5 X[0]) (* 0.75 X[1])) 2.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: With X[0], X[1] each in [0.0, 2.0], 0.5*X[0] + 0.75*X[1] reaches exactly 2.5 at X[0]=2.0, X[1]=2.0 -- the only point satisfying >= 2.5. The solver failed to find this tight witness."
