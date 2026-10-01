"""Arithmetic Complexity theory test: LIN level, output-combination axis,
sat direction. Paired with arithmetic_lin03.py -- same network and input
ranges. The threshold here (0.5) is the standard's own worked example
for this exact assertion shape, (+ Y[0] Y[1]) >= 0.5."""

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
    # Same network and ranges as arithmetic_lin03.py. Y[0]+Y[1] has true
    # range [0, 2]. Asserting it's >= 0.5 is reachable (witness X[0]=1.0,
    # X[1]=0.0 -> Y[0]+Y[1]=1.0) -> SAT.
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
(assert (>= (+ Y[0] Y[1]) 0.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: With X[0], X[1] each in [0.0, 1.0] and Y=X, finding Y[0]+Y[1] >= 0.5 is satisfiable (at X[0]=1.0, X[1]=0.0, Y[0]+Y[1]=1.0). The solver failed to find this solution."
