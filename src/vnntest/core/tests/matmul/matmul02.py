"""MatMul node test case: sat direction. Paired with matmul01.py so the
suite covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def matmul_w(x: FLOAT[1, 2]) -> FLOAT[1, 1]:
        w = op.Constant(value=onnx_helper.make_tensor("w", TensorProto.FLOAT, [2, 1], [0.5, 0.5]))
        return op.MatMul(x, w)

    return matmul_w.to_model_proto()


def build_vnnlib_spec():
    # X[0,0], X[0,1] each 0 to 1. Y[0,0] = 0.5*X[0,0] + 0.5*X[0,1], so
    # Y[0,0] is 0 to 1.0. Y[0,0] >= 0.7 is reachable (witness X[0,0]=1.0,
    # X[0,1]=0.4 -> Y[0,0]=0.7) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,2])
    (declare-output Y float32 [1,1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))

; Output Constraints
(assert (>= Y[0,0] 0.7))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y[0,0] = 0.5*X[0,0] + 0.5*X[0,1]. With both inputs in [0.0, 1.0], finding Y[0,0] >= 0.7 is satisfiable (at X[0,0]=1.0, X[0,1]=0.4, Y[0,0]=0.7). The solver failed to find this solution."
