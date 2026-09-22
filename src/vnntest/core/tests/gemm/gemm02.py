"""Gemm node test case: sat direction, default attributes (alpha=1,
beta=1, no transpose). Paired with gemm01.py so the suite covers both
polarities for this node type. See gemm03/gemm04 for alpha/beta
scaling, and gemm05/gemm06 for transB."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def gemm_layer(x: FLOAT[1, 2]) -> FLOAT[1, 1]:
        b = op.Constant(value=onnx_helper.make_tensor(
            "b", TensorProto.FLOAT, [2, 1], [1.0, 1.0]
        ))
        c = op.Constant(value=onnx_helper.make_tensor(
            "c", TensorProto.FLOAT, [1], [0.0]
        ))
        return op.Gemm(x, b, c)

    return gemm_layer.to_model_proto()

def build_vnnlib_spec():
    # X[0,0] and X[0,1] each in [0,1]. Y[0,0] = X[0,0] + X[0,1],
    # so Y[0,0] ranges 0 to 2. Y[0,0] >= 1.5 is reachable
    # (witness X[0,0]=1.0, X[0,1]=1.0 -> Y[0,0]=2.0) -> SAT.
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
(assert (>= Y[0,0] 1.5))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Model computes Y[0,0] = X[0,0] + X[0,1]. With X[0,0]=1.0 and X[0,1]=1.0, Y[0,0]=2.0, satisfying the output condition Y[0,0] >= 1.5. The solver failed to find this solution."