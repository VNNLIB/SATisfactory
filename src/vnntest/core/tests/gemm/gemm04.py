"""Gemm node test case: SAT direction, alpha/beta scaling. Paired with
gemm03.py. Tests that alpha actually scales A * B. If a solver ignored
alpha (treated it as 1.0 instead of 2.0), it would compute a lower max
and wrongly report UNSAT, missing this solution."""

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
            "c", TensorProto.FLOAT, [1], [1.0]
            ))
        return op.Gemm(x, b, c, alpha=2.0, beta=0.5)

    return gemm_layer.to_model_proto()


def build_vnnlib_spec():
    # X[0,0], X[0,1] each in [0,1]. Y[0,0] = 2.0*(X[0,0]+X[0,1]) + 0.5*1.0,
    # so Y[0,0] ranges 0.5 to 4.5. Y[0,0] >= 4.0 is reachable (witness
    # X[0,0]=1.0, X[0,1]=1.0 -> Y[0,0]=4.5) -> SAT.
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
(assert (>= Y[0,0] 4.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y[0,0] = 2.0 * ( X[0,0] + X[0,1] ) + 0.5 * 1.0 using alpha = 2.0, beta = 0.5. With X[0,0] = 1.0 and X[0,1] = 1.0, Y[0,0] = 4.5, satisfying Y[0,0] >= 4.0. The solver failed to find this solution likely because it ignored alpha and used 1.0 instead of 2.0, which would cap Y[0,0] at 2.5."