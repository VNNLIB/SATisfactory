"""Gemm node test case: SAT direction, transB attribute. Paired with
gemm05.py same B tensor, but targets Y[0,1] (row 1 of raw B, [10.0,
20.0]) so the true answer is SAT. A solver that forgot to transpose B
would use column 1 ([2.0, 20.0]) instead, capping the value lower and
wrongly reporting UNSAT."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def gemm_layer(x: FLOAT[1, 2]) -> FLOAT[1, 2]:
        b = op.Constant(value=onnx_helper.make_tensor(
            "b", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 10.0, 20.0]
            ))
        c = op.Constant(value=onnx_helper.make_tensor(
            "c", TensorProto.FLOAT, [2], [0.0, 0.0]
            ))
        return op.Gemm(x, b, c, transB=1)

    return gemm_layer.to_model_proto()


def build_vnnlib_spec():
    # Y[0,1] = X[0,0]*10.0 + X[0,1]*20.0 (row 1 of raw B), so Y[0,1]
    # ranges 0 to 30. Y[0,1] >= 25.0 is reachable (witness X[0,0]=1.0,
    # X[0,1]=1.0 -> Y[0,1]=30.0) -> SAT. 
    # (A solver that forgot to transpose B would instead use column 
    # 1 [2.0, 20.0], giving a max of 22.0, and wrongly say UNSAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,2])
    (declare-output Y float32 [1,2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))

; Output Constraints
(assert (>= Y[0,1] 25.0))
(assert (<= Y[0,1] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: With transB = 1, Y[0,1] = X[0,0] * 10.0 + X[0,1] * 20.0 (row 1 of the raw B tensor [[1.0, 2.0], [10.0, 20.0]]). With X[0,0] = 1.0 and X[0,1] = 1.0, Y[0,1] = 30.0, satisfying Y[0,1] >= 25.0. The solver failed to find this solution likely because it skipped the transpose and used column 1 ([2.0, 20.0]) instead, which only reaches 22.0."