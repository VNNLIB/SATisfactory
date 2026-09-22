"""Gemm node test case: UNSAT Direction, transB attribute. B is declared
with transB = 1, so the correct computation transposes B before
multiplying. Row 0 of the raw B tensor ([1.0, 2.0]) is deliberately
much smaller than column 0 ([1.0, 10.0]), so a solver that forgot to
transpose B would compute a much larger value and wrongly report SAT."""

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
    # B is stored as [[1.0, 2.0], [10.0, 20.0]] with transB=1, so the true
    # computation uses B' = transpose(B). Y[0,0] = X[0,0]*1.0 + X[0,1]*2.0
    # (row 0 of raw B), so Y[0,0] ranges 0 to 3. Y[0,0] >= 5.0 is
    # unreachable -> UNSAT. (A solver that forgot to transpose B would
    # instead use column 0 [1.0, 10.0], giving a max of 11.0, and wrongly
    # say SAT.)
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
(assert (>= Y[0,0] 5.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: With transB = 1, Y[0,0] = X[0,0] * 1.0 + X[0,1] * 2.0 (row 0 of the raw B tensor [[1.0, 2.0], [10.0, 20.0]]). With X[0,0] and X[0,1] each in [0,1], Y[0,0] tops out at 3.0, making Y[0,0] >= 5.0 impossible. The solver incorrectly claimed it is satisfiable likely because it skipped the transpose and used column 0 ([1.0, 10.0]) instead, which can reach 11.0."