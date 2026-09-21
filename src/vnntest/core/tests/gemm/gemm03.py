"""Gemm node test case: UNSAT direction, alpha/beta scaling. Tests that
beta actually scales C if a solver ignored beta (treated it as 1.0
instead of 0.5), it would compute a higher max and wrongly report SAT."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def gemm_layer(x: FLOAT[1, 2]) -> FLOAT[1, 1]:
        b = op.Constant(value=onnx_helper.make_tensor( "b", TensorProto.FLOAT, [2, 1], [1.0, 1.0]))
        c = op.Constant(value=onnx_helper.make_tensor( "c", TensorProto.FLOAT, [1], [1.0] ))
        return op.Gemm(x, b, c, alpha=2.0, beta=0.5)

    return gemm_layer.to_model_proto()


def build_vnnlib_spec():
    # X[0,0], X[0,1] each in [0,1]. Y[0,0] = 2.0*(X[0,0]+X[0,1]) + 0.5*1.0,
    # so Y[0,0] ranges 0.5 to 4.5. Y[0,0] >= 4.8 is unreachable -> UNSAT.
    # (A solver that ignored beta, treating it as 1.0 instead of 0.5, would
    # compute Y[0,0] up to 5.0 and wrongly say SAT.)
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
(assert (>= Y[0,0] 4.8))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y[0,0] = 2.0 * ( X[0,0]+X[0,1] ) + 0.5 * 1.0 using alpha = 2.0, beta = 0.5. With X[0,0] and X[0,1] each in [0,1], Y[0,0] tops out at 4.5, making Y[0,0] >= 4.8 impossible. The solver incorrectly claimed it is satisfiable likely because it ignored beta and used 1.0 instead of 0.5."