"""Transpose node test case: UNSAT direction"""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def transpose_x(x: FLOAT[1, 2]) -> FLOAT[2, 1]:
        return op.Transpose(x, perm=[1, 0])

    return transpose_x.to_model_proto()


def build_vnnlib_spec():
    # X[0][0], X[0][1] each 0 to 1. Transpose only permutes dims, so
    # Y[0,0] = X[0,0], Y[1,0] = X[0,1] unchanged. Y[0,0] >= 2.0 is
    # unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,2])
    (declare-output Y float32 [2,1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))

; Output Constraints
(assert (>= Y[0,0] 2.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Transpose(perm=[1,0]) only permutes dimensions, so Y[0,0] = X[0,0]. With X[0,0] in [0.0, 1.0], the output condition Y[0,0] >= 2.0 is impossible. The solver incorrectly claimed it is satisfiable."