"""Flatten node test case: UNSAT direction, axis=1, order-sensitive. 
X[0,1,0] has a different range than the rest so we can catch a solver
doing column-major flattening instead of row-major. If it flattens
correctly, Y[0,1]=X[0,0,1] (maxes at 1.0). If it flattens wrong,
it'd grab X[0,1,0] instead (goes up to 11.0) and wrongly say SAT.
flatten04.py checks the same thing from a different position."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def flatten_x(x: FLOAT[1, 2, 2]) -> FLOAT[1, 4]:
        return op.Flatten(x, axis=1)

    return flatten_x.to_model_proto()


def build_vnnlib_spec():
    # X[0,1,0] is 10-11, the rest are 0-1. Row-major flatten: Y[0,1]=
    # X[0,0,1], maxes at 1.0, so Y[0,1] >= 10.0 is unreachable -> UNSAT.
    # A column-major solver would grab X[0,1,0] instead (10-11) and
    # wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,2,2])
    (declare-output Y float32 [1,4])
)

; Input Constraints
(assert (>= X[0,0,0] 0.0))
(assert (<= X[0,0,0] 1.0))
(assert (>= X[0,0,1] 0.0))
(assert (<= X[0,0,1] 1.0))
(assert (>= X[0,1,0] 10.0))
(assert (<= X[0,1,0] 11.0))
(assert (>= X[0,1,1] 0.0))
(assert (<= X[0,1,1] 1.0))

; Output Constraints
(assert (>= Y[0,1] 10.0))
(assert (<= Y[0,1] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model flattens X into a 1x4 matrix; under correct row-major flattening Y[0,1]=X[0,0,1]. With X[0,0,1] in [0.0, 1.0], Y[0,1] is at most 1.0, making the output condition Y[0,1] >= 10.0 impossible. The solver incorrectly claimed it is satisfiable -- likely because it flattened column-major instead, picking up X[0,1,0] (which is 10.0 to 11.0)."
