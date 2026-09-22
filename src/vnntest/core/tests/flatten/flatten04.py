"""Flatten node test case: SAT direction, axis=1, order-sensitive. Paired
with flatten03.py same distinctive bounds, but targets Y[0,2] 
(= X[0,1,0] under correct row-major flattening) so the true answer is SAT.
A column-major implementation would wrongly compute Y[0,2]=X[0,0,1]
instead and say UNSAT."""

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
    # X[0,1,0] is 10-11, the rest are 0-1. Row-major flatten: Y[0,2]=
    # X[0,1,0], so Y[0,2] >= 10.0 is reachable (X[0,1,0]=10.0) -> SAT.
    # A column-major solver would grab X[0,0,1] instead (maxes at 1.0)
    # and wrongly say UNSAT.
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
(assert (>= Y[0,2] 10.0))
(assert (<= Y[0,2] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model flattens X into a 1x4 matrix; under correct row-major flattening Y[0,2]=X[0,1,0]. With X[0,1,0] in [10.0, 11.0], finding Y[0,2] >= 10.0 is satisfiable (at X[0,1,0]=10.0). The solver failed to find this solution -- likely because it flattened column-major instead, picking up X[0,0,1] (which maxes at 1.0)."