"""Reshape node test case: unsat direction, order-sensitive. X[1] and
X[2] are given very different ranges so that Y[0,1] only reaches the
asserted threshold if the solver applies ONNX's row-major flattening
(Y[0,1]=X[1]) rather than a column-major one (which would wrongly give
Y[0,1]=X[2]). Paired with reshape04.py, which targets the opposite
corner (Y[1,0]) to catch the same mistake from the other direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reshape_x(x: FLOAT[4]) -> FLOAT[2, 2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2]))
        return op.Reshape(x, shape)

    return reshape_x.to_model_proto()


def build_vnnlib_spec():
    # X[0],X[1],X[3] each 0 to 1; X[2] is 10 to 11 -- deliberately far
    # away so row-major vs column-major flattening give different
    # verdicts. Row-major: Y[0,1]=X[1], which maxes at 1.0. Y[0,1] >=
    # 10.0 is unreachable -> UNSAT. (A column-major solver would compute
    # Y[0,1]=X[2] instead, which CAN reach 10.0, and wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 10.0))
(assert (<= X[2] 11.0))
(assert (>= X[3] 0.0))
(assert (<= X[3] 1.0))

; Output Constraints
(assert (>= Y[0,1] 10.0))
(assert (<= Y[0,1] 100.0))"""


def return_expected_output():
    return "UNSAT"
