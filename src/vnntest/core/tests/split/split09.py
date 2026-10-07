"""Split node test case: unsat direction. Split along axis 1; returns
the first column, Y[0,0] = X[0,0]."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_axis1_first(x: FLOAT[1, 2]) -> FLOAT[1, 1]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [2], [1, 1])
        )
        first, _second = op.Split(x, sizes, axis=1)
        return first

    return split_axis1_first.to_model_proto()


def build_vnnlib_spec():
    # X[0,0] is 0 to 1, X[0,1] is 10 to 11. Split on axis 1 makes
    # Y = [[X[0,0]]]. Y[0,0] >= 5.0 is unreachable. A solver that
    # split on axis 0 or swapped columns would wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1, 2])
    (declare-output Y float32 [1, 1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))

; Output Constraints
(assert (>= Y[0,0] 5.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model splits X along axis 1 and returns the first column, so Y[0,0]=X[0,0]. With X[0,0] in [0.0, 1.0], Y[0,0] >= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it used axis 0 or returned the second column."
