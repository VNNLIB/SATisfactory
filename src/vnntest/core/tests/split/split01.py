"""Split node test case: unsat direction. Returns the first split of a
length-2 vector; Y[0] tracks X[0], not X[1]."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_first(x: FLOAT[2]) -> FLOAT[1]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [2], [1, 1])
        )
        first, _second = op.Split(x, sizes, axis=0)
        return first

    return split_first.to_model_proto()


def build_vnnlib_spec():
    # X[0] is 0 to 1, X[1] is 10 to 11. Split([1,1]) on axis 0 makes
    # Y = [X[0]]. Y[0] >= 5.0 is unreachable for the first piece
    # (max 1.0). A solver that swapped the splits would see X[1] and
    # wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 10.0))
(assert (<= X[1] 11.0))

; Output Constraints
(assert (>= Y[0] 5.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model splits X into [X[0]] and [X[1]] and returns the first, so Y[0]=X[0]. With X[0] in [0.0, 1.0], Y[0] >= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it returned the second split (X[1] in [10.0, 11.0]) instead."
