"""Split node test case: unsat direction. Unequal sizes [1, 2]; returns
the length-2 second piece, Y = [X[1], X[2]]."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_uneven_second(x: FLOAT[3]) -> FLOAT[2]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [2], [1, 2])
        )
        _first, second = op.Split(x, sizes, axis=0)
        return second

    return split_uneven_second.to_model_proto()


def build_vnnlib_spec():
    # Split([1, 2]) makes Y = [X[1], X[2]]. Y[0] is 10 to 11, so
    # Y[0] <= 5.0 is unreachable. If the solver used [2, 1], the second
    # piece would be [X[2]] (20 to 21) and it might wrongly say SAT on
    # a different index; Y[0] <= 5 still fails either way, but Y[0]
    # coming from X[1] is the sizes check.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [3])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 10.0))
(assert (<= X[1] 11.0))
(assert (>= X[2] 20.0))
(assert (<= X[2] 21.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 5.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model splits X with sizes [1, 2] and returns the second piece, so Y[0]=X[1]. With X[1] in [10.0, 11.0], Y[0] <= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it used the wrong split sizes or returned the first piece."
