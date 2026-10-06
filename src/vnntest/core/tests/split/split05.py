"""Split node test case: unsat direction. Unequal sizes [1, 2]; returns
the length-1 first piece, Y[0] = X[0]."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_uneven_first(x: FLOAT[3]) -> FLOAT[1]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [2], [1, 2])
        )
        first, _second = op.Split(x, sizes, axis=0)
        return first

    return split_uneven_first.to_model_proto()


def build_vnnlib_spec():
    # Split([1, 2]) makes Y = [X[0]]. X[0] is 0 to 1, so Y[0] >= 5.0 is
    # unreachable. If the solver used [2, 1] instead, the first piece
    # would include X[1] (10 to 11) and it might wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [3])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 10.0))
(assert (<= X[1] 11.0))
(assert (>= X[2] 20.0))
(assert (<= X[2] 21.0))

; Output Constraints
(assert (>= Y[0] 5.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model splits X with sizes [1, 2] and returns the first piece, so Y[0]=X[0]. With X[0] in [0.0, 1.0], Y[0] >= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it ignored the split sizes."
