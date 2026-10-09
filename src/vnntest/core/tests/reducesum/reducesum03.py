"""ReduceSum node test case: axes=[1], keepdims=0, unsat direction. Sums
each row of a [2, 2] input; row 1 is given a very different range from
row 0, so Y[0] only reaches the asserted threshold if the solver sums the
wrong elements."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # axes=[1] sums along each row. keepdims=0 drops the reduced axis, so
    # the [2, 2] input becomes a [2] output.
    @script()
    def reduce_sum_rows(x: FLOAT[2, 2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [1]))
        return op.ReduceSum(x, axes, keepdims=0)

    return reduce_sum_rows.to_model_proto()


def build_vnnlib_spec():
    # Row 0 (X[0,0], X[0,1]) each 0 to 1; row 1 (X[1,0], X[1,1]) each 10
    # to 11. Y[0] = X[0,0] + X[0,1] is 0 to 2, so Y[0] >= 5.0 is
    # unreachable -> UNSAT. (A solver that summed columns would compute
    # Y[0] = X[0,0] + X[1,0], which CAN reach 5.0, and a solver that summed
    # everything would too; both would wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))
(assert (>= X[1,0] 10.0))
(assert (<= X[1,0] 11.0))
(assert (>= X[1,1] 10.0))
(assert (<= X[1,1] 11.0))

; Output Constraints
(assert (>= Y[0] 5.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ReduceSum with axes=[1], keepdims=0 sums each row, so Y[0] = X[0,0] + X[0,1]. With row 0 in [0.0, 1.0], Y[0] is at most 2.0, making Y[0] >= 5.0 impossible. The solver incorrectly claimed it is satisfiable, likely because it summed the wrong axis (columns) or every element, pulling in row 1 (10.0 to 11.0)."