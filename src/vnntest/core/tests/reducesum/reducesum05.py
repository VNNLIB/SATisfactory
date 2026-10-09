"""ReduceSum node test case: axes=[0], keepdims=1, unsat direction. Sums
each column of a [2, 2] input and keeps the reduced axis, so the output
is [1, 2]. Column 1 is given a very different range from column 0."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # axes=[0] sums down each column. keepdims=1 keeps the reduced axis as
    # size 1, so the [2, 2] input becomes a [1, 2] output.
    @script()
    def reduce_sum_cols(x: FLOAT[2, 2]) -> FLOAT[1, 2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        return op.ReduceSum(x, axes, keepdims=1)

    return reduce_sum_cols.to_model_proto()


def build_vnnlib_spec():
    # Column 0 (X[0,0], X[1,0]) each 0 to 1; column 1 (X[0,1], X[1,1])
    # each 10 to 11. Y[0,0] = X[0,0] + X[1,0] is 0 to 2, so Y[0,0] >= 5.0
    # is unreachable -> UNSAT. (A solver that summed rows would compute
    # Y[0,0] = X[0,0] + X[0,1], which CAN reach 5.0, and wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 2])
    (declare-output Y float32 [1, 2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[1,0] 0.0))
(assert (<= X[1,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))
(assert (>= X[1,1] 10.0))
(assert (<= X[1,1] 11.0))

; Output Constraints
(assert (>= Y[0,0] 5.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ReduceSum with axes=[0], keepdims=1 sums each column, so Y[0,0] = X[0,0] + X[1,0]. With column 0 in [0.0, 1.0], Y[0,0] is at most 2.0, making Y[0,0] >= 5.0 impossible. The solver incorrectly claimed it is satisfiable, likely because it summed rows instead of columns, pulling in column 1 (10.0 to 11.0)."