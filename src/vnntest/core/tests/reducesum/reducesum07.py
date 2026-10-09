"""ReduceSum node test case: negative axis, unsat direction. axes=[-1] means
the last axis, so on a [2, 2] input it sums each row. Column 1 is given a
very different range, so reading -1 as the wrong axis changes the answer."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # axes=[-1] counts from the back: on a rank-2 input it is axis 1, so
    # this sums each row. keepdims=0 gives a [2] output.
    @script()
    def reduce_sum_last(x: FLOAT[2, 2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [-1]))
        return op.ReduceSum(x, axes, keepdims=0)

    return reduce_sum_last.to_model_proto()


def build_vnnlib_spec():
    # Column 0 (X[0,0], X[1,0]) each 0 to 1; column 1 (X[0,1], X[1,1])
    # each 10 to 11. Y[0] = X[0,0] + X[0,1] is 10 to 12, so Y[0] <= 5.0 is
    # unreachable -> UNSAT. (A solver that read -1 as axis 0 would sum
    # column 0 instead, giving Y[0] in [0, 2], and wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 2])
    (declare-output Y float32 [2])
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
(assert (>= Y[0] -100.0))
(assert (<= Y[0] 5.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ReduceSum with axes=[-1] sums along the last axis (each row), so Y[0] = X[0,0] + X[0,1]. With X[0,0] in [0.0, 1.0] and X[0,1] in [10.0, 11.0], Y[0] is at least 10.0, making Y[0] <= 5.0 impossible. The solver incorrectly claimed it is satisfiable, likely because it treated axis -1 as axis 0 and summed column 0 (0.0 to 1.0) instead."