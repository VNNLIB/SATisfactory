"""ReduceSum node test case: axes=[0], keepdims=1, sat direction. Paired
with reducesum05.py. Asserts on Y[0,1], so the solver must also keep the
[1, 2] output shape."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reduce_sum_cols(x: FLOAT[2, 2]) -> FLOAT[1, 2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        return op.ReduceSum(x, axes, keepdims=1)

    return reduce_sum_cols.to_model_proto()


def build_vnnlib_spec():
    # Same model as reducesum05. Y[0,1] = X[0,1] + X[1,1] is 20 to 22, so
    # Y[0,1] >= 21.0 is reachable (witness column 1 = [11.0, 11.0] ->
    # Y[0,1]=22.0) -> SAT. A solver that summed rows would compute
    # Y[0,1] from row 1 instead, or one that dropped the kept axis would
    # have no Y[0,1] at all, and fail.
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
(assert (>= Y[0,1] 21.0))
(assert (<= Y[0,1] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ReduceSum with axes=[0], keepdims=1 sums each column into a [1, 2] output, so Y[0,1] = X[0,1] + X[1,1], which is in [20.0, 22.0]. Finding Y[0,1] >= 21.0 is satisfiable (at column 1 = [11.0, 11.0], Y[0,1]=22.0). The solver failed to find this solution, likely because it summed the wrong axis or did not keep the reduced dimension."