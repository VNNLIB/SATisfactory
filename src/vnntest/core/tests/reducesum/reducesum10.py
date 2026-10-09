"""ReduceSum node test case: empty axes with noop_with_empty_axes=1, sat
direction. Paired with reducesum09.py. Asserts on Y[1], so the solver must
also keep the [2] output shape."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reduce_sum_noop(x: FLOAT[2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [0], []))
        return op.ReduceSum(x, axes, keepdims=1, noop_with_empty_axes=1)

    return reduce_sum_noop.to_model_proto()


def build_vnnlib_spec():
    # Same model as reducesum09. Y[1] = X[1] unchanged is 10 to 11, so
    # Y[1] >= 10.5 is reachable (witness X[1]=11.0 -> Y[1]=11.0) -> SAT.
    # A solver that summed everything into one value would have no Y[1]
    # (or Y[1] in [10, 12] only by accident of shape), and fail.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 10.0))
(assert (<= X[1] 11.0))

; Output Constraints
(assert (>= Y[1] 10.5))
(assert (<= Y[1] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ReduceSum with empty axes and noop_with_empty_axes=1 is the identity, so Y[1] = X[1], which is in [10.0, 11.0]. Finding Y[1] >= 10.5 is satisfiable (at X[1]=11.0, Y[1]=11.0). The solver failed to find this solution, likely because it ignored noop_with_empty_axes and collapsed X into a single sum."