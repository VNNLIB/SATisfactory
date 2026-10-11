"""ReduceSum node test case: empty axes with noop_with_empty_axes=1, unsat
direction. With an empty axes input and noop_with_empty_axes=1, ReduceSum
reduces over no axes at all, so it is the identity: Y = X."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # An explicitly empty axes tensor plus noop_with_empty_axes=1 turns the
    # reduction into a no-op, so the [2] input passes through as a [2]
    # output with values unchanged.
    @script()
    def reduce_sum_noop(x: FLOAT[2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [0], []))
        return op.ReduceSum(x, axes, keepdims=1, noop_with_empty_axes=1)

    return reduce_sum_noop.to_model_proto()


def build_vnnlib_spec():
    # X[0] is 0 to 1, X[1] is 10 to 11. Y[0] = X[0] unchanged, so
    # Y[0] >= 5.0 is unreachable -> UNSAT. (A solver that ignored
    # noop_with_empty_axes and summed everything would compute
    # Y[0] = X[0] + X[1], which CAN reach 5.0, and wrongly say SAT.)
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
(assert (>= Y[0] 5.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ReduceSum with empty axes and noop_with_empty_axes=1 is the identity, so Y[0] = X[0]. With X[0] in [0.0, 1.0], Y[0] >= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it ignored noop_with_empty_axes and summed every element, pulling in X[1] (10.0 to 11.0)."