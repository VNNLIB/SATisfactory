"""ReduceSum node test case: unsat direction. No axes input and the default
noop_with_empty_axes=0, so ReduceSum adds up every element of X."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # Opset 18 takes `axes` as an optional input. Leaving it out (with the
    # default noop_with_empty_axes=0) reduces over all axes. keepdims
    # defaults to 1, so the [2] input becomes a [1] output.
    @script()
    def reduce_sum_all(x: FLOAT[2]) -> FLOAT[1]:
        return op.ReduceSum(x)

    return reduce_sum_all.to_model_proto()


def build_vnnlib_spec():
    # X[0], X[1] each 0 to 1. Y[0] = X[0] + X[1] is 0 to 2. Y[0] >= 3.0
    # is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= Y[0] 3.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ReduceSum with no axes sums every element, so Y[0] = X[0] + X[1]. With X[0] and X[1] in [0.0, 1.0], Y[0] is at most 2.0, making the output condition Y[0] >= 3.0 impossible. The solver incorrectly claimed it is satisfiable."