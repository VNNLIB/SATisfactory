"""ReduceSum node test case: sat direction. Paired with reducesum01.py so
the suite covers both polarities for this node type."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reduce_sum_all(x: FLOAT[2]) -> FLOAT[1]:
        return op.ReduceSum(x)

    return reduce_sum_all.to_model_proto()


def build_vnnlib_spec():
    # Same model as reducesum01. Y[0] = X[0] + X[1] is 0 to 2. Y[0] >= 1.5
    # is reachable (witness X=[1.0, 1.0] -> Y[0]=2.0) -> SAT. A solver that
    # returned a single element instead of the sum would cap Y[0] at 1.0
    # and wrongly say UNSAT.
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
(assert (>= Y[0] 1.5))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ReduceSum with no axes sums every element, so Y[0] = X[0] + X[1]. With X[0] and X[1] in [0.0, 1.0], finding Y[0] >= 1.5 is satisfiable (at X=[1.0, 1.0], Y[0]=2.0). The solver failed to find this solution, likely because it did not add the elements together."