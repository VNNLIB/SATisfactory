"""Where node test case: unsat direction. Condition is always false, so
Y follows the constant Y branch, not X."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def where_false_branch(x: FLOAT[1]) -> FLOAT[1]:
        thresh = op.Constant(
            value=onnx_helper.make_tensor("thresh", TensorProto.FLOAT, [1], [5.0])
        )
        false_val = op.Constant(
            value=onnx_helper.make_tensor("false_val", TensorProto.FLOAT, [1], [10.0])
        )
        cond = op.Greater(x, thresh)
        return op.Where(cond, x, false_val)

    return where_false_branch.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1, so X > 5 is always false. Y = 10, not X. Y[0] <= 5.0
    # is unreachable. A solver that took the true branch would see X
    # and wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 5.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Where(X > 5, X, 10). With X in [0.0, 1.0] the condition is always false, so Y = 10.0 and Y[0] <= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it selected the true branch (X) instead."
