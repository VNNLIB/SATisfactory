"""Where node test case: unsat direction. Condition is always true, so
Y follows the X branch, not the constant false branch."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def where_true_branch(x: FLOAT[1]) -> FLOAT[1]:
        thresh = op.Constant(
            value=onnx_helper.make_tensor("thresh", TensorProto.FLOAT, [1], [-1.0])
        )
        false_val = op.Constant(
            value=onnx_helper.make_tensor("false_val", TensorProto.FLOAT, [1], [10.0])
        )
        cond = op.Greater(x, thresh)
        return op.Where(cond, x, false_val)

    return where_true_branch.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1, so X > -1 is always true. Y = X, not 10. Y >= 5.0 is
    # unreachable. A solver that took the false branch would see 10 and
    # wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 5.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Where(X > -1, X, 10). With X in [0.0, 1.0] the condition is always true, so Y = X and Y[0] >= 5.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it selected the false branch (10.0) instead."
