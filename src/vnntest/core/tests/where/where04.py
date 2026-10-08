"""Where node test case: sat direction. Paired with where03.py for the
false (Y) branch."""

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
    # Same model as where03. Y = 10.0 for every legal X, so Y[0] >= 10.0
    # is SAT. A solver that took the true branch would compute Y = X
    # and wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 10.0))
(assert (<= Y[0] 10.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Where(X > 5, X, 10). With X in [0.0, 1.0] the condition is always false, so Y is always 10.0 (witness: any legal X). The solver failed to find this solution, likely because it selected the true branch (X) instead."
