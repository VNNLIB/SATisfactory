"""Where node test case: sat direction. Paired with where05.py so both
branches are checked in one tensor."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def where_mixed(x: FLOAT[2]) -> FLOAT[2]:
        thresh = op.Constant(
            value=onnx_helper.make_tensor("thresh", TensorProto.FLOAT, [1], [0.5])
        )
        zeros = op.Constant(
            value=onnx_helper.make_tensor("zeros", TensorProto.FLOAT, [1], [0.0])
        )
        cond = op.Greater(x, thresh)
        return op.Where(cond, x, zeros)

    return where_mixed.to_model_proto()


def build_vnnlib_spec():
    # Same model as where05. Y[1] = X[1] because X[1] > 0.5, so
    # Y[1] >= 0.6 holds for every legal X -> SAT. A solver that used
    # the false branch for every element would compute Y[1] = 0 and
    # wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 0.4))
(assert (>= X[1] 0.6))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= Y[1] 0.6))
(assert (<= Y[1] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Where(X > 0.5, X, 0). X[1] is in [0.6, 1.0] so Y[1] uses the true branch and equals X[1], which is always >= 0.6 (witness: any legal X). The solver failed to find this solution, likely because it selected the false branch for Y[1]."
