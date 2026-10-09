"""Where node test case: unsat direction. Mixed condition: first element
takes the false branch, second takes the true branch."""

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
    # X[0] is 0 to 0.4 so condition is false -> Y[0] = 0.
    # X[1] is 0.6 to 1.0 so condition is true -> Y[1] = X[1].
    # Y[0] >= 0.3 is unreachable. A solver that used X for both
    # elements could wrongly say SAT.
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
(assert (>= Y[0] 0.3))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Where(X > 0.5, X, 0). X[0] is in [0.0, 0.4] so Y[0] uses the false branch and is 0.0, making Y[0] >= 0.3 impossible. The solver incorrectly claimed it is satisfiable, likely because it selected the true branch for Y[0]."
