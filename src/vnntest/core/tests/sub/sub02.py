"""Sub node test case: sat direction. Paired with sub01.py so the suite
covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def sub_bias(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [0.5]))
        return op.Sub(x, c)

    return sub_bias.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = X - 0.5, so Y is -0.5 to 0.5. Y >= 0.2 is
    # reachable (witness X=1.0 -> Y=0.5) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 0.2))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X - 0.5. With X in [0.0, 1.0], finding Y[0] >= 0.2 is satisfiable (at X=1.0, Y=0.5). The solver failed to find this solution."
