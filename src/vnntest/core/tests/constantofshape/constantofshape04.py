"""ConstantOfShape node test case: value=1.0 on a 2-D shape, sat direction.
Paired with constantofshape03.py."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def constant_of_shape_ones(x: FLOAT[2, 3]) -> FLOAT[2, 3]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3]))
        ones = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0])
        )
        return op.Add(x, ones)

    return constant_of_shape_ones.to_model_proto()


def build_vnnlib_spec():
    # Same model as constantofshape03. Y[1,2] = X[1,2] + 1.0 is 1 to 2, so
    # Y[1,2] >= 1.5 is reachable (witness X[1,2]=1.0 -> Y[1,2]=2.0) -> SAT.
    # A solver that filled with 0.0 instead would cap Y[1,2] at 1.0 and
    # wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 3])
    (declare-output Y float32 [2, 3])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))
(assert (>= X[0,2] 0.0))
(assert (<= X[0,2] 1.0))
(assert (>= X[1,0] 0.0))
(assert (<= X[1,0] 1.0))
(assert (>= X[1,1] 0.0))
(assert (<= X[1,1] 1.0))
(assert (>= X[1,2] 0.0))
(assert (<= X[1,2] 1.0))

; Output Constraints
(assert (>= Y[1,2] 1.5))
(assert (<= Y[1,2] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ConstantOfShape with shape [2, 3] and value=1.0 gives a 2x3 tensor of ones, so Y = X + 1. With X[1,2] in [0.0, 1.0], finding Y[1,2] >= 1.5 is satisfiable (at X[1,2]=1.0, Y[1,2]=2.0). The solver failed to find this solution, likely because it ignored the value attribute (filling with 0.0) or did not fill every position of the [2, 3] shape."