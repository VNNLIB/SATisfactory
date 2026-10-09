"""ConstantOfShape node test case: value=1.0 on a 2-D shape, unsat direction.
Based on the ONNX `float_ones` example: a tensor of ones of shape [2, 3] is
added to X. Asserts on the last element, Y[1,2], so every position of the
constant must be filled."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # shape=[2, 3] with value=1.0 gives a 2x3 tensor of ones, so Y = X + 1
    # elementwise.
    @script()
    def constant_of_shape_ones(x: FLOAT[2, 3]) -> FLOAT[2, 3]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3]))
        ones = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [1.0])
        )
        return op.Add(x, ones)

    return constant_of_shape_ones.to_model_proto()


def build_vnnlib_spec():
    # Every X entry is 0 to 1. Y[1,2] = X[1,2] + 1.0 is 1 to 2, so
    # Y[1,2] <= 0.5 is unreachable -> UNSAT. (A solver that ignored the
    # `value` attribute and filled with the default 0.0, or only filled
    # the first element, would compute Y[1,2] in [0, 1] and wrongly say
    # SAT.)
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
(assert (>= Y[1,2] -100.0))
(assert (<= Y[1,2] 0.5))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ConstantOfShape with shape [2, 3] and value=1.0 gives a 2x3 tensor of ones, so Y = X + 1. With X[1,2] in [0.0, 1.0], Y[1,2] is at least 1.0, making Y[1,2] <= 0.5 impossible. The solver incorrectly claimed it is satisfiable, likely because it ignored the value attribute (filling with 0.0) or did not fill every position of the [2, 3] shape."