"""ConstantOfShape node test case: negative fractional value, unsat
direction. The constant (-2.5 everywhere) multiplies X, so a solver that
loses the sign or rounds the value gets a different answer."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # shape=[2] with value=-2.5 gives [-2.5, -2.5], so Y = X * -2.5
    # elementwise.
    @script()
    def constant_of_shape_scale(x: FLOAT[2]) -> FLOAT[2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [1], [2]))
        scale = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [-2.5])
        )
        return op.Mul(x, scale)

    return constant_of_shape_scale.to_model_proto()


def build_vnnlib_spec():
    # X[0], X[1] each 1 to 2. Y[0] = -2.5 * X[0] is -5.0 to -2.5, so
    # Y[0] >= 0.0 is unreachable -> UNSAT. (A solver that dropped the sign
    # would compute Y[0] in [2.5, 5.0] and wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 1.0))
(assert (<= X[0] 2.0))
(assert (>= X[1] 1.0))
(assert (<= X[1] 2.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ConstantOfShape with value=-2.5 gives [-2.5, -2.5], so Y = X * -2.5. With X[0] in [1.0, 2.0], Y[0] is in [-5.0, -2.5], making Y[0] >= 0.0 impossible. The solver incorrectly claimed it is satisfiable, likely because it lost the sign of the fill value."