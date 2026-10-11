"""ConstantOfShape node test case: negative fractional value, sat direction.
Paired with constantofshape07.py. Asserts on Y[1], so the constant must
cover both positions."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def constant_of_shape_scale(x: FLOAT[2]) -> FLOAT[2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [1], [2]))
        scale = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [-2.5])
        )
        return op.Mul(x, scale)

    return constant_of_shape_scale.to_model_proto()


def build_vnnlib_spec():
    # Same model as constantofshape07. Y[1] = -2.5 * X[1] is -5.0 to -2.5,
    # so Y[1] <= -4.5 is reachable (witness X[1]=2.0 -> Y[1]=-5.0) -> SAT.
    # A solver that rounded -2.5 to -2.0 would cap Y[1] at -4.0, and one
    # that dropped the sign would make Y[1] positive; both would wrongly
    # say UNSAT.
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
(assert (>= Y[1] -100.0))
(assert (<= Y[1] -4.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ConstantOfShape with value=-2.5 gives [-2.5, -2.5], so Y = X * -2.5. With X[1] in [1.0, 2.0], finding Y[1] <= -4.5 is satisfiable (at X[1]=2.0, Y[1]=-5.0). The solver failed to find this solution, likely because it lost the sign or rounded the fractional fill value."