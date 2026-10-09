"""ConstantOfShape node test case: unsat direction. No `value` attribute, so
ConstantOfShape fills its output with the default value 0.0 (float32).
Adding that to X must leave X unchanged."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # ConstantOfShape's output never depends on X's values, so it is added
    # to X to put it on the path from input to output. With no `value`
    # attribute it produces zeros of shape [2], so Y = X + [0, 0] = X.
    @script()
    def constant_of_shape_default(x: FLOAT[2]) -> FLOAT[2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [1], [2]))
        zeros = op.ConstantOfShape(shape)
        return op.Add(x, zeros)

    return constant_of_shape_default.to_model_proto()


def build_vnnlib_spec():
    # X[0], X[1] each 0 to 1. Y[0] = X[0] + 0.0 is 0 to 1, so Y[0] >= 1.5
    # is unreachable -> UNSAT. (A solver that used the wrong default fill,
    # e.g. 1.0, would compute Y[0] in [1, 2], which CAN reach 1.5, and
    # wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= Y[0] 1.5))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ConstantOfShape with no value attribute fills with 0.0, so Y = X + [0, 0] = X. With X[0] in [0.0, 1.0], Y[0] is at most 1.0, making Y[0] >= 1.5 impossible. The solver incorrectly claimed it is satisfiable, likely because it used the wrong default fill value (the default is 0.0)."