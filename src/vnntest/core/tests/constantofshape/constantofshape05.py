"""ConstantOfShape node test case: shape taken from Shape(X), unsat
direction. Y = ConstantOfShape(Shape(X), value=5.0), a common pattern in
exported networks. Like the Shape tests, the output never depends on X's
values -- every Y entry is exactly 5.0 for every legal X."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # Shape(X) is [2, 3], so ConstantOfShape builds a 2x3 tensor filled
    # with 5.0. X's values are never used, only its dimensions.
    @script()
    def constant_of_input_shape(x: FLOAT[2, 3]) -> FLOAT[2, 3]:
        shape = op.Shape(x)
        return op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [5.0])
        )

    return constant_of_input_shape.to_model_proto()


def build_vnnlib_spec():
    # Every Y entry is exactly 5.0 regardless of X, so Y[1,2] >= 6.0 is
    # unreachable -> UNSAT. (A solver that passed X through, or treated the
    # output as depending on X, could not prove this; one that used the
    # default fill 0.0 would also be wrong, see constantofshape06.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 3])
    (declare-output Y float32 [2, 3])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 10.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 10.0))
(assert (>= X[0,2] 0.0))
(assert (<= X[0,2] 10.0))
(assert (>= X[1,0] 0.0))
(assert (<= X[1,0] 10.0))
(assert (>= X[1,1] 0.0))
(assert (<= X[1,1] 10.0))
(assert (>= X[1,2] 0.0))
(assert (<= X[1,2] 10.0))

; Output Constraints
(assert (>= Y[1,2] 6.0))
(assert (<= Y[1,2] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = ConstantOfShape(Shape(X), value=5.0), a 2x3 tensor that is exactly 5.0 everywhere for every X. Y[1,2] >= 6.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it treated the output as depending on X's values (X ranges up to 10.0) instead of being a constant."