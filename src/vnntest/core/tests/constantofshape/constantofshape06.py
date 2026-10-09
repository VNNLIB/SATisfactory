"""ConstantOfShape node test case: shape taken from Shape(X), sat direction.
Paired with constantofshape05.py."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def constant_of_input_shape(x: FLOAT[2, 3]) -> FLOAT[2, 3]:
        shape = op.Shape(x)
        return op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [5.0])
        )

    return constant_of_input_shape.to_model_proto()


def build_vnnlib_spec():
    # Same model as constantofshape05. Every Y entry is exactly 5.0, so
    # 4.5 <= Y[1,2] <= 5.5 holds for every legal X (witness: any X) -> SAT.
    # A solver that used the default fill 0.0, or could not resolve the
    # shape coming from Shape(X), would wrongly say UNSAT.
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
(assert (>= Y[1,2] 4.5))
(assert (<= Y[1,2] 5.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = ConstantOfShape(Shape(X), value=5.0), a 2x3 tensor that is exactly 5.0 everywhere, so 4.5 <= Y[1,2] <= 5.5 holds for every X (witness: any X). The solver failed to find this solution, likely because it used the default fill 0.0 or could not resolve the shape coming from Shape(X)."