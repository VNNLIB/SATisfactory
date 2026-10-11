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
        fives = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [5.0])
        )
        return op.Add(x, fives)

    return constant_of_input_shape.to_model_proto()


def build_vnnlib_spec():
    # Same model as constantofshape05. Y[1,2] = X[1,2] + 5.0 is 5 to 15, so
    # Y[1,2] >= 14.5 is reachable (witness X[1,2]=10.0 -> Y[1,2]=15.0) ->
    # SAT. A solver that used the default fill 0.0 would cap Y[1,2] at 10.0
    # and wrongly say UNSAT.
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
(assert (>= Y[1,2] 14.5))
(assert (<= Y[1,2] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X + ConstantOfShape(Shape(X), value=5.0), so every Y entry is X + 5.0. With X[1,2] in [0.0, 10.0], finding Y[1,2] >= 14.5 is satisfiable (at X[1,2]=10.0, Y[1,2]=15.0). The solver failed to find this solution, likely because it could not resolve the shape coming from Shape(X) or used the default fill 0.0."