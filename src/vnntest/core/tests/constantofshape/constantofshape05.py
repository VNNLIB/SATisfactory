"""ConstantOfShape node test case: shape taken from Shape(X), unsat
direction. Y = X + ConstantOfShape(Shape(X), value=5.0), a common pattern in
exported networks: the constant's shape comes from the input, not from a
fixed tensor."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # Shape(X) is [2, 3], so ConstantOfShape builds a 2x3 tensor of 5.0.
    # It is added to X so the network still has a real op on the path
    # from input to output (a graph that is only Shape -> ConstantOfShape
    # can be folded into a constant, leaving an empty network).
    @script()
    def constant_of_input_shape(x: FLOAT[2, 3]) -> FLOAT[2, 3]:
        shape = op.Shape(x)
        fives = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.FLOAT, [1], [5.0])
        )
        return op.Add(x, fives)

    return constant_of_input_shape.to_model_proto()


def build_vnnlib_spec():
    # Every X entry is 0 to 10. Y[1,2] = X[1,2] + 5.0 is 5 to 15, so
    # Y[1,2] <= 4.5 is unreachable -> UNSAT. (A solver that could not
    # resolve the shape coming from Shape(X), or used the default fill 0.0,
    # would compute Y[1,2] = X[1,2] in [0, 10] and wrongly say SAT.)
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
(assert (>= Y[1,2] -100.0))
(assert (<= Y[1,2] 4.5))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X + ConstantOfShape(Shape(X), value=5.0), so every Y entry is X + 5.0. With X[1,2] in [0.0, 10.0], Y[1,2] is at least 5.0, making Y[1,2] <= 4.5 impossible. The solver incorrectly claimed it is satisfiable, likely because it could not resolve the shape coming from Shape(X) or used the default fill 0.0."