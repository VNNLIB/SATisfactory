"""ConstantOfShape node test case: sat direction. Paired with
constantofshape01.py so the suite covers both polarities for this node
type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def constant_of_shape_default(x: FLOAT[2]) -> FLOAT[2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [1], [2]))
        zeros = op.ConstantOfShape(shape)
        return op.Add(x, zeros)

    return constant_of_shape_default.to_model_proto()


def build_vnnlib_spec():
    # Same model as constantofshape01. Y[0] = X[0] + 0.0 is 0 to 1, so
    # Y[0] <= 0.5 is reachable (witness X[0]=0.0 -> Y[0]=0.0) -> SAT. A
    # solver that used a default fill of 1.0 would get Y[0] >= 1.0 and
    # wrongly say UNSAT.
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
(assert (>= Y[0] -100.0))
(assert (<= Y[0] 0.5))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ConstantOfShape with no value attribute fills with 0.0, so Y = X + [0, 0] = X. With X[0] in [0.0, 1.0], finding Y[0] <= 0.5 is satisfiable (at X[0]=0.0, Y[0]=0.0). The solver failed to find this solution, likely because it used the wrong default fill value (the default is 0.0)."