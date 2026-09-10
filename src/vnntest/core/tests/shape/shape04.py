"""Shape node test case: sat direction, axis-sensitive. Paired with
shape03.py -- same 2x3 network, but targets Y[0] (axis 0's size, which
is 2) instead of Y[1] (axis 1's size, which is 3)."""

from onnxscript import FLOAT, INT64, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def shape_x(x: FLOAT[2, 3]) -> INT64[2]:
        return op.Shape(x)

    return shape_x.to_model_proto()


def build_vnnlib_spec():
    # X is 2x3, so Shape(X) is always exactly [2, 3]. Y[0] (axis 0's
    # size) is always 2. Y[0] >= 2 is true for every legal X -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,3])
    (declare-output Y int64 [2])
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
(assert (>= Y[0] 2))
(assert (<= Y[0] 1000))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Shape(X) for a 2x3 input, always [2, 3]. Y[0] (axis 0's size) is always 2, so Y[0] >= 2 is true for every legal X. The solver failed to find this solution -- likely because it reported the wrong axis's size at index 0."
