"""Shape node test case: sat direction. Paired with shape01.py so the
suite covers both polarities for this node type."""

from onnxscript import FLOAT, INT64, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def shape_x(x: FLOAT[3]) -> INT64[1]:
        return op.Shape(x)

    return shape_x.to_model_proto()


def build_vnnlib_spec():
    # X has 3 elements, so Shape(X) is always exactly [3]. Y[0] >= 3 is
    # true for every legal X (not just one witness) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [3])
    (declare-output Y int64 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 0.0))
(assert (<= X[2] 1.0))

; Output Constraints
(assert (>= Y[0] 3))
(assert (<= Y[0] 1000))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Shape(X), which is always [3] since X has 3 elements. Y[0] >= 3 is true for every legal X (witness: any X in range). The solver failed to find this solution."
