"""Shape node test case: unsat direction, axis-sensitive. X is 2x3 (two
different axis sizes on purpose), so this checks that the solver reports
each axis's size at the correct output index -- not just that Shape
works at all. Paired with shape04.py, which targets the other axis."""

from onnxscript import FLOAT, INT64, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def shape_x(x: FLOAT[2, 3]) -> INT64[2]:
        return op.Shape(x)

    return shape_x.to_model_proto()


def build_vnnlib_spec():
    # X is 2x3, so Shape(X) is always exactly [2, 3]. Y[1] (axis 1's
    # size) is always 3. Y[1] >= 4 is unreachable -> UNSAT.
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
(assert (>= Y[1] 4))
(assert (<= Y[1] 1000))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Shape(X) for a 2x3 input, always [2, 3]. Y[1] (axis 1's size) is always 3, so Y[1] >= 4 is impossible. The solver incorrectly claimed it is satisfiable -- likely because it reported the wrong axis's size at index 1."
