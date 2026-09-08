"""Shape node test case: unsat direction. Unlike every other node type in
this suite, Shape's output never depends on X's actual values -- only on
X's declared dimensions. Checks that the solver understands the output
is a fixed constant, not something that varies with X."""

from onnxscript import FLOAT, INT64, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def shape_x(x: FLOAT[3]) -> INT64[1]:
        return op.Shape(x)

    return shape_x.to_model_proto()


def build_vnnlib_spec():
    # X has 3 elements, so Shape(X) is always exactly [3] -- true for
    # every legal X, regardless of X's actual values. Y[0] >= 4 is
    # unreachable -> UNSAT.
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
(assert (>= Y[0] 4))
(assert (<= Y[0] 1000))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Shape(X), which is always [3] since X has 3 elements. Y[0] >= 4 is impossible for any X. The solver incorrectly claimed it is satisfiable."
