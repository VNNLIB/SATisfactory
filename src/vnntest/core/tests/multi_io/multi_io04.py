"""Multiple output nodes test case: SAT direction.
One input X, two separate ONNX outputs Y1 = 2 * X and Y2 = X + 10."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def multi_output(x: FLOAT[1]) -> tuple[FLOAT[1], FLOAT[1]]:
        two = op.Constant(value=onnx_helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]))
        ten = op.Constant(value=onnx_helper.make_tensor("ten", TensorProto.FLOAT, [1], [10.0]))
        y1 = op.Mul(x, two)
        y2 = op.Add(x, ten)
        return y1, y2

    return multi_output.to_model_proto()


def build_vnnlib_spec():
    # X in [0.0, 1.0]. Y1 = 2X is in [0.0, 2.0], Y2 = X + 10 is in [10.0, 11.0].
    # Y1 <= 0.5 needs X <= 0.25, Y2 >= 10.0 holds for any X >= 0.0.
    # Witness X = 0.0 gives Y1 = 0.0, Y2 = 10.0 -> SAT.
    # If the outputs were swapped, X + 10 <= 0.5 is impossible and the solver would say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y1 float32 [1])
    (declare-output Y2 float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (<= Y1[0] 0.5))
(assert (>= Y2[0] 10.0))
"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model has two outputs, Y1 = 2X and Y2 = X + 10, with X in [0.0, 1.0]. Witness X = 0.0 gives Y1 = 0.0 and Y2 = 10.0, satisfying Y1 <= 0.5 and Y2 >= 10.0. An UNSAT answer suggests the solver mapped the declared outputs to the wrong ONNX outputs."