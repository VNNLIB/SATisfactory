"""Reshape node test case: unsat direction. Checks that Y[0,0] tracks
X[0] after flattening a length-4 vector into a 2x2 matrix."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reshape_x(x: FLOAT[4]) -> FLOAT[2, 2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2]))
        return op.Reshape(x, shape)

    return reshape_x.to_model_proto()


def build_vnnlib_spec():
    # X[0..3] each 0 to 1. Row-major reshape: Y[0,0]=X[0], Y[0,1]=X[1],
    # Y[1,0]=X[2], Y[1,1]=X[3]. Y[0,0] >= 2.0 is unreachable (X[0] maxes
    # at 1.0) -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 0.0))
(assert (<= X[2] 1.0))
(assert (>= X[3] 0.0))
(assert (<= X[3] 1.0))

; Output Constraints
(assert (>= Y[0,0] 2.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model reshapes X into a 2x2 matrix, Y[0,0]=X[0] unchanged. With X[0] in [0.0, 1.0], Y[0,0] is at most 1.0, making the output condition Y[0,0] >= 2.0 impossible. The solver incorrectly claimed it is satisfiable."
