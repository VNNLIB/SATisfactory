"""SAT case for Expand broadcasting from [2,1] to [2,3]."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def expand_x(x: FLOAT[2, 1]) -> FLOAT[2, 3]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3]))
        return op.Expand(x, shape)

    return expand_x.to_model_proto()


def build_vnnlib_spec():
    # Each input value is repeated across its output row.
    # X = [[1.0], [-1.0]] satisfies all output bounds after broadcasting.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,1])
    (declare-output Y float32 [2,3])
)

; Input Constraints
(assert (>= X[0,0] 1.0))
(assert (<= X[0,0] 2.0))
(assert (>= X[1,0] -2.0))
(assert (<= X[1,0] -1.0))

; Output Constraints
(assert (>= Y[0,0] 1.0))
(assert (<= Y[0,0] 2.0))
(assert (>= Y[0,1] 1.0))
(assert (<= Y[0,1] 2.0))
(assert (>= Y[0,2] 1.0))
(assert (<= Y[0,2] 2.0))
(assert (>= Y[1,0] -2.0))
(assert (<= Y[1,0] -1.0))
(assert (>= Y[1,1] -2.0))
(assert (<= Y[1,1] -1.0))
(assert (>= Y[1,2] -2.0))
(assert (<= Y[1,2] -1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Expand repeats each input across three columns, satisfying the positive first-row and negative second-row bounds."
