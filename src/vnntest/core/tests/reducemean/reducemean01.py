"""SAT case for ReduceMean along axis 1 with keepdims=1."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reducemean_x(x: FLOAT[2, 2]) -> FLOAT[2, 1]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [1]))
        return op.ReduceMean(x, axes, keepdims=1)

    return reducemean_x.to_model_proto()


def build_vnnlib_spec():
    # The row means are (2 + 6) / 2 = 4 and (10 + 14) / 2 = 12.
    # Keeping the reduced dimension gives Y = [[4.0], [12.0]].
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,2])
    (declare-output Y float32 [2,1])
)

; Input Constraints
(assert (>= X[0,0] 2.0))
(assert (<= X[0,0] 2.0))
(assert (>= X[0,1] 6.0))
(assert (<= X[0,1] 6.0))
(assert (>= X[1,0] 10.0))
(assert (<= X[1,0] 10.0))
(assert (>= X[1,1] 14.0))
(assert (<= X[1,1] 14.0))

; Output Constraints
(assert (>= Y[0,0] 4.0))
(assert (<= Y[0,0] 4.0))
(assert (>= Y[1,0] 12.0))
(assert (<= Y[1,0] 12.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ReduceMean along axis 1 gives row means 4 and 12, with output shape [2,1]."
