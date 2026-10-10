"""UNSAT case for ReduceMean along axis 1 with keepdims=0."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reducemean_x(x: FLOAT[2, 2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [1]))
        return op.ReduceMean(x, axes, keepdims=0)

    return reducemean_x.to_model_proto()


def build_vnnlib_spec():
    # Removing the reduced dimension gives Y = [4.0, 12.0].
    # The first mean is 4, so requiring it to be at least 6 is UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,2])
    (declare-output Y float32 [2])
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
(assert (>= Y[0] 6.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ReduceMean along axis 1 gives first mean 4, which cannot be at least 6; output shape is [2]."
