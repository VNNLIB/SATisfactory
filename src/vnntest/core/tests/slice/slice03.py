"""Slice node test case: slice with steps, unsat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def slice_x(x: FLOAT[5]) -> FLOAT[3]:
        starts = op.Constant(value=onnx_helper.make_tensor("starts", TensorProto.INT64, [1], [0]))
        ends = op.Constant(value=onnx_helper.make_tensor("ends", TensorProto.INT64, [1], [5]))
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        steps = op.Constant(value=onnx_helper.make_tensor("steps", TensorProto.INT64, [1], [2]))
        return op.Slice(x, starts, ends, axes, steps)

    return slice_x.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [5])
    (declare-output Y float32 [3])
)

; Input Constraints
(assert (>= X[0] 1.0)) (assert (<= X[0] 2.0))
(assert (>= X[1] -2.0)) (assert (<= X[1] -1.0))
(assert (>= X[2] 1.0)) (assert (<= X[2] 2.0))
(assert (>= X[3] -2.0)) (assert (<= X[3] -1.0))
(assert (>= X[4] 1.0)) (assert (<= X[4] 2.0))

; Output Constraints
(assert (<= Y[1] 0.0))"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Model slices X[0:5:2] into Y. Y[1] corresponds to X[2], which is strictly positive [1.0, 2.0]. The output constraint Y[1] <= 0.0 is unreachable."
