"""Slice node test case: multi-dimensional slice, sat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def slice_x(x: FLOAT[3, 3]) -> FLOAT[2, 2]:
        starts = op.Constant(value=onnx_helper.make_tensor("starts", TensorProto.INT64, [2], [0, 1]))
        ends = op.Constant(value=onnx_helper.make_tensor("ends", TensorProto.INT64, [2], [2, 3]))
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [2], [0, 1]))
        steps = op.Constant(value=onnx_helper.make_tensor("steps", TensorProto.INT64, [2], [1, 1]))
        return op.Slice(x, starts, ends, axes, steps)

    return slice_x.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [3,3])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0,0] 0.0)) (assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0)) (assert (<= X[0,1] 1.0))
(assert (>= X[0,2] 0.0)) (assert (<= X[0,2] 1.0))
(assert (>= X[1,0] 0.0)) (assert (<= X[1,0] 1.0))
(assert (>= X[1,1] 0.0)) (assert (<= X[1,1] 1.0))
(assert (>= X[1,2] 10.0)) (assert (<= X[1,2] 11.0)) ; Target high value
(assert (>= X[2,0] 0.0)) (assert (<= X[2,0] 1.0))
(assert (>= X[2,1] 0.0)) (assert (<= X[2,1] 1.0))
(assert (>= X[2,2] 0.0)) (assert (<= X[2,2] 1.0))

; Output Constraints
(assert (>= Y[1,1] 10.0))"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Model slices X[0:2, 1:3] into Y. Y[1,1] corresponds to X[1,2], which is bounded by [10.0, 11.0]. The condition Y[1,1] >= 10.0 is therefore easily reachable."
