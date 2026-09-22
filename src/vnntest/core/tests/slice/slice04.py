"""Expected SAT: Model slices axis -1 from -2 to 1000. For shape [2, 3], this extracts the last two elements of axis 1. Y[0,0] corresponds to X[0,1], bounded by [5.0, 6.0], making Y[0,0] >= 5.0 reachable."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def slice_x(x: FLOAT[2, 3]) -> FLOAT[2, 2]:
        starts = op.Constant(value=onnx_helper.make_tensor("starts", TensorProto.INT64, [1], [-2]))
        ends = op.Constant(value=onnx_helper.make_tensor("ends", TensorProto.INT64, [1], [1000]))
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [-1]))
        steps = op.Constant(value=onnx_helper.make_tensor("steps", TensorProto.INT64, [1], [1]))
        return op.Slice(x, starts, ends, axes, steps)

    return slice_x.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [2,3])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0,0] 0.0)) (assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 5.0)) (assert (<= X[0,1] 6.0))
(assert (>= X[0,2] 0.0)) (assert (<= X[0,2] 1.0))
(assert (>= X[1,0] 0.0)) (assert (<= X[1,0] 1.0))
(assert (>= X[1,1] 0.0)) (assert (<= X[1,1] 1.0))
(assert (>= X[1,2] 0.0)) (assert (<= X[1,2] 1.0))

; Output Constraints
(assert (>= Y[0,0] 5.0))"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Model slices axis -1 from -2 to 1000. For shape [2, 3], this extracts the last two elements of axis 1. Y[0,0] corresponds to X[0,1], bounded by [5.0, 6.0], making Y[0,0] >= 5.0 reachable."
