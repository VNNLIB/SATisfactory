"""Expected UNSAT: Model slices X backwards from 4 to 0 (exclusive) with step -2. Y[0] = X[4] and Y[1] = X[2]. X[2] is bounded by [5.0, 6.0], so Y[1] <= 4.0 is unreachable."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def slice_x(x: FLOAT[5]) -> FLOAT[2]:
        starts = op.Constant(value=onnx_helper.make_tensor("starts", TensorProto.INT64, [1], [4]))
        ends = op.Constant(value=onnx_helper.make_tensor("ends", TensorProto.INT64, [1], [0]))
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        steps = op.Constant(value=onnx_helper.make_tensor("steps", TensorProto.INT64, [1], [-2]))
        return op.Slice(x, starts, ends, axes, steps)

    return slice_x.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [5])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0)) (assert (<= X[0] 1.0))
(assert (>= X[1] 0.0)) (assert (<= X[1] 1.0))
(assert (>= X[2] 5.0)) (assert (<= X[2] 6.0))
(assert (>= X[3] 0.0)) (assert (<= X[3] 1.0))
(assert (>= X[4] 10.0)) (assert (<= X[4] 11.0))

; Output Constraints
(assert (<= Y[1] 4.0))"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Model slices X backwards from 4 to 0 (exclusive) with step -2. Y[0] = X[4] and Y[1] = X[2]. X[2] is bounded by [5.0, 6.0], so Y[1] <= 4.0 is unreachable."
