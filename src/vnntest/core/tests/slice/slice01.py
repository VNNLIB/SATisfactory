"""Expected UNSAT: Model slices X[1:3] into Y. Y[0] corresponds to X[1], which is bounded by [2.0, 3.0]. The condition Y[0] >= 4.0 is therefore unreachable."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def slice_x(x: FLOAT[4]) -> FLOAT[2]:
        starts = op.Constant(value=onnx_helper.make_tensor("starts", TensorProto.INT64, [1], [1]))
        ends = op.Constant(value=onnx_helper.make_tensor("ends", TensorProto.INT64, [1], [3]))
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        steps = op.Constant(value=onnx_helper.make_tensor("steps", TensorProto.INT64, [1], [1]))
        return op.Slice(x, starts, ends, axes, steps)

    return slice_x.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 2.0))
(assert (<= X[1] 3.0))
(assert (>= X[2] 4.0))
(assert (<= X[2] 5.0))
(assert (>= X[3] 6.0))
(assert (<= X[3] 7.0))

; Output Constraints
(assert (>= Y[0] 4.0))"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Model slices X[1:3] into Y. Y[0] corresponds to X[1], which is bounded by [2.0, 3.0]. The condition Y[0] >= 4.0 is therefore unreachable."
