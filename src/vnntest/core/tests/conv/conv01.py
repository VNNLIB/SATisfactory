"""Expected SAT: With all inputs in [0, 1], the center pixel of the padded convolution involves summing all 9 elements plus 0.5 bias. Max possible is 9.5, so >= 9.0 is satisfiable."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def conv_test(x: FLOAT[1, 1, 3, 3]) -> FLOAT[1, 1, 3, 3]:
        w = op.Constant(value=onnx_helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 3, 3], [1.0] * 9))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [1], [0.5]))
        return op.Conv(x, w, b, pads=[1, 1, 1, 1])

    return conv_test.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [1, 1, 3, 3])
    (declare-output Y float32 [1, 1, 3, 3])
)
; Input Constraints
(assert (>= X[0,0,0,0] 0.0))
(assert (<= X[0,0,0,0] 1.0))
(assert (>= X[0,0,0,1] 0.0))
(assert (<= X[0,0,0,1] 1.0))
(assert (>= X[0,0,0,2] 0.0))
(assert (<= X[0,0,0,2] 1.0))
(assert (>= X[0,0,1,0] 0.0))
(assert (<= X[0,0,1,0] 1.0))
(assert (>= X[0,0,1,1] 0.0))
(assert (<= X[0,0,1,1] 1.0))
(assert (>= X[0,0,1,2] 0.0))
(assert (<= X[0,0,1,2] 1.0))
(assert (>= X[0,0,2,0] 0.0))
(assert (<= X[0,0,2,0] 1.0))
(assert (>= X[0,0,2,1] 0.0))
(assert (<= X[0,0,2,1] 1.0))
(assert (>= X[0,0,2,2] 0.0))
(assert (<= X[0,0,2,2] 1.0))
; Output Constraints
(assert (>= Y[0,0,1,1] 9.0))
(assert (<= Y[0,0,1,1] 100.0))
"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: With all inputs in [0, 1], the center pixel of the padded convolution involves summing all 9 elements plus 0.5 bias. Max possible is 9.5, so >= 9.0 is satisfiable."
