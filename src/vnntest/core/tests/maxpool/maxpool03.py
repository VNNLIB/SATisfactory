"""MaxPool node test case: ceil_mode (SAT)"""

from onnx import TensorProto
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def maxpool_test(x: FLOAT[1, 1, 3, 3]) -> FLOAT[1, 1, 2, 2]:
        return op.MaxPool(x, kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)

    return maxpool_test.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [1, 1, 3, 3])
    (declare-output Y float32 [1, 1, 2, 2])
)
; Input Constraints
(assert (>= X[0,0,2,2] 5.0))
(assert (<= X[0,0,2,2] 6.0))

; Output Constraints
(assert (>= Y[0,0,1,1] 5.0))
(assert (<= Y[0,0,1,1] 6.0))
"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: With ceil_mode=1, the 3x3 input results in a 2x2 output. The bottom-right element of the input (at 2,2) will be captured by the bottom-right pooling window, so Y[0,0,1,1] should match X[0,0,2,2]."
