"""MaxPool node test case: basic (UNSAT)"""

from onnx import TensorProto
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def maxpool_test(x: FLOAT[1, 1, 2, 2]) -> FLOAT[1, 1, 1, 1]:
        return op.MaxPool(x, kernel_shape=[2, 2])

    return maxpool_test.to_model_proto()

def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [1, 1, 2, 2])
    (declare-output Y float32 [1, 1, 1, 1])
)
; Input Constraints
(assert (>= X[0,0,0,0] 0.0))
(assert (<= X[0,0,0,0] 1.0))
(assert (>= X[0,0,0,1] 0.0))
(assert (<= X[0,0,0,1] 1.0))
(assert (>= X[0,0,1,0] 0.0))
(assert (<= X[0,0,1,0] 1.0))
(assert (>= X[0,0,1,1] 2.0))
(assert (<= X[0,0,1,1] 3.0))

; Output Constraints
(assert (>= Y[0,0,0,0] 4.0))
"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Max possible value in the 2x2 window is 3.0, so >= 4.0 is unsatisfiable."
