"""Expected SAT: Model computes Y = X ^ E element-wise, where X and E are both 2x1 tensors."""

import onnx
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def pow_model(X: FLOAT[2, 1]) -> FLOAT[2, 1]:
        E = op.Constant(value=onnx.numpy_helper.from_array(np.array([[2.0], [3.0]], dtype=np.float32)))
        return op.Pow(X, E)

    return pow_model.to_model_proto()

def build_vnnlib_spec():
    # X[0,0] is in [0.0, 2.0]. E[0,0] = 2.0 -> Y[0,0] is in [0.0, 4.0]
    # X[1,0] is in [1.0, 3.0]. E[1,0] = 3.0 -> Y[1,0] is in [1.0, 27.0]
    # Condition: Y[0,0] >= 3.0 and Y[1,0] >= 20.0 -> SAT
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,1])
    (declare-output Y float32 [2,1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 2.0))
(assert (>= X[1,0] 1.0))
(assert (<= X[1,0] 3.0))

; Output Constraints
(assert (>= Y[0,0] 3.0))
(assert (>= Y[1,0] 20.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X ^ E element-wise, where X and E are 2x1 tensors. For X[0,0] in [0,2] and X[1,0] in [1,3], Y[0,0] >= 3.0 and Y[1,0] >= 20.0 is reachable (e.g., X=[[2.0], [3.0]])."
