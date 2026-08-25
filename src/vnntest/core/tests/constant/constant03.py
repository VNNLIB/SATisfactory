import onnx
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def constant_layer(x: FLOAT[1]) -> FLOAT[2]:
        return op.Constant(value=onnx.numpy_helper.from_array(np.array([1.0, -1.0], dtype=np.float32)))

    onnx_model = constant_layer.to_model_proto()
    return onnx_model

def build_vnnlib_spec():
    return """
    ; VNN-LIB 2.0 declaration
    (vnnlib-version <2.0>)

    ; neural network declaration
    (declare-network N
        (declare-input X real [1])
        (declare-output Y real [2])
    )

    ; Input Constraints
    (assert (>= X[0] 0))

    ; Output Constraints (sat)
    (assert (>= Y[0] 0))
    (assert (<= Y[1] 0))
    """

def return_expected_output():
    return "SAT"
