import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    # Define your custom graph layer using Python syntax
    @script()
    def relu_layer(x: FLOAT[1]) -> FLOAT[1]:
        return op.Constant(value=onnx.numpy_helper.from_array(np.zeros((1,), dtype=np.float32)))

    # Convert the script function into an ONNX model object
    onnx_model = relu_layer.to_model_proto()
    return onnx_model

def build_vnnlib_spec():
    return """
    ; VNN-LIB 2.0 declaration
    (vnnlib-version <2.0>)

    ; neural network declaration
    (declare-network N
        (declare-input X real [1])
        (declare-output Y real [1])
    )

    ; Input Constraints
    (assert (>= X[0] 1))

    ; Output Constraints
    (assert (<= Y[0] 0))
    """

def return_expected_output():
    return "SAT"