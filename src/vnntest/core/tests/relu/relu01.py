import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    # Define your custom graph layer using Python syntax
    @script()
    def relu_layer(x: FLOAT[1]) -> FLOAT[1]:
        # Using intuitive Python operators translates directly to ONNX ops
        return op.Relu(x)

    # Convert the script function into an ONNX model object
    onnx_model = relu_layer.to_model_proto()
    return onnx_model

def build_vnnlib_spec():
    lines = []

    # VNN-LIB 2.0 declaration
    lines.append("(vnnlib-version <2.0>)")
    lines.append("")

    # neural network declaration
    lines.append("(declare-network N")
    lines.append("    (declare-input X real [1])") 
    lines.append("    (declare-output Y real [1])")
    lines.append(")")
    lines.append("")

    # Input constraints
    lines.append("; Input Constraints")
    lines.append("(assert (and (<= X[0] 0.0) (>= X[0] -1000.0)))")
    lines.append("")

    # output constraints
    lines.append("; Output Constraints")
    lines.append("(assert (or (<= Y[0] 0.0) (>= Y[0] 1.0)))")
    
    return lines