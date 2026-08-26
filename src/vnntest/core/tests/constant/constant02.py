import onnx
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def constant_layer(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx.numpy_helper.from_array(np.zeros((1,), dtype=np.float32)))
        return op.Add(x, c)

    onnx_model = constant_layer.to_model_proto()
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
    (assert (and (<= X[0] 0.0) (>= X[0] -1000.0)))

    ; Output Constraints (unsat)
    (assert (>= Y[0] 1.0))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X + 0. With input X <= 0.0, the output condition Y >= 1.0 is impossible. The solver incorrectly claimed it is satisfiable."