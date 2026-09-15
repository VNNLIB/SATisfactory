import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def mul_layer(X: FLOAT[1]) -> FLOAT[1]:
        return op.Mul(X, X)
    return mul_layer.to_model_proto()

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
    (assert (and (>= X[0] -500.0) (<= X[0] 500.0)))

    ; Output Constraints
    (assert (<= Y[0] -1.0))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X * X. For any real X, X^2 >= 0. Finding Y < -1.0 is unsatisfiable. The solver failed to prove this."
