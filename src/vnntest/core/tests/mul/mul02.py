import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def mul_layer(X1: FLOAT[1], X2: FLOAT[1]) -> FLOAT[1]:
        return op.Mul(X1, X2)
    return mul_layer.to_model_proto()

def build_vnnlib_spec():
    return """
    ; VNN-LIB 2.0 declaration
    (vnnlib-version <2.0>)

    ; neural network declaration
    (declare-network N
        (declare-input X1 real [1])
        (declare-input X2 real [1])
        (declare-output Y real [1])
    )
    ; Input Constraints
    (assert (and (>= X1[0] 1.0) (<= X1[0] 1000.0)))
    (assert (and (>= X2[0] 1.0) (<= X2[0] 1000.0)))

    ; Output Constraints
    (assert (<= Y[0] 0.0))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X1 * X2. With inputs X1 >= 1.0 and X2 >= 1.0, Y >= 1.0. Finding Y < 0.0 is unsatisfiable. The solver failed to prove this."
