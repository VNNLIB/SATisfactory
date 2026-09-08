import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def mul_layer(X1: FLOAT[2, 1], X2: FLOAT[1, 2]) -> FLOAT[2, 2]:
        return op.Mul(X1, X2)
    return mul_layer.to_model_proto()

def build_vnnlib_spec():
    return """
    ; VNN-LIB 2.0 declaration
    (vnnlib-version <2.0>)

    ; neural network declaration
    (declare-network N
        (declare-input X1 real [2,1])
        (declare-input X2 real [1,2])
        (declare-output Y real [2,2])
    )
    
    ; Input Constraints
    (assert (and (>= X1[0,0] 1.0) (<= X1[0,0] 2.0)))
    (assert (and (>= X1[1,0] 1.0) (<= X1[1,0] 2.0)))
    (assert (and (>= X2[0,0] 1.0) (<= X2[0,0] 2.0)))
    (assert (and (>= X2[0,1] 1.0) (<= X2[0,1] 2.0)))

    ; Output Constraints
    (assert (or (>= Y[0,0] 5.0) (<= Y[0,0] 0.0)
                (>= Y[0,1] 5.0) (<= Y[0,1] 0.0)
                (>= Y[1,0] 5.0) (<= Y[1,0] 0.0)
                (>= Y[1,1] 5.0) (<= Y[1,1] 0.0)))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X1 * X2 with tensor broadcasting. X1 and X2 elements are all in [1.0, 2.0], so Y elements should all be in [1.0, 4.0]. Finding any element of Y outside this range is unsatisfiable. The solver failed to prove this."
