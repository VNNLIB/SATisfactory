"""Softmax node test case: UNSAT direction, monotonicity. Softmax preserves
the relative ordering of inputs. If X_0 > X_1, then Y_0 > Y_1. We force
X_0 > X_1 and assert Y_0 <= Y_1 to test if the solver captures monotonicity."""

import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def softmax_layer(X: FLOAT[2]) -> FLOAT[2]:
        return op.Softmax(X, axis=-1)

    return softmax_layer.to_model_proto()

def build_vnnlib_spec():
    return """
    ; VNN-LIB 2.0 declaration
    (vnnlib-version <2.0>)

    ; neural network declaration
    (declare-network N
        (declare-input X real [2])
        (declare-output Y real [2])
    )

    ; Input Constraints
    (assert (and (<= X[0] 1000.0) (>= X[0] 1.0)))
    (assert (and (<= X[1] 0.0) (>= X[1] -1000.0)))

    ; Output Constraints
    (assert (<= Y[0] Y[1]))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Softmax is monotonic. Since X_0 is strictly greater than X_1, Y_0 must be strictly greater than Y_1. Finding Y_0 <= Y_1 is a violation of monotonicity."
