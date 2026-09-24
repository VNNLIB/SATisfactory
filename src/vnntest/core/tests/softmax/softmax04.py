"""Softmax node test case: UNSAT direction, normalization sum. The sum of
all outputs along the normalized axis must be exactly 1. Asserts that
Y_0 + Y_1 > 1.01 or Y_0 + Y_1 < 0.99 (using a small epsilon for tolerance)."""

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
    (assert (and (<= X[0] 10.0) (>= X[0] -10.0)))
    (assert (and (<= X[1] 10.0) (>= X[1] -10.0)))

    ; Output Constraints
    (assert (or (> (+ Y[0] Y[1]) 1.01) (< (+ Y[0] Y[1]) 0.99)))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: The sum of softmax outputs along the axis must be 1.0. Finding a sum outside [0.99, 1.01] indicates a failure in capturing the denominator normalization."
