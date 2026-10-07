"""Softmax node test case: UNSAT direction, output domain bounds. Tests
that softmax outputs must always be strictly <= 1.0 and >= 0.0. Asserts
that Y_0 < 0.0 or Y_0 > 1.0, meaning the verifier failed to bound the outputs
correctly."""

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
    (assert (and (<= X[0] 1000.0) (>= X[0] -1000.0)))
    (assert (and (<= X[1] 1000.0) (>= X[1] -1000.0)))

    ; Output Constraints
    (assert (or (< Y[0] 0.0) (> Y[0] 1.0)))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Softmax outputs must always be between 0.0 and 1.0 inclusive. Finding a solution where Y[0] < 0.0 or Y[0] > 1.0 means the verifier failed to bound the outputs correctly."
