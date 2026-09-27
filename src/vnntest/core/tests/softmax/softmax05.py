"""Softmax node test case: UNSAT direction, axis parameter normalization. 
Tests that Softmax normalizes along the correct axis. With a 2x2 input, 
normalizing along axis=1 means the first two elements sum to 1.0. Asserts
that Y_0 + Y_1 > 1.01."""

import onnx
from pathlib import Path
from onnxscript import FLOAT
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import numpy as np

def build_onnx_model():
    @script()
    def softmax_layer(X: FLOAT[4]) -> FLOAT[4]:
        shape = op.Constant(value_ints=[2, 2])
        flat_shape = op.Constant(value_ints=[4])
        X_reshaped = op.Reshape(X, shape)
        Y_reshaped = op.Softmax(X_reshaped, axis=1)
        return op.Reshape(Y_reshaped, flat_shape)

    return softmax_layer.to_model_proto()

def build_vnnlib_spec():
    return """
    ; VNN-LIB 2.0 declaration
    (vnnlib-version <2.0>)

    ; neural network declaration
    (declare-network N
        (declare-input X real [4])
        (declare-output Y real [4])
    )

    ; Input Constraints
    (assert (and (<= X[0] 10.0) (>= X[0] -10.0)))
    (assert (and (<= X[1] 10.0) (>= X[1] -10.0)))
    (assert (and (<= X[2] 10.0) (>= X[2] -10.0)))
    (assert (and (<= X[3] 10.0) (>= X[3] -10.0)))

    ; Output Constraints
    (assert (> (+ Y[0] Y[1]) 1.01))
    """

def return_expected_output():
    return "unsat"

def unexpected_result_message():
    return "Expected UNSAT: Softmax with axis=1 on a 2x2 tensor normalizes across the second dimension, so Y[0] + Y[1] must be 1.0. Finding a sum > 1.01 indicates it's normalizing across the wrong axis or not at all."
