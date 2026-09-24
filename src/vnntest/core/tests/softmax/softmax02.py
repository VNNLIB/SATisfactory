"""Softmax node test case: SAT direction, output domain bounds. Verifies
that outputs within the valid (0, 1) range are reachable. If the solver
returns UNSAT, it failed to find a valid assignment within this range."""

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
    (assert (and (<= X[0] 1.0) (>= X[0] -1.0)))
    (assert (and (<= X[1] 1.0) (>= X[1] -1.0)))

    ; Output Constraints
    (assert (and (>= Y[0] 0.1) (<= Y[0] 0.9)))
    """

def return_expected_output():
    return "sat"

def unexpected_result_message():
    return "Expected SAT: Softmax outputs should be reachable within the valid 0.1 to 0.9 range. The verifier failed to find a valid assignment."
