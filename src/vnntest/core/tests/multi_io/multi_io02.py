"""Multiple input nodes test case: SAT direction.
Two separate ONNX inputs X1  and X2, one output Y = X1 - X2."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def multi_input(x1: FLOAT[1], x2: FLOAT[1]) -> FLOAT[1]:
        return op.Sub(x1, x2)

    return multi_input.to_model_proto()

def build_vnnlib_spec():
    # X1 in [0.0, 1.0], X2 in [10.0, 11.0]. Y = X1 - X2, so Y is in [-11.0, -9.0].
    # Y <= -10.5 is reachable, e.g. witness X1 = 0.0, X2 = 11.0 gives Y = -11.0 -> SAT.
    # If the inputs were swapped, Y would be in [9.0, 11.0] and the solver would say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X1 float32 [1])
    (declare-input X2 float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X1[0] 0.0))
(assert (<= X1[0] 1.0))
(assert (>= X2[0] 10.0))
(assert (<= X2[0] 11.0))

; Output Constraints
(assert (<= Y[0] -10.5))
"""

def return_expected_output():
    return "SAT"

def unexpected_restults_message():
    return "Expected SAT: Model has two inputs and computes Y = X1 - X2. Witness X1 = 0.0, X2 = 11.0 gives Y = -11.0, which satisfies Y <= -10.5. An UNSAT answer suggests the solver mapped the declared inputs to the wrong ONNX inputs."