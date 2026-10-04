"""UNSAT case for the Neg operator."""
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def neg_layer(x: FLOAT[2]) -> FLOAT[2]:
        return op.Neg(x)

    return neg_layer.to_model_proto()


def build_vnnlib_spec():
    # These output constraints contradict the expected Neg output.
    # This case should be UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (<= X[0] 2.0))
(assert (>= X[0] 1.0))
(assert (<= X[1] -1.0))
(assert (>= X[1] -2.0))

; Output Constraints
(assert (or (>= Y[0] 0.0) (<= Y[1] 0.0)))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT because the output constraints contradict Neg sign flipping."
