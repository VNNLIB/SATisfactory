"""SAT case for the Neg operator."""
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def neg_layer(x: FLOAT[2]) -> FLOAT[2]:
        return op.Neg(x)

    return neg_layer.to_model_proto()


def build_vnnlib_spec():
    # Check that Neg flips both input signs.
    # The output bounds should be satisfiable.
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
(assert (<= Y[0] -1.0))
(assert (>= Y[0] -2.0))
(assert (<= Y[1] 2.0))
(assert (>= Y[1] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT because Neg should flip both input signs."
