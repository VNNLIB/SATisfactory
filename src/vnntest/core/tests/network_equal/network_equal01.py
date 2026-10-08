"""Valid Equal Networks Test."""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def model(x: FLOAT[1]) -> FLOAT[1]:
        return op.Relu(x)

    return model.to_model_proto()


def build_vnnlib_spec():
    return """(vnnlib-version <2.0>)

(declare-network N1
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

(declare-network N2
    (equal-to N1)
    (declare-input X2 float32 [1])
    (declare-output Y2 float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X2[0] 2.0))
(assert (<= X2[0] 3.0))

; Output Constraints
(assert (<= Y2[0] Y[0]))
"""


def return_expected_output():
    return "UNSAT"
