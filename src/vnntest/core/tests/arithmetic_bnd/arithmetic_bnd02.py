"""Arithmetic Complexity theory test: BND level, sat direction. Paired
with arithmetic_bnd01.py so the suite covers both polarities at this
complexity level."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def model(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [0.5]))
        return op.Add(x, c)

    return model.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = X + 0.5, so Y is 0.5 to 1.5. Y >= 0.5 is
    # reachable (witness X=0.0 -> Y=0.5) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 0.5))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X + 0.5. With X in [0.0, 1.0], finding Y[0] >= 0.5 is satisfiable (at X=0.0, Y=0.5). This is a BND-level query -- a single variable compared to a constant -- so a solver claiming BND support should get this right trivially."
