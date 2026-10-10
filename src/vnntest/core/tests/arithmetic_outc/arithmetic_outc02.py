"""Arithmetic Complexity theory test: OUTC level, sat direction. Paired
with arithmetic_outc01.py so the suite covers both polarities at this
complexity level."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def model(x: FLOAT[2]) -> FLOAT[2]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [2], [0.0, 0.0]))
        return op.Add(x, c)

    return model.to_model_proto()


def build_vnnlib_spec():
    # Same network and ranges as arithmetic_outc01.py. Y[0] is always
    # less than Y[1], so Y[0] <= Y[1] holds for every legal input -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 2.0))
(assert (<= X[1] 3.0))

; Output Constraints
(assert (<= Y[0] Y[1]))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X. With X[0] in [0.0, 1.0] and X[1] in [2.0, 3.0], Y[0] is always less than Y[1], so Y[0] <= Y[1] holds for every legal input (witness: any legal X)."
