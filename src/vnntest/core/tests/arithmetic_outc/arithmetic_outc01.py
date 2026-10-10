"""Arithmetic Complexity theory test: OUTC level, unsat direction. OUTC
allows assertions to directly compare two output variables to each other
(Y[0] >= Y[1]), while inputs must still be simple bounds -- one level up
from BND, which cannot express any variable-to-variable comparison at
all. Checks that a solver claiming OUTC support evaluates a direct
output-to-output comparison correctly."""

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
    # X[0] is 0 to 1, X[1] is 2 to 3 -- disjoint ranges, X[1] always
    # bigger. Y = X (zero bias), so Y[0] is always less than Y[1].
    # Y[0] >= Y[1] is therefore unreachable -> UNSAT.
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
(assert (>= Y[0] Y[1]))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X. With X[0] in [0.0, 1.0] and X[1] in [2.0, 3.0], Y[0] is always less than Y[1], making Y[0] >= Y[1] impossible."
