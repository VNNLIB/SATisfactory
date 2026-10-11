"""UNSAT case for Expand broadcasting from [2,1] to [2,3]."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def expand_x(x: FLOAT[2, 1]) -> FLOAT[2, 3]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3]))
        return op.Expand(x, shape)

    return expand_x.to_model_proto()


def build_vnnlib_spec():
    # Broadcasting keeps the first row positive and the second row negative.
    # Requiring a newly broadcast element to have the wrong sign is UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,1])
    (declare-output Y float32 [2,3])
)

; Input Constraints
(assert (>= X[0,0] 1.0))
(assert (<= X[0,0] 2.0))
(assert (>= X[1,0] -2.0))
(assert (<= X[1,0] -1.0))

; Output Constraints
(assert (or (<= Y[0,1] 0.0) (<= Y[0,2] 0.0)
            (>= Y[1,1] 0.0) (>= Y[1,2] 0.0)))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Expand preserves each row's sign, so no newly broadcast element can satisfy the wrong-sign constraints."
