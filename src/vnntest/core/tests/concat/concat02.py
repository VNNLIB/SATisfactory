"""Concat node test case: sat direction, order-sensitive. Paired with
concat01.py same model, but targets Y[2] (the constant slice) so the
true answer is SAT. A solver that mixed up concat order (put the
constant first) would wrongly compute Y[2]=X[0] instead and say UNSAT."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def concat_layer(x: FLOAT[2]) -> FLOAT[4]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [2], [10.0, 10.0]))
        return op.Concat(x, c, axis=0)

    return concat_layer.to_model_proto()

def build_vnnlib_spec():
    # X[0], X[1] each 0 to 1. Y = [X[0], X[1], 10.0, 10.0]. Y[2] is
    # always exactly 10.0 (the constant), so Y[2] >= 10.0 is true for
    # every legal X -> SAT. (A solver that put the constant first would
    # compute Y[2]=X[0] instead, which CANNOT reach 10.0, and wrongly say
    # UNSAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [4])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= Y[2] 10.0))
(assert (<= Y[2] 100.0))

"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Model concatenates X with a constant [10.0, 10.0] along axis 0, so Y[2] is always exactly 10.0 (witness: any legal X). The solver failed to find this solution likely because it applied the concat in the wrong order, putting the constant first and computing Y[2]=X[0] instead."