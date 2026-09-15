"""Concat node test case: Unsat direction.
 Concatenates the network input with a constant tensor along axis 0"""

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
    # X[0], X[1] each 0 to 1. Y = [X[0], X[1], 10.0, 10.0] (X concatenated
    # with the constant [10.0, 10.0]). Y[0] >= 2.0 is unreachable (X[0]
    # maxes at 1.0) -> UNSAT.
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
(assert (>= Y[0] 2.0))
(assert (<= Y[0] 100.0))

"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Model concatenates X with a constant [10.0, 10.0] along axis 0, so Y[0]=X[0]. With X[0] in [0.0, 1.0], Y[0] is at most 1.0, making the output condition Y[0] >= 2.0 impossible. The solver incorrectly claimed it is satisfiable."
    