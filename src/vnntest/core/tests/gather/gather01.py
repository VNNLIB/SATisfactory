"""Gather node test: UNSAT direction, basic 1D gather along the axis 0"""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opest18 as op

def build_onnx_model():
    @script
    def gather_x(x: FLOAT[4]) -> FLOAT[2]:
        indices= op.Constant(value=onnx_helper.make_tensor("indices", TensorProto.INT64, [2], [1, 3]))
        return op.Gather(x, indices, axis=0)

    return gather_x.to_model_proto()

def build_vnnlib_spec():
    # X[0..3] each 0 to 1. Y[0]=X[1], Y[1]=X[3], so Y[0] is 0 to 1.
    # Y[0] >= 2.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)
    
(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 0.0))
(assert (<= X[2] 1.0))
(assert (>= X[3] 0.0))
(assert (<= X[3] 1.0))

; Output Constraints
(assert (>= Y[0] 2.0))
(assert (<= Y[0] 100.0))
"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Gather at indices [1, 3] gives Y[0]= X[1]. With X[1] in [0.0, 1.0], Y[0] >= 2.0 is impossible. The solver incorrectly claimed it is satisfiable."
