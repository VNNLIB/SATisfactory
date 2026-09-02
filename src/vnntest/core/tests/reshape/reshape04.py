"""Reshape node test case: sat direction, order-sensitive. Paired with
reshape03.py -- same asymmetric bounds, but targets Y[1,0] (=X[2] under
correct row-major flattening) so the true answer is SAT. A column-major
implementation would wrongly compute Y[1,0]=X[1] instead and say UNSAT."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reshape_x(x: FLOAT[4]) -> FLOAT[2, 2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [2], [2, 2]))
        return op.Reshape(x, shape)

    return reshape_x.to_model_proto()


def build_vnnlib_spec():
    # X[0],X[1],X[3] each 0 to 1; X[2] is 10 to 11. Row-major: Y[1,0]=
    # X[2]. Y[1,0] >= 10.0 is reachable (witness X[2]=10.0) -> SAT. (A
    # column-major solver would compute Y[1,0]=X[1] instead, which CANNOT
    # reach 10.0, and wrongly say UNSAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 10.0))
(assert (<= X[2] 11.0))
(assert (>= X[3] 0.0))
(assert (<= X[3] 1.0))

; Output Constraints
(assert (>= Y[1,0] 10.0))
(assert (<= Y[1,0] 100.0))"""


def return_expected_output():
    return "SAT"
