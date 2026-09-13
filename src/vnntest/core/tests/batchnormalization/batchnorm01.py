"""BatchNormalization node test case: unsat direction. Checks that the
solver correctly applies the normalization formula (division and square
root), not just addition/multiplication like Add/Sub/MatMul."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def batchnorm_x(x: FLOAT[1, 1]) -> FLOAT[1, 1]:
        scale = op.Constant(value=onnx_helper.make_tensor("scale", TensorProto.FLOAT, [1], [4.0]))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [1], [0.5]))
        mean = op.Constant(value=onnx_helper.make_tensor("mean", TensorProto.FLOAT, [1], [1.0]))
        var = op.Constant(value=onnx_helper.make_tensor("var", TensorProto.FLOAT, [1], [4.0]))
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=0.0)

    return batchnorm_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = (X-1.0)/sqrt(4.0)*4.0 + 0.5 = 2X - 1.5, so Y is
    # -1.5 to 0.5. Y >= 2.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,1])
    (declare-output Y float32 [1,1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))

; Output Constraints
(assert (>= Y[0,0] 2.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = (X-1.0)/sqrt(4.0)*4.0 + 0.5 = 2X - 1.5. With X in [0.0, 1.0], Y is at most 0.5, making the output condition Y[0,0] >= 2.0 impossible. The solver incorrectly claimed it is satisfiable."
