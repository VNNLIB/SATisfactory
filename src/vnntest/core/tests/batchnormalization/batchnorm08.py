"""BatchNormalization node test case: attribute-isolation pair, var.
Paired with batchnorm07.py -- identical network except var=1.0 instead
of 0.01, same scale/bias/mean, same X range, and the exact same
assertion (Y[0,0] >= 3.0)."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def batchnorm_x(x: FLOAT[1, 1]) -> FLOAT[1, 1]:
        scale = op.Constant(value=onnx_helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.0]))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [1], [0.0]))
        mean = op.Constant(value=onnx_helper.make_tensor("mean", TensorProto.FLOAT, [1], [0.0]))
        var = op.Constant(value=onnx_helper.make_tensor("var", TensorProto.FLOAT, [1], [1.0]))
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=0.0)

    return batchnorm_x.to_model_proto()


def build_vnnlib_spec():
    # Same network as batchnorm07.py except var=1.0. Y = X/sqrt(1.0) =
    # X, so Y is 0 to 1.0. The same assertion, Y[0,0] >= 3.0, is now
    # unreachable (max is 1.0) -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,1])
    (declare-output Y float32 [1,1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))

; Output Constraints
(assert (>= Y[0,0] 3.0))
(assert (<= Y[0,0] 1000.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X/sqrt(1.0) = X (scale=1, bias=0, mean=0, var=1.0). With X in [0.0, 1.0], Y is at most 1.0, making Y[0,0] >= 3.0 impossible. The solver incorrectly claimed it is satisfiable -- and if it also got batchnorm07 (var=0.01, otherwise identical) wrong in the opposite direction, that's a strong signal it's ignoring the var attribute entirely."
