"""BatchNormalization node test case: attribute-isolation pair, var.
Same methodology as batchnorm05/06 (scale), applied to var:
scale/bias/mean/X-range/assertion are IDENTICAL to batchnorm08.py, only
var differs. A solver that ignores or defaults var would compute the
same output for both files and so can be right for at most one of the
two."""

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
        var = op.Constant(value=onnx_helper.make_tensor("var", TensorProto.FLOAT, [1], [0.01]))
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=0.0)

    return batchnorm_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = X/sqrt(0.01)*1.0 + 0.0 = X/0.1 = 10X, so Y is 0
    # to 10.0. Y >= 3.0 is reachable (witness X=0.3 -> Y=3.0, or X=1.0
    # -> Y=10.0) -> SAT. Paired with batchnorm08.py, which uses var=1.0
    # instead of 0.01 -- scale, bias, mean, X's range, and this exact
    # assertion are all identical.
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
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = X/sqrt(0.01) = 10X (scale=1, bias=0, mean=0, var=0.01). With X in [0.0, 1.0], finding Y[0,0] >= 3.0 is satisfiable (at X=0.3, Y=3.0). The solver failed to find this solution -- and if it also got batchnorm08 (var=1.0, otherwise identical) wrong in the opposite direction, that's a strong signal it's ignoring the var attribute entirely."
