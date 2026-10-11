"""BatchNormalization node test case: attribute-isolation pair (1 of 2),
sat direction. Unlike batchnorm01/02 (same parameters, different assertion
threshold) or batchnorm03/04 (same parameters, different output index),
this pair holds bias/mean/var/X-range/assertion IDENTICAL to batchnorm06.py
and varies ONLY scale. A solver that silently ignores or defaults `scale`
would compute the same output for both files and so can be right for at
most one of the two -- it cannot pass both by coincidence, unlike the
existing pairs."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def batchnorm_x(x: FLOAT[1, 1]) -> FLOAT[1, 1]:
        scale = op.Constant(value=onnx_helper.make_tensor("scale", TensorProto.FLOAT, [1], [5.0]))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [1], [0.0]))
        mean = op.Constant(value=onnx_helper.make_tensor("mean", TensorProto.FLOAT, [1], [0.0]))
        var = op.Constant(value=onnx_helper.make_tensor("var", TensorProto.FLOAT, [1], [1.0]))
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=0.0)

    return batchnorm_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = X/sqrt(1.0)*5.0 + 0.0 = 5X, so Y is 0 to 5.0. Y >=
    # 3.0 is reachable (witness X=1.0 -> Y=5.0) -> SAT. Paired with
    # batchnorm06.py, which uses scale=2.0 instead of 5.0 -- everything
    # else (bias, mean, var, X's range, this exact assertion) is identical.
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
    return "Expected SAT: Model computes Y = X*5.0 (scale=5.0, bias=mean=0, var=1). With X in [0.0, 1.0], finding Y[0,0] >= 3.0 is satisfiable (at X=1.0, Y=5.0). The solver failed to find this solution -- and if it also got batchnorm06 (scale=2.0, otherwise identical) wrong in the opposite direction, that's a strong signal it's ignoring the scale attribute entirely rather than mishandling this one value."
