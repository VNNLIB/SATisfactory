"""Expected UNSAT: Model computes Y = 2^X. With X in [0.0, 2.0], Y is at most 4.0, making the output condition Y[0] >= 5.0 impossible. The solver incorrectly claimed it is satisfiable."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def pow_model(x: FLOAT[1]) -> FLOAT[1]:
        base = op.Constant(value=onnx_helper.make_tensor("base", TensorProto.FLOAT, [1], [2.0]))
        return op.Pow(base, x)

    return pow_model.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 2. Y = 2^X, so Y is 1.0 to 4.0. Y >= 5.0 is
    # unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 2.0))

; Output Constraints
(assert (>= Y[0] 5.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = 2^X. With X in [0.0, 2.0], Y is at most 4.0, making the output condition Y[0] >= 5.0 impossible. The solver incorrectly claimed it is satisfiable."
