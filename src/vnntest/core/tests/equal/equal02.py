"""Equal node test case: sat direction. Paired with equal01.py for the
false (not-equal) result."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def equal_zero(x: FLOAT[1]) -> FLOAT[1]:
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
        )
        eq = op.Equal(x, zero)
        return op.Cast(eq, to=TensorProto.FLOAT)

    return equal_zero.to_model_proto()


def build_vnnlib_spec():
    # Same model as equal01. Y is always 0.0, so Y[0] <= 0.0 holds for
    # every legal X -> SAT. A solver that treated Equal as always true
    # would compute Y = 1 and wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.5))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 0.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = 1.0 if X == 0.0 else 0.0. With X in [0.5, 1.0], Equal is always false, so Y is always 0.0 (witness: any legal X). The solver failed to find this solution, likely because it treated the values as equal."
