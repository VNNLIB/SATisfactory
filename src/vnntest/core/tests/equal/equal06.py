"""Equal node test case: sat direction. Paired with equal05.py so both
true and false Equal results are checked in one tensor."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def equal_zero(x: FLOAT[2]) -> FLOAT[2]:
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
        )
        eq = op.Equal(x, zero)
        return op.Cast(eq, to=TensorProto.FLOAT)

    return equal_zero.to_model_proto()


def build_vnnlib_spec():
    # Same model as equal05. Y[0] = 1.0 because X[0] == 0.0, so
    # Y[0] >= 0.5 holds for every legal X -> SAT. A solver that treated
    # every element as not equal would compute Y[0] = 0 and wrongly say
    # UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 0.0))
(assert (>= X[1] 1.0))
(assert (<= X[1] 2.0))

; Output Constraints
(assert (>= Y[0] 0.5))
(assert (<= Y[0] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = 1.0 where X == 0.0 else 0.0. X[0] is fixed at 0.0 so Y[0] is 1.0 (witness: any legal X). The solver failed to find this solution, likely because it treated every element as not equal."
