"""Cast node test case: sat direction. Paired with cast03.py so both
False and True cast results are checked in one tensor."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def cast_bool(x: FLOAT[2]) -> FLOAT[2]:
        as_bool = op.Cast(x, to=TensorProto.BOOL)
        back = op.Cast(as_bool, to=TensorProto.FLOAT)
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
        )
        # Add 0.0 keeps the graph from being Cast-only: some solvers drop
        # Cast nodes and are left with an empty network.
        return op.Add(back, zero)

    return cast_bool.to_model_proto()


def build_vnnlib_spec():
    # Same model as cast03. Y[1] = 1.0 because X[1] is nonzero, so
    # Y[1] >= 0.75 holds for every legal X -> SAT. A solver that treated
    # Cast as the identity would compute Y[1] = X[1], at most 0.5, and
    # wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 0.0))
(assert (>= X[1] 0.25))
(assert (<= X[1] 0.5))

; Output Constraints
(assert (>= Y[1] 0.75))
(assert (<= Y[1] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Cast(Cast(X, BOOL), FLOAT) + 0, which is 0.0 where X == 0.0 and 1.0 elsewhere. X[1] is in [0.25, 0.5], never 0.0, so Y[1] is 1.0 (witness: any legal X). The solver failed to find this solution, likely because it treated Cast as the identity and kept Y[1] = X[1]."