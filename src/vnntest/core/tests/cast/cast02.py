"""Cast node test case: sat direction. Paired with cast01.py for a
negative input cast to BOOL."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def cast_bool(x: FLOAT[1]) -> FLOAT[1]:
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
    # Same model as cast01. Y is always 1.0, so Y[0] >= 0.5 holds for
    # every legal X -> SAT. A solver that cast negative values to False
    # would compute Y = 0 and wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] -2.0))
(assert (<= X[0] -1.0))

; Output Constraints
(assert (>= Y[0] 0.5))
(assert (<= Y[0] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Cast(Cast(X, BOOL), FLOAT) + 0, which is 1.0 for any nonzero X. With X in [-2.0, -1.0], Y is always 1.0 (witness: any legal X). The solver failed to find this solution, likely because it cast negative values to False."