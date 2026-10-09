"""Cast node test case: unsat direction. X is always negative, so casting
to BOOL gives True (any nonzero float is True), and casting back to FLOAT
gives 1.0."""

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
    # X is -2 to -1, so X is never 0.0. Cast to BOOL is True, cast back
    # gives Y = 1.0. Y[0] <= 0.5 is unreachable. A solver that treated
    # negative values as False (as if the cast were X > 0) would wrongly
    # say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] -2.0))
(assert (<= X[0] -1.0))

; Output Constraints
(assert (>= Y[0] 0.0))
(assert (<= Y[0] 0.5))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Cast(Cast(X, BOOL), FLOAT) + 0, which is 1.0 for any nonzero X. With X in [-2.0, -1.0], Y is 1.0 and Y[0] <= 0.5 is impossible. The solver incorrectly claimed it is satisfiable, likely because it cast negative values to False."