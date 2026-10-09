"""Cast node test case: unsat direction. Mixed vector: first element is
exactly 0.0 (casts to False), second is a small nonzero value (casts to
True)."""

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
    # X[0] is 0.0 -> False -> Y[0] = 0. X[1] is 0.25 to 0.5 -> True ->
    # Y[1] = 1. Y[0] >= 0.5 is unreachable. A solver that cast 0.0 to True
    # would wrongly say SAT.
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
(assert (>= Y[0] 0.5))
(assert (<= Y[0] 1.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Cast(Cast(X, BOOL), FLOAT) + 0, which is 0.0 where X == 0.0 and 1.0 elsewhere. X[0] is fixed at 0.0 so Y[0] is 0.0, making Y[0] >= 0.5 impossible. The solver incorrectly claimed it is satisfiable, likely because it cast 0.0 to True."