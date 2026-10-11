"""Cast node test case: unsat direction. Cast with an INT64 network
output (declared like the Shape tests). X in [3.1, 3.4] casts to exactly
3."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, INT64, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def cast_to_int(x: FLOAT[1]) -> INT64[1]:
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.INT64, [1], [0])
        )
        # Add 0 keeps the graph from being Cast-only: some solvers drop
        # Cast nodes and are left with an empty network.
        as_int = op.Cast(x, to=TensorProto.INT64)
        return op.Add(as_int, zero)

    return cast_to_int.to_model_proto()


def build_vnnlib_spec():
    # X is 3.1 to 3.4, so Y = 3 (truncating or rounding agree here).
    # Y[0] >= 4 is unreachable. A solver that rounded the cast up would
    # allow Y = 4 and wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y int64 [1])
)

; Input Constraints
(assert (>= X[0] 3.1))
(assert (<= X[0] 3.4))

; Output Constraints
(assert (>= Y[0] 4))
(assert (<= Y[0] 1000))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Cast(X, INT64) + 0. With X in [3.1, 3.4], Y is 3 and Y[0] >= 4 is impossible. The solver incorrectly claimed it is satisfiable, likely because it rounded the cast up."