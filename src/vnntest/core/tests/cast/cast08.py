"""Cast node test case: sat direction. Paired with cast07.py for the
INT64 network output."""

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
    # Same model as cast07. Y is always 3, so Y[0] = 3 holds for every
    # legal X -> SAT. A solver that kept the fractional part or rounded up
    # would wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y int64 [1])
)

; Input Constraints
(assert (>= X[0] 3.1))
(assert (<= X[0] 3.4))

; Output Constraints
(assert (>= Y[0] 3))
(assert (<= Y[0] 3))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Cast(X, INT64) + 0. With X in [3.1, 3.4], Y is always 3 (witness: any legal X). The solver failed to find this solution, likely because it kept the fractional part or rounded the cast up."