"""Cast node test case: sat direction. Paired with cast05.py for the
INT64 round trip."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def cast_int(x: FLOAT[1]) -> FLOAT[1]:
        as_int = op.Cast(x, to=TensorProto.INT64)
        back = op.Cast(as_int, to=TensorProto.FLOAT)
        zero = op.Constant(
            value=onnx_helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
        )
        # Add 0.0 keeps the graph from being Cast-only: some solvers drop
        # Cast nodes and are left with an empty network.
        return op.Add(back, zero)

    return cast_int.to_model_proto()


def build_vnnlib_spec():
    # Same model as cast05. Y is always 2.0, so Y[0] = 2.0 holds for
    # every legal X -> SAT. A solver that treated Cast as the identity
    # would compute Y in [2.1, 2.4] and wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 2.1))
(assert (<= X[0] 2.4))

; Output Constraints
(assert (>= Y[0] 2.0))
(assert (<= Y[0] 2.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = Cast(Cast(X, INT64), FLOAT) + 0, which drops the fractional part. With X in [2.1, 2.4], Y is always 2.0 (witness: any legal X). The solver failed to find this solution, likely because it treated Cast as the identity."