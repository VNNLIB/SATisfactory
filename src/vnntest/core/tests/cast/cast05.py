"""Cast node test case: unsat direction. Casting to INT64 drops the
fractional part, so X in [2.1, 2.4] becomes exactly 2."""

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
    # X is 2.1 to 2.4. Cast to INT64 gives 2 (truncating or rounding
    # agree here), cast back gives Y = 2.0. Y[0] >= 2.1 is unreachable. A
    # solver that treated Cast as the identity would see Y up to 2.4 and
    # wrongly say SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 2.1))
(assert (<= X[0] 2.4))

; Output Constraints
(assert (>= Y[0] 2.1))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = Cast(Cast(X, INT64), FLOAT) + 0, which drops the fractional part. With X in [2.1, 2.4], Y is 2.0 and Y[0] >= 2.1 is impossible. The solver incorrectly claimed it is satisfiable, likely because it treated Cast as the identity."