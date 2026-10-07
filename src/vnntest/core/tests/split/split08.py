"""Split node test case: sat direction. Paired with split07.py for
unequal sizes [1, 2] on the second output."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_uneven_second(x: FLOAT[3]) -> FLOAT[2]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [2], [1, 2])
        )
        _first, second = op.Split(x, sizes, axis=0)
        return second

    return split_uneven_second.to_model_proto()


def build_vnnlib_spec():
    # Same model as split07. Y = [X[1], X[2]], so Y[1] = X[2] is at
    # least 20.0. Y[1] >= 20.0 holds for every legal X -> SAT. If the
    # solver used [2, 1], the second piece would be a single value and
    # would not carry X[2] at Y[1].
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [3])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 10.0))
(assert (<= X[1] 11.0))
(assert (>= X[2] 20.0))
(assert (<= X[2] 21.0))

; Output Constraints
(assert (>= Y[1] 20.0))
(assert (<= Y[1] 21.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model splits X with sizes [1, 2] and returns the second piece, so Y[1]=X[2] which is always in [20.0, 21.0] (witness: any legal X). The solver failed to find this solution, likely because it used the wrong split sizes."
