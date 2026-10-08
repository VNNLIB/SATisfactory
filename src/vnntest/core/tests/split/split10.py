"""Split node test case: sat direction. Paired with split09.py for
axis-1 first output."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_axis1_first(x: FLOAT[1, 2]) -> FLOAT[1, 1]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [2], [1, 1])
        )
        first, _second = op.Split(x, sizes, axis=1)
        return first

    return split_axis1_first.to_model_proto()


def build_vnnlib_spec():
    # Same model as split09. Y[0,0] = X[0,0] is at most 1.0, so
    # Y[0,0] <= 1.0 holds for every legal X -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1, 2])
    (declare-output Y float32 [1, 1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))

; Output Constraints
(assert (>= Y[0,0] 0.0))
(assert (<= Y[0,0] 1.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model splits X along axis 1 and returns the first column, so Y[0,0]=X[0,0] which is always in [0.0, 1.0] (witness: any legal X). The solver failed to find this solution, likely because it returned the second column (X[0,1] in [10.0, 11.0])."
