"""Split node test case: sat direction. Paired with split13.py for the
middle output of a three-way split."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def split_middle(x: FLOAT[3]) -> FLOAT[1]:
        sizes = op.Constant(
            value=onnx_helper.make_tensor("sizes", TensorProto.INT64, [3], [1, 1, 1])
        )
        _first, middle, _last = op.Split(x, sizes, axis=0)
        return middle

    return split_middle.to_model_proto()


def build_vnnlib_spec():
    # Same model as split13. Y[0] = X[1] is at least 10.0, so Y[0] >= 10.0
    # holds for every legal X -> SAT. Returning the first piece would
    # wrongly make this UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [3])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 10.0))
(assert (<= X[1] 11.0))
(assert (>= X[2] 20.0))
(assert (<= X[2] 21.0))

; Output Constraints
(assert (>= Y[0] 10.0))
(assert (<= Y[0] 11.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model splits X into three length-1 pieces and returns the middle, so Y[0]=X[1] which is always in [10.0, 11.0] (witness: any legal X). The solver failed to find this solution, likely because it returned the first piece (X[0] in [0.0, 1.0])."
