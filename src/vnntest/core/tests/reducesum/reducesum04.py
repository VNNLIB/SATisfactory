"""ReduceSum node test case: axes=[1], keepdims=0, sat direction. Paired
with reducesum03.py."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reduce_sum_rows(x: FLOAT[2, 2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [1]))
        return op.ReduceSum(x, axes, keepdims=0)

    return reduce_sum_rows.to_model_proto()


def build_vnnlib_spec():
    # Same model as reducesum03. Y[0] = X[0,0] + X[0,1] is 0 to 2, so
    # 1.5 <= Y[0] <= 2.0 is reachable (witness row 0 = [1.0, 1.0] ->
    # Y[0]=2.0) -> SAT. A solver that summed columns (Y[0] in [10, 12]) or
    # every element (Y[0] in [20, 24]) would wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))
(assert (>= X[1,0] 10.0))
(assert (<= X[1,0] 11.0))
(assert (>= X[1,1] 10.0))
(assert (<= X[1,1] 11.0))

; Output Constraints
(assert (>= Y[0] 1.5))
(assert (<= Y[0] 2.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ReduceSum with axes=[1], keepdims=0 sums each row, so Y[0] = X[0,0] + X[0,1], which is in [0.0, 2.0]. Finding 1.5 <= Y[0] <= 2.0 is satisfiable (at row 0 = [1.0, 1.0], Y[0]=2.0). The solver failed to find this solution, likely because it summed the wrong axis (columns) or every element, pulling in row 1 (10.0 to 11.0)."