"""ReduceSum node test case: negative axis, sat direction. Paired with
reducesum07.py."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def reduce_sum_last(x: FLOAT[2, 2]) -> FLOAT[2]:
        axes = op.Constant(value=onnx_helper.make_tensor("axes", TensorProto.INT64, [1], [-1]))
        return op.ReduceSum(x, axes, keepdims=0)

    return reduce_sum_last.to_model_proto()


def build_vnnlib_spec():
    # Same model as reducesum07. Y[0] = X[0,0] + X[0,1] is 10 to 12, so
    # Y[0] >= 11.5 is reachable (witness X[0,0]=1.0, X[0,1]=11.0 ->
    # Y[0]=12.0) -> SAT. A solver that read -1 as axis 0 would get Y[0] in
    # [0, 2] and wrongly say UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[1,0] 0.0))
(assert (<= X[1,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))
(assert (>= X[1,1] 10.0))
(assert (<= X[1,1] 11.0))

; Output Constraints
(assert (>= Y[0] 11.5))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: ReduceSum with axes=[-1] sums along the last axis (each row), so Y[0] = X[0,0] + X[0,1], which is in [10.0, 12.0]. Finding Y[0] >= 11.5 is satisfiable (at X[0,0]=1.0, X[0,1]=11.0, Y[0]=12.0). The solver failed to find this solution, likely because it treated axis -1 as axis 0 and summed column 0 (0.0 to 1.0) instead."