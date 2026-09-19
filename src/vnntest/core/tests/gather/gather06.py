"""Gather node test case: SAT direction, 2D data, axis=1, axis-sensitive.
Paired with gather05.py same network and input bounds, but targets Y[0,0]
(=X[0,1]) so the true answer is SAT. A solver that gathered along axis 0
would wrongly compute Y[0,0]=X[1,0] and say UNSAT."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def gather_x(x: FLOAT[2, 3]) -> FLOAT[2, 2]:
        indices = op.Constant(value=onnx_helper.make_tensor("indices", TensorProto.INT64, [2], [1, 0]))
        return op.Gather(x, indices, axis=1)

    return gather_x.to_model_proto()


def build_vnnlib_spec():
    # X[0,1] is 10 to 11; every other entry is 0 to 1. Gathering columns
    # [1, 0] along axis 1 gives Y[0,0]=X[0,1]. Y[0,0] >= 10.0 is reachable
    # (witness X[0,1]=10.0) -> SAT. (A solver that gathered along axis 0
    # instead would compute Y[0,0]=X[1,0], which CANNOT reach 10.0, and
    # wrongly say UNSAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2,3])
    (declare-output Y float32 [2,2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))
(assert (>= X[0,2] 0.0))
(assert (<= X[0,2] 1.0))
(assert (>= X[1,0] 0.0))
(assert (<= X[1,0] 1.0))
(assert (>= X[1,1] 0.0))
(assert (<= X[1,1] 1.0))
(assert (>= X[1,2] 0.0))
(assert (<= X[1,2] 1.0))

; Output Constraints
(assert (>= Y[0,0] 10.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model gathers columns [1, 0] of X along axis 1, so Y[0,0]=X[0,1]. With X[0,1] in [10.0, 11.0], finding Y[0,0] >= 10.0 is satisfiable (at X[0,1]=10.0). The solver failed to find this solution -- likely because it gathered along axis 0 instead, picking up X[1,0] (which maxes at 1.0)."