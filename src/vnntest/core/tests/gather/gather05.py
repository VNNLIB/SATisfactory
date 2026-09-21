"""Gather node test case: UNSAT direction, 2D data, axis=1, axis-sensitive.
Indices [1, 0] are valid on both axes of the 2x3 input, so a solver that
gathers along the wrong axis will still run but it just picks up the wrong
values."""

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
    # [1, 0] along axis 1 gives Y[i,0]=X[i,1] and Y[i,1]=X[i,0]. So
    # Y[1,1]=X[1,0], which maxes at 1.0. Y[1,1] >= 10.0 is unreachable ->
    # UNSAT. (A solver that gathered along axis 0 instead would compute
    # Y[1,1]=X[0,1], which CAN reach 10.0, and wrongly say SAT.)
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
(assert (>= Y[1,1] 10.0))
(assert (<= Y[1,1] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model gathers columns [1, 0] of X along axis 1, so Y[1,1]=X[1,0]. With X[1,0] in [0.0, 1.0], Y[1,1] is at most 1.0, making the output condition Y[1,1] >= 10.0 impossible. The solver incorrectly claimed it is satisfiable likely because it gathered along axis 0 instead, picking up X[0,1] (which is 10.0 to 11.0)."