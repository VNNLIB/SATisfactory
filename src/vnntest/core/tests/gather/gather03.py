"""Gather node test case: UNSAT direction, index order sensitive. Indices are
[2, 0] (deliberately not sorted) and X[2] is given a very different range
from the rest, so Y[1] only reaches the asserted threshold if the solver
picks up X[2] where it should not."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def gather_x(x: FLOAT[4]) -> FLOAT[2]:
        indices = op.Constant(value=onnx_helper.make_tensor("indices", TensorProto.INT64, [2], [2, 0]))
        return op.Gather(x, indices, axis=0)

    return gather_x.to_model_proto()


def build_vnnlib_spec():
    # X[0],X[1],X[3] each 0 to 1; X[2] is 10 to 11. Indices [2, 0] give
    # Y[0]=X[2] and Y[1]=X[0]. Y[1] maxes at 1.0, so Y[1] >= 10.0 is
    # unreachable -> UNSAT. (A solver that ignored the order of indices and
    # gathered them sorted, [0, 2], would compute Y[1]= X[2] instead, which
    # CAN reach 10.0, and wrongly say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [4])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))
(assert (>= X[2] 10.0))
(assert (<= X[2] 11.0))
(assert (>= X[3] 0.0))
(assert (<= X[3] 1.0))

; Output Constraints
(assert (>= Y[1] 10.0))
(assert (<= Y[1] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Gather at indices [2, 0] gives Y[1]=X[0]. With X[0] in [0.0, 1.0], Y[1] >= 10.0 is impossible. The solver incorrectly claimed it is satisfiable likely because it ignored the index order and used X[2] (10.0 to 11.0) for Y[1]."