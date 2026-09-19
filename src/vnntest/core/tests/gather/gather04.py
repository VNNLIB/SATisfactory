"""Gather node test case: SAT direction, index order sensitive. Paired with
gather03.py same indices [2, 0] and asymmetric bounds, but targets Y[0]
(=X[2]) so the true answer is SAT. A solver that ignored the indices and
returned X[0] for Y[0] would wrongly say UNSAT."""

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
    # Y[0]=X[2]. Y[0] >= 10.0 is reachable (witness X[2]=10.0) -> SAT. (A
    # solver that ignored the indices and returned X[0] for Y[0] CANNOT
    # reach 10.0, and would wrongly say UNSAT.)
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
(assert (>= Y[0] 10.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Gather at indices [2, 0] gives Y[0]=X[2]. With X[2] in [10.0, 11.0], Y[0] >= 10.0 is reachable (at X[2]=10.0). The solver failed to find this solution -- likely because it ignored the indices and used X[0] (max 1.0) for Y[0]."