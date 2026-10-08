"""Split node test case: unsat direction. Three-way split [1, 1, 1];
returns the middle piece, Y[0] = X[1]."""

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
    # Y = [X[1]] in [10, 11]. Y[0] >= 15.0 is unreachable. Returning the
    # last piece (20 to 21) would wrongly make this SAT.
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
(assert (>= Y[0] 15.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model splits X into three length-1 pieces and returns the middle, so Y[0]=X[1]. With X[1] in [10.0, 11.0], Y[0] >= 15.0 is impossible. The solver incorrectly claimed it is satisfiable, likely because it returned the last piece (X[2] in [20.0, 21.0])."
