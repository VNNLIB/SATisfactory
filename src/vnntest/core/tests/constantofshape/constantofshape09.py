"""ConstantOfShape node test case: INT32 value, unsat direction. Based on
the ONNX `int32_zeros` example, but with a nonzero fill so a wrong value
shows up. The INT32 constant is cast to FLOAT so it can be added to X."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    # value is an INT32 tensor, so ConstantOfShape outputs int32 [3, 3].
    # Cast turns it into float [3.0, 3.0], and Y = X + 3.
    @script()
    def constant_of_shape_int(x: FLOAT[2]) -> FLOAT[2]:
        shape = op.Constant(value=onnx_helper.make_tensor("shape", TensorProto.INT64, [1], [2]))
        threes = op.ConstantOfShape(
            shape, value=onnx_helper.make_tensor("value", TensorProto.INT32, [1], [3])
        )
        threes_f = op.Cast(threes, to=TensorProto.FLOAT)
        return op.Add(x, threes_f)

    return constant_of_shape_int.to_model_proto()


def build_vnnlib_spec():
    # X[0], X[1] each 0 to 1. Y[0] = X[0] + 3.0 is 3 to 4, so Y[0] <= 2.5
    # is unreachable -> UNSAT. (A solver that ignored the INT32 value and
    # filled with the default 0.0 would compute Y[0] in [0, 1] and wrongly
    # say SAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2])
    (declare-output Y float32 [2])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))
(assert (>= X[1] 0.0))
(assert (<= X[1] 1.0))

; Output Constraints
(assert (>= Y[0] -100.0))
(assert (<= Y[0] 2.5))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: ConstantOfShape with an INT32 value of 3 gives [3, 3], cast to FLOAT and added to X, so Y = X + 3. With X[0] in [0.0, 1.0], Y[0] is at least 3.0, making Y[0] <= 2.5 impossible. The solver incorrectly claimed it is satisfiable, likely because it ignored the INT32 value tensor and filled with the default 0.0."