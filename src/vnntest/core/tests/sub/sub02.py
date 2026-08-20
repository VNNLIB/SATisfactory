"""Sub node test case: sat direction. Paired with sub01.py so the suite
covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def sub_bias(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [0.5]))
        return op.Sub(x, c)

    return sub_bias.to_model_proto()


def build_vnnlib_spec():
    lines = []

    lines.append("(vnnlib-version 2.0)")
    lines.append("")

    lines.append("(declare-network N")
    lines.append("    (declare-input X float32 [1])")
    lines.append("    (declare-output Y float32 [1])")
    lines.append(")")
    lines.append("")

    # Y = X - 0.5, X in [0,1] -> Y reachable in [-0.5, 0.5]
    lines.append("(assert (>= X[0] 0.0))")
    lines.append("(assert (<= X[0] 1.0))")
    lines.append("")

    # Witness X=1.0 -> Y=0.5, satisfies Y >= 0.2 -> expect SAT
    lines.append("(assert (>= Y[0] 0.2))")

    return lines
