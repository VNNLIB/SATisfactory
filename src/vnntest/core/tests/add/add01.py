"""Add node test case: unsat direction."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

EXPECTED_RESULT = "unsat"

def build_onnx_model():
    # Second operand baked in as a constant rather than a second declared
    # input -- keeps the network single-input, far more broadly supported
    # across solvers than a two-input network.
    #
    # Uses the tensor-valued `value` attribute explicitly rather than
    # onnxscript's default `value_float` scalar form: vibecheck silently
    # ignores Constant nodes built with `value_float`, so the plain
    # `x + 0.5` shorthand produces a wrong verdict there.
    @script()
    def add_bias(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [0.5]))
        return op.Add(x, c)

    return add_bias.to_model_proto()


def build_vnnlib_spec():
    lines = []

    lines.append("(vnnlib-version 2.0)")
    lines.append("")

    lines.append("(declare-network N")
    lines.append("    (declare-input X float32 [1])")
    lines.append("    (declare-output Y float32 [1])")
    lines.append(")")
    lines.append("")

    # Y = X + 0.5, X in [0,1] -> Y reachable in [0.5, 1.5]
    lines.append("(assert (>= X[0] 0.0))")
    lines.append("(assert (<= X[0] 1.0))")
    lines.append("")

    # Max reachable Y is 1.5, so Y >= 2.0 is unreachable -> expect UNSAT
    lines.append("(assert (>= Y[0] 2.0))")

    return lines
