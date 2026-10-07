"""Expected UNSAT: 1D convolution over 3 elements maxes out at 3.0. Y[0,0,0] >= 4.0 is impossible."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def conv_test(x: FLOAT[1, 1, 5]) -> FLOAT[1, 1, 3]:
        w = op.Constant(value=onnx_helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 3], [1.0, 1.0, 1.0]))
        return op.Conv(x, w, pads=[0, 0])

    return conv_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 1, 5])", "    (declare-output Y float32 [1, 1, 3])", ")"]
    lines.append("; Input Constraints")
    for i in range(5):
        lines.append(f"(assert (>= X[0,0,{i}] 0.0))")
        lines.append(f"(assert (<= X[0,0,{i}] 1.0))")
    lines.append("; Output Constraints")
    lines.append("(assert (>= Y[0,0,0] 4.0))")
    lines.append("(assert (<= Y[0,0,0] 100.0))")
    return "\n".join(lines)

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: 1D convolution over 3 elements maxes out at 3.0. Y[0,0,0] >= 4.0 is impossible."
