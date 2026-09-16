"""Conv node test case: multiple input and output channels (UNSAT)"""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def conv_test(x: FLOAT[1, 2, 2, 2]) -> FLOAT[1, 2, 1, 1]:
        w = op.Constant(value=onnx_helper.make_tensor("w", TensorProto.FLOAT, [2, 2, 2, 2], [1.0] * 16))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [2], [0.0, 0.0]))
        return op.Conv(x, w, b, pads=[0, 0, 0, 0])

    return conv_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 2, 2, 2])", "    (declare-output Y float32 [1, 2, 1, 1])", ")"]
    lines.append("; Input Constraints")
    for c in range(2):
        for i in range(2):
            for j in range(2):
                lines.append(f"(assert (>= X[0,{c},{i},{j}] 0.0))")
                lines.append(f"(assert (<= X[0,{c},{i},{j}] 1.0))")
    lines.append("; Output Constraints")
    lines.append("(assert (>= Y[0,0,0,0] 9.0))")
    lines.append("(assert (<= Y[0,0,0,0] 100.0))")
    return "\n".join(lines)

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Convolution outputs the sum of 8 elements from the input (2 channels x 2x2 spatial). The max possible is 8.0. Y[0,0,0,0] >= 9.0 is impossible."
