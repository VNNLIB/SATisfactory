"""Expected UNSAT: Convolution outputs the sum of 4 elements (max 4.0) since there is no bias. Condition Y[0,0,0,0] >= 5.0 is impossible."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def conv_test(x: FLOAT[1, 1, 4, 4]) -> FLOAT[1, 1, 2, 2]:
        w = op.Constant(value=onnx_helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0]))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [1], [0.0]))
        return op.Conv(x, w, b, strides=[2, 2], pads=[0, 0, 0, 0])

    return conv_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 1, 4, 4])", "    (declare-output Y float32 [1, 1, 2, 2])", ")"]
    lines.append("; Input Constraints")
    for i in range(4):
        for j in range(4):
            lines.append(f"(assert (>= X[0,0,{i},{j}] 0.0))")
            lines.append(f"(assert (<= X[0,0,{i},{j}] 1.0))")
    lines.append("; Output Constraints")
    lines.append("(assert (>= Y[0,0,0,0] 5.0))")
    lines.append("(assert (<= Y[0,0,0,0] 100.0))")
    return "\n".join(lines)

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Convolution outputs the sum of 4 elements (max 4.0) since there is no bias. Condition Y[0,0,0,0] >= 5.0 is impossible."
