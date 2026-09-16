"""MaxPool node test case: multiple channels (SAT)"""

from onnx import TensorProto
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def maxpool_test(x: FLOAT[1, 3, 2, 2]) -> FLOAT[1, 3, 1, 1]:
        return op.MaxPool(x, kernel_shape=[2, 2])

    return maxpool_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 3, 2, 2])", "    (declare-output Y float32 [1, 3, 1, 1])", ")"]
    lines.append("; Input Constraints")
    for c in range(3):
        for i in range(2):
            for j in range(2):
                lines.append(f"(assert (>= X[0,{c},{i},{j}] {c * 1.0}))")
                lines.append(f"(assert (<= X[0,{c},{i},{j}] {c * 1.0 + 1.0}))")
                
    lines.append("; Output Constraints")
    for c in range(3):
        lines.append(f"(assert (>= Y[0,{c},0,0] {c * 1.0}))")
        lines.append(f"(assert (<= Y[0,{c},0,0] {c * 1.0 + 1.0}))")
    return "\n".join(lines)

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: MaxPool operates per channel. If channel C elements are in [C, C+1], its output is in [C, C+1]."
