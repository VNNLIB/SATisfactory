"""Expected SAT: With padding, the pooled outputs will still come from the input elements, meaning they should remain in the bounds [2.0, 3.0]."""

from onnx import TensorProto
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def maxpool_test(x: FLOAT[1, 1, 2, 2]) -> FLOAT[1, 1, 3, 3]:
        return op.MaxPool(x, kernel_shape=[2, 2], pads=[1, 1, 1, 1])

    return maxpool_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 1, 2, 2])", "    (declare-output Y float32 [1, 1, 3, 3])", ")"]
    lines.append("; Input Constraints")
    for i in range(2):
        for j in range(2):
            lines.append(f"(assert (>= X[0,0,{i},{j}] 2.0))")
            lines.append(f"(assert (<= X[0,0,{i},{j}] 3.0))")
            
    lines.append("; Output Constraints")
    lines.append("(assert (>= Y[0,0,1,1] 2.0))")
    lines.append("(assert (<= Y[0,0,1,1] 3.0))")
    return "\n".join(lines)

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: With padding, the pooled outputs will still come from the input elements, meaning they should remain in the bounds [2.0, 3.0]."
