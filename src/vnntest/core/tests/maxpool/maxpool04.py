"""Expected UNSAT: With strides=[2, 2], the peak at X[0,0,1,1] only affects the top-left output Y[0,0,0,0]. The top-right output Y[0,0,0,1] must be 0.0, making >= 5.0 impossible."""

from onnx import TensorProto
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def maxpool_test(x: FLOAT[1, 1, 4, 4]) -> FLOAT[1, 1, 2, 2]:
        return op.MaxPool(x, kernel_shape=[2, 2], strides=[2, 2])

    return maxpool_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 1, 4, 4])", "    (declare-output Y float32 [1, 1, 2, 2])", ")"]
    lines.append("; Input Constraints")
    
    # Set all elements to 0.0 except X[0,0,1,1]
    for i in range(4):
        for j in range(4):
            if (i, j) == (1, 1):
                lines.append(f"(assert (>= X[0,0,{i},{j}] 10.0))")
                lines.append(f"(assert (<= X[0,0,{i},{j}] 10.0))")
            else:
                lines.append(f"(assert (>= X[0,0,{i},{j}] 0.0))")
                lines.append(f"(assert (<= X[0,0,{i},{j}] 0.0))")
            
    lines.append("; Output Constraints")
    # With strides=[2,2], Y[0,0,0,1] sees X[0:2, 2:4] which is all 0.0.
    # If the network incorrectly used strides=[1,1], Y[0,0,0,1] would see X[0:2, 1:3],
    # which includes X[0,0,1,1] = 10.0.
    # Therefore, asserting Y[0,0,0,1] >= 5.0 should be UNSAT for a correct implementation.
    lines.append("(assert (>= Y[0,0,0,1] 5.0))")
    return "\n".join(lines)

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: With strides=[2, 2], the peak at X[0,0,1,1] only affects the top-left output Y[0,0,0,0]. The top-right output Y[0,0,0,1] must be 0.0, making >= 5.0 impossible."
