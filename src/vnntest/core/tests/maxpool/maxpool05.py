"""MaxPool node test case: dilations (UNSAT)"""

from onnx import TensorProto
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def maxpool_test(x: FLOAT[1, 1, 3, 3]) -> FLOAT[1, 1, 1, 1]:
        return op.MaxPool(x, kernel_shape=[2, 2], dilations=[2, 2])

    return maxpool_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 1, 3, 3])", "    (declare-output Y float32 [1, 1, 1, 1])", ")"]
    lines.append("; Input Constraints")
    
    # Set all elements to 0.0 except the center X[0,0,1,1]
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                lines.append(f"(assert (>= X[0,0,{i},{j}] 10.0))")
                lines.append(f"(assert (<= X[0,0,{i},{j}] 10.0))")
            else:
                lines.append(f"(assert (>= X[0,0,{i},{j}] 0.0))")
                lines.append(f"(assert (<= X[0,0,{i},{j}] 0.0))")
            
    lines.append("; Output Constraints")
    # With dilations=[2,2], the 2x2 kernel touches (0,0), (0,2), (2,0), (2,2)
    # The center (1,1) is completely ignored, so the output Y[0,0,0,0] must be 0.0.
    # If the network ignores dilations, Y would be 10.0.
    # Therefore, asserting Y >= 5.0 must be UNSAT.
    lines.append("(assert (>= Y[0,0,0,0] 5.0))")
    return "\n".join(lines)

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: With dilations=[2, 2], the center element is ignored. Output must be 0.0, so Y >= 5.0 is impossible."
