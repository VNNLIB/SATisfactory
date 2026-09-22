"""Expected SAT: With dilations=[2,2], the 2x2 kernel accesses corners of the 3x3 input (4 elements). Max is 4.0, so Y[0,0,0,0] >= 3.5 is satisfiable."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def conv_test(x: FLOAT[1, 1, 3, 3]) -> FLOAT[1, 1, 1, 1]:
        w = op.Constant(value=onnx_helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0]))
        return op.Conv(x, w, dilations=[2, 2], pads=[0, 0, 0, 0])

    return conv_test.to_model_proto()

def build_vnnlib_spec():
    lines = ["(vnnlib-version <2.0>)", "(declare-network N", "    (declare-input X float32 [1, 1, 3, 3])", "    (declare-output Y float32 [1, 1, 1, 1])", ")"]
    lines.append("; Input Constraints")
    for i in range(3):
        for j in range(3):
            lines.append(f"(assert (>= X[0,0,{i},{j}] 0.0))")
            lines.append(f"(assert (<= X[0,0,{i},{j}] 1.0))")
    lines.append("; Output Constraints")
    lines.append("(assert (>= Y[0,0,0,0] 3.5))")
    lines.append("(assert (<= Y[0,0,0,0] 100.0))")
    return "\n".join(lines)

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: With dilations=[2,2], the 2x2 kernel accesses corners of the 3x3 input (4 elements). Max is 4.0, so Y[0,0,0,0] >= 3.5 is satisfiable."
