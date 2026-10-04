"""Expected SAT: Multiple hidden nodes constrained simultaneously."""

import onnx
from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def hidden_model(x: FLOAT[1]) -> FLOAT[1]:
        two = op.Constant(value=onnx_helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]))
        h1 = op.Mul(x, two)
        three = op.Constant(value=onnx_helper.make_tensor("three", TensorProto.FLOAT, [], [3.0]))
        h2 = op.Add(h1, three)
        return op.Relu(h2)

    model = hidden_model.to_model_proto()
    for node in model.graph.node:
        if node.op_type == "Mul":
            node.output[0] = "hidden_out_1"
        elif node.op_type == "Add":
            node.input[0] = "hidden_out_1"
            node.output[0] = "hidden_out_2"
        elif node.op_type == "Relu":
            node.input[0] = "hidden_out_2"
    return model

def build_vnnlib_spec():
    # X in [0.0, 2.0]
    # H1 = X * 2 -> [0.0, 4.0]
    # H2 = H1 + 3 -> [3.0, 7.0]
    # Assert H1 <= 2.0 and H2 >= 4.0.
    # If X = 1.0, H1 = 2.0, H2 = 5.0 -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-hidden H1 float32 [1] "hidden_out_1")
    (declare-hidden H2 float32 [1] "hidden_out_2")
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 2.0))

; Output Constraints
(assert (and (<= H1[0] 2.0) (>= H2[0] 4.0)))
"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Constraining multiple hidden nodes."
