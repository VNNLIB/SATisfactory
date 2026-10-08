"""Expected UNSAT: Contradictory hidden node constraint."""

import onnx
from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def hidden_model(x: FLOAT[1]) -> FLOAT[1]:
        two = op.Constant(value=onnx_helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]))
        h = op.Mul(x, two)
        r = op.Relu(h)
        return op.Add(r, two)

    model = hidden_model.to_model_proto()
    for node in model.graph.node:
        if node.op_type == "Relu":
            node.output[0] = "hidden_relu"
        elif node.op_type == "Add":
            node.input[0] = "hidden_relu"
    return model

def build_vnnlib_spec():
    # X in [0.0, 2.0]
    # hidden_relu = Relu(X * 2), so hidden_relu is >= 0.
    # Assert hidden_relu <= -1.0 -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-hidden H float32 [1] "hidden_relu")
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 2.0))

; Output Constraints
(assert (<= H[0] -1.0))
"""

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Hidden node is output of Relu, but constrained to be <= -1.0."
