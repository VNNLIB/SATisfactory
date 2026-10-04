"""Expected SAT: Simple 2-layer network with a constraint on the hidden node."""

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
        return op.Relu(h)

    model = hidden_model.to_model_proto()
    for node in model.graph.node:
        if node.op_type == "Mul":
            node.output[0] = "hidden_out"
        elif node.op_type == "Relu":
            node.input[0] = "hidden_out"
    return model

def build_vnnlib_spec():
    # X in [0.0, 2.0]
    # hidden_out = X * 2, so hidden_out in [0.0, 4.0]
    # We assert hidden_out <= 3.0, which is SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-hidden H float32 [1] "hidden_out")
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 2.0))

; Output Constraints
(assert (<= H[0] 3.0))
"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Model computes H = X * 2. With X in [0.0, 2.0], H can be up to 4.0. The condition H <= 3.0 is reachable."
