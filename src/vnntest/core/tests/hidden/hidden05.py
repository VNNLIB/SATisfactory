"""Expected SAT: Hidden node with multi-dimensional shape."""

import onnx
from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def hidden_model(x: FLOAT[2, 2]) -> FLOAT[2, 2]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0]))
        h = op.Add(x, c)
        return op.Relu(h)

    model = hidden_model.to_model_proto()
    for node in model.graph.node:
        if node.op_type == "Add":
            node.output[0] = "hidden_add"
        elif node.op_type == "Relu":
            node.input[0] = "hidden_add"
    return model

def build_vnnlib_spec():
    # X in [0.0, 1.0] for all elements.
    # hidden_add = X + [[1.0, 2.0], [3.0, 4.0]]
    # hidden_add[1, 0] = X[1, 0] + 3.0 -> [3.0, 4.0]
    # Assert hidden_add[1, 0] >= 3.5 -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [2, 2])
    (declare-hidden H float32 [2, 2] "hidden_add")
    (declare-output Y float32 [2, 2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))
(assert (>= X[1,0] 0.0))
(assert (<= X[1,0] 1.0))
(assert (>= X[1,1] 0.0))
(assert (<= X[1,1] 1.0))

; Output Constraints
(assert (>= H[1,0] 3.5))
"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: Constraining an element of a multi-dimensional hidden node."
