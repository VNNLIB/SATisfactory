"""
Add node test case: sat direction.

Same network as add01.py (Y = A + B, A,B in [0,1], Y reachable in [0,2]),
paired with it per the dev log's D5 rule: every node type needs at least
one sat case and one unsat case, or a solver that always answers the same
way would pass by accident.

Property under test: does an input exist in the box where Y >= 1.5?
Provably true: A=1.0, B=0.9 -> Y=1.9, which satisfies Y >= 1.5 -> expect SAT.
Any solver returning unsat here has a bug.
"""

import onnx
from onnx import TensorProto, helper


def build_onnx_model():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])

    add_node = helper.make_node(
        "Add", inputs=["A", "B"], outputs=["Y"], name="add_node"
    )

    graph = helper.make_graph(
        nodes=[add_node],
        name="AddGraph",
        inputs=[a, b],
        outputs=[y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = onnx.IR_VERSION
    return model


def build_vnnlib_spec():
    lines = []

    lines.append("(vnnlib-version 2.0)")
    lines.append("")

    lines.append("(declare-network N")
    lines.append("    (declare-input A float32 [1])")
    lines.append("    (declare-input B float32 [1])")
    lines.append("    (declare-output Y float32 [1])")
    lines.append(")")
    lines.append("")

    lines.append("; Input box: A, B in [0,1]")
    lines.append("(assert (>= A[0] 0.0))")
    lines.append("(assert (<= A[0] 1.0))")
    lines.append("(assert (>= B[0] 0.0))")
    lines.append("(assert (<= B[0] 1.0))")
    lines.append("")

    lines.append("; Property: does an input exist where Y >= 1.5?")
    lines.append("; Witness A=1.0, B=0.9 -> Y=1.9 -> provably true -> expect SAT")
    lines.append("(assert (>= Y[0] 1.5))")

    return lines
