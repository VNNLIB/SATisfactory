"""
Add node test case: unsat direction.

Network: Y = A + B, elementwise, both inputs float32 in [0,1].
Reachable output interval: Y in [0,2] (interval arithmetic; addition is
monotonic over a convex box so every value in [0,2] is actually reachable,
not just the endpoints).

Property under test: does an input exist in the box where Y >= 2.5?
Provably impossible since the maximum reachable Y is 2.0 -> expect UNSAT.
Any solver returning sat here has a bug.
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

    # VNN-LIB 2.0 declaration. No angle brackets -- the formal grammar is
    # (vnnlib-version [0-9]+.[0-9]+), not <2.0>.
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

    lines.append("; Property: does an input exist where Y >= 2.5?")
    lines.append("; Max reachable Y is 2.0 -> provably impossible -> expect UNSAT")
    lines.append("(assert (>= Y[0] 2.5))")

    return lines
