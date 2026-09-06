"""BatchNormalization node test case: sat direction. Paired with
batchnorm01.py so the suite covers both polarities for this node type."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def _bake_constants_as_initializers(model):
    # vibecheck's BatchNormalization support (frontend/onnx_loader.py) only
    # reads scale/bias/mean/var from real ONNX initializers, not from
    # separate Constant nodes -- without this, it fails with KeyError and
    # silently falls back to an unrelated handler that gives wrong verdicts.
    graph = model.graph
    remaining_nodes = []
    for node in graph.node:
        if node.op_type == "Constant":
            tensor = next(a.t for a in node.attribute if a.name == "value")
            tensor.name = node.output[0]
            graph.initializer.append(tensor)
        else:
            remaining_nodes.append(node)
    del graph.node[:]
    graph.node.extend(remaining_nodes)
    return model


def build_onnx_model():
    @script()
    def batchnorm_x(x: FLOAT[1, 1]) -> FLOAT[1, 1]:
        scale = op.Constant(value=onnx_helper.make_tensor("scale", TensorProto.FLOAT, [1], [4.0]))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [1], [0.5]))
        mean = op.Constant(value=onnx_helper.make_tensor("mean", TensorProto.FLOAT, [1], [1.0]))
        var = op.Constant(value=onnx_helper.make_tensor("var", TensorProto.FLOAT, [1], [4.0]))
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=0.0)

    return _bake_constants_as_initializers(batchnorm_x.to_model_proto())


def build_vnnlib_spec():
    # X is 0 to 1. Y = 2X - 1.5, so Y is -1.5 to 0.5. Y >= 0.3 is
    # reachable (witness X=1.0 -> Y=0.5) -> SAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,1])
    (declare-output Y float32 [1,1])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))

; Output Constraints
(assert (>= Y[0,0] 0.3))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model computes Y = (X-1.0)/sqrt(4.0)*4.0 + 0.5 = 2X - 1.5. With X in [0.0, 1.0], finding Y[0,0] >= 0.3 is satisfiable (at X=1.0, Y=0.5). The solver failed to find this solution."
