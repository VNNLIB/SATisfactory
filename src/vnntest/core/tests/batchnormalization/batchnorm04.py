"""BatchNormalization node test case: sat direction, channel-sensitive.
Paired with batchnorm03.py -- same per-channel parameters, but targets
Y[0,0] (channel 0, whose own bias is +10) so the true answer is SAT. A
solver that mixed up channels would wrongly compute Y[0,0]=X[0,0]
(channel 1's bias) instead and say UNSAT."""

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
    def batchnorm_x(x: FLOAT[1, 2]) -> FLOAT[1, 2]:
        scale = op.Constant(value=onnx_helper.make_tensor("scale", TensorProto.FLOAT, [2], [1.0, 1.0]))
        b = op.Constant(value=onnx_helper.make_tensor("b", TensorProto.FLOAT, [2], [10.0, 0.0]))
        mean = op.Constant(value=onnx_helper.make_tensor("mean", TensorProto.FLOAT, [2], [0.0, 0.0]))
        var = op.Constant(value=onnx_helper.make_tensor("var", TensorProto.FLOAT, [2], [1.0, 1.0]))
        return op.BatchNormalization(x, scale, b, mean, var, epsilon=0.0)

    return _bake_constants_as_initializers(batchnorm_x.to_model_proto())


def build_vnnlib_spec():
    # X[0,0], X[0,1] each 0 to 1. Channel 0: Y[0,0]=X[0,0]+10. Y[0,0] >=
    # 10.0 is reachable (witness X[0,0]=0.0 -> Y[0,0]=10.0) -> SAT. (A
    # solver that mixed up channels would compute Y[0,0]=X[0,0] instead
    # (channel 1's parameters), which CANNOT reach 10.0, and wrongly say
    # UNSAT.)
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1,2])
    (declare-output Y float32 [1,2])
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))

; Output Constraints
(assert (>= Y[0,0] 10.0))
(assert (<= Y[0,0] 100.0))"""


def return_expected_output():
    return "SAT"


def unexpected_result_message():
    return "Expected SAT: Model applies per-channel BatchNormalization; channel 0's own parameters give Y[0,0]=X[0,0]+10. With X[0,0] in [0.0, 1.0], finding Y[0,0] >= 10.0 is satisfiable (at X[0,0]=0.0, Y[0,0]=10.0). The solver failed to find this solution -- likely because it applied channel 1's bias (+0) to channel 0's output instead."
