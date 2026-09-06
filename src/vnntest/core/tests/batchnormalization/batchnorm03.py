"""BatchNormalization node test case: unsat direction, channel-sensitive.
Channel 0 and channel 1 are given very different parameters so that
Y[0,1] only reaches the asserted threshold if the solver applies each
channel's own scale/bias/mean/var (Y[0,1]=X[0,1]) rather than mixing up
which channel's parameters belong to which output (which would wrongly
give Y[0,1]=X[0,1]+10). Paired with batchnorm04.py, which targets the
opposite channel (Y[0,0]) to catch the same mistake from the other
direction."""

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
    # X[0,0], X[0,1] each 0 to 1. Channel 0: Y[0,0]=X[0,0]+10 (range
    # [10,11]). Channel 1: Y[0,1]=X[0,1] (range [0,1]). Y[0,1] >= 10.0 is
    # unreachable -> UNSAT. (A solver that mixed up which channel's B
    # applies where would compute Y[0,1]=X[0,1]+10 instead, which CAN
    # reach 10.0, and wrongly say SAT.)
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
(assert (>= Y[0,1] 10.0))
(assert (<= Y[0,1] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model applies per-channel BatchNormalization; channel 1's own parameters give Y[0,1]=X[0,1]. With X[0,1] in [0.0, 1.0], Y[0,1] is at most 1.0, making the output condition Y[0,1] >= 10.0 impossible. The solver incorrectly claimed it is satisfiable -- likely because it applied channel 0's bias (+10) to channel 1's output instead."
