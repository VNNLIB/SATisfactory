"""Arithmetic Complexity theory test: BND level, unsat direction. BND is
the base theory (VNN-LIB 2.0 Section 4.1.5): assertions may only relate
a single variable to a constant, e.g. (assert (>= Y[0] 0.5)) -- no
combining variables, no comparing two variables to each other. This is
a query-language classification, not a network one, so the network here
is kept as simple as possible -- the same Add-with-baked-bias shape as
add01/add02, deliberately NOT an elided op like Identity/Reshape (those
crash vibecheck's graph builder entirely when they're the network's only
node, which would contaminate this test with a different, already-
documented bug rather than isolating the assertion's complexity). Every
existing node-type test in this suite already happens to be BND-level;
this file exists to label and verify that explicitly, using literally
the standard's own worked example threshold (Y[0] >= 0.5, see bnd02)."""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def model(x: FLOAT[1]) -> FLOAT[1]:
        c = op.Constant(value=onnx_helper.make_tensor("c", TensorProto.FLOAT, [1], [0.5]))
        return op.Add(x, c)

    return model.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = X + 0.5, so Y is 0.5 to 1.5. Y >= 2.0 is
    # unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)

; Input Constraints
(assert (>= X[0] 0.0))
(assert (<= X[0] 1.0))

; Output Constraints
(assert (>= Y[0] 2.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = X + 0.5. With X in [0.0, 1.0], Y is at most 1.5, making Y[0] >= 2.0 impossible. This is a BND-level query -- a single variable compared to a constant, the standard's own example pattern -- so a solver claiming BND support should get this right trivially."
