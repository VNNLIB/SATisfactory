"""Neg node test case: unsat direction.

Proof that a new node type can be added by dropping files in this folder
without editing generator.py, registry.py, runner.py, or the CLI.
"""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op


def build_onnx_model():
    @script()
    def neg_x(x: FLOAT[1]) -> FLOAT[1]:
        return op.Neg(x)

    return neg_x.to_model_proto()


def build_vnnlib_spec():
    # X is 0 to 1. Y = -X, so Y is -1.0 to 0.0. Y >= 1.0 is
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
(assert (>= Y[0] 1.0))
(assert (<= Y[0] 100.0))"""


def return_expected_output():
    return "UNSAT"


def unexpected_result_message():
    return "Expected UNSAT: Model computes Y = -X. With X in [0.0, 1.0], Y is at most 0.0, making the output condition Y[0] >= 1.0 impossible. The solver incorrectly claimed it is satisfiable."
