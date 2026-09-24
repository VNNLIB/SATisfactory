"""Precision-suite generator plugin.

Replace ``generate()`` with real cases, or yield ``CaseSource`` subclasses
defined in this file. Core discovery, artifact building, and the runner
do not need to change.

Example of a programmatic case (uncomment and adapt):

    from onnxscript import FLOAT, script
    from onnxscript.onnx_opset import opset18 as op
    from vnntest.core.protocols import CaseSource

    class ReluNearZero(CaseSource):
        id = "relu_near_zero"
        node_type = "relu"
        suite = "precision"
        generator = "precision"

        def build_onnx_model(self):
            @script()
            def relu_layer(x: FLOAT[1]) -> FLOAT[1]:
                return op.Relu(x)
            return relu_layer.to_model_proto()

        def build_vnnlib_spec(self):
            return '''(vnnlib-version <2.0>)
(declare-network N
    (declare-input X float32 [1])
    (declare-output Y float32 [1])
)
(assert (>= X[0] -1e-6))
(assert (<= X[0] 1e-6))
(assert (>= Y[0] 1e-3))
'''

        def expected_output(self):
            return "UNSAT"
"""

from vnntest.core.protocols import TestGenerator
from vnntest.core.registry import register_generator


class PrecisionGenerator(TestGenerator):
    name = "precision"
    suite = "precision"

    def generate(self):
        # Yield TestCase instances here. Not currently wired into the CLI
        # (only the "full" suite is exposed via ``run``) -- registered
        # here so it's ready to plug back in if this suite gets revived.
        return ()


register_generator(PrecisionGenerator())
