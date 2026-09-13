"""flatten node test case: unsat direction"""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def flatten_x(x: FLOAT [1, 2, 2]) -> FLOAT[1, 4]:
        return op.Flatten(x, axis=1)

    return flatten_x.to_model_proto()

def build_vnnlib_spec():
    # X[0,0,0] is 0 to 1. Flatten just reshapes -> Y[0,0] = X[0,0,0], so
    # Y[0,0] is 0 to 1. Y[0,0] >= 2.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

    (declare-network N
        (declare-input X float32 [1, 2, 2])
        (declare-output Y float32 [1, 4])
    )

    ; Input Constraints
    (assert (>= X[0,0,0] 0.0))
    (assert (<= X[0,0,0] 1.0))
    (assert (>= X[0,0,1] 0.0))
    (assert (<= X[0,0,1] 1.0))
    (assert (>= X[0,1,0] 0.0))
    (assert (<= X[0,1,0] 1.0))
    (assert (>= X[0,1,1] 0.0))
    (assert (<= X[0,1,1] 1.0))
    
    ; Output Constraints
    (assert (>= Y[0,0] 2.0))
    (assert (<= Y[0,0] 100.0))
    
    """

def return_expected_output():
    return "UNSAT"

def unexpected_result_message():
    return "Expected UNSAT: Model reshapes X into Y via Flatten with values unchanged, so Y[0,0] = X[0,0,0]. With X[0,0,0] bounded in [0.0, 1.0], the output condition Y[0,0] >= 2.0 is impossible. The solver incorrectly claimed it is satisfiable."
