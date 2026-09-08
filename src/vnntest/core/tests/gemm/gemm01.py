"""Gemm node test case: unsat direction"""

from onnx import TensorProto
from onnx import helper as onnx_helper
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def gemm_layer(x: FLOAT[1, 2]) -> FLOAT[1, 1]:
        b = op.Constant(value=onnx_helper.make_tensor(
            "b", TensorProto.FLOAT, [2, 1],[1.0, 1.0]
        ))
        c = op.Constant(value=onnx_helper.make_tensor(
            "c", TensorProto.FLOAT, [1], [0.0]
        ))
        return op.Gemm(x, b, c)

    return gemm_layer.to_model_proto()

def build_vnnlib_spec():
    # X[0,0] and X[0,1] each in [0,1]. Y[0,0] = X[0,0] + X[0,1],
    # so Y[0,0] ranges 0 to 2. Y[0,0] >= 3.0 is unreachable -> UNSAT.
    return """(vnnlib-version <2.0>)

(declare-network N
    (declare-input X float32 [1, 2])
    (declare-output Y float32 [1, 1])
    )

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 0.0))
(assert (<= X[0,1] 1.0))
    
; Output Constraints
(assert (>= Y[0,0] 3.0))
(assert (<= Y[0,0] 100.0))

"""

def return_expected_output():
    return "UNSAT"
