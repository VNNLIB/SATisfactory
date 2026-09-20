"""Transpose node test case: UNSAT direction, order sensitive. X[0,1] is given a distictive 
range so this will check  solvers who swap the axes wrong Y[0,0] should only equal X[0,0], 
not X[0,1]"""

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def transpose_x(x: FLOAT[1,2]) -> FLOAT[2,1]:
        return op.Transpose(x, perm=[1,0])

    return transpose_x.to_model_proto()

def build_vnnlib_spec():
    # X[0,0] is 0 to 1; X[0,1] is 10 to 11 deliberately far away so
    # correct vs swapped-axes give different verdicts. Correct: Y[0,0]= X[0,0]
    # which maxes at 1.0. Y[0,0] >= 10.0 is unreachable -> UNSAT.
    return """ (vnnlib-version <2.0> )
    
(declare-network N
    (declare-input X float32 [1,2])
    (declare-output Y float32 [2,1])
)
    
; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))

; Output Constraints
(assert (>= Y[0,0] 10.0))
(assert (<= Y[0,0] 100.0))

"""

def return_expected_output():
    return "UNSAT"

def unexpected_results_message():
    return "Expected UNSAT: under the correct permutation, Y[0,0] equals X[0,0], which only goes up to 1.0. So Y[0,0] >= 10.0 should be impossible. The solver said it's possible anyway likely because it mixed up the axes and used X[0,1] (10.0 to 11.0) instead of X[0,0]."
