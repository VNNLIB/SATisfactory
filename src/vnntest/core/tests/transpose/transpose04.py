""" Transpose node test case: SAT direction, order-sensitive. """

from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

def build_onnx_model():
    @script()
    def transpose_x(x: FLOAT[1,2]) -> FLOAT[2,1]:
        return op.Transpose(x, perm=[1, 0])

    return transpose_x.to_model_proto()

def bulid_vnnlib_spec():
    # X[0,0] is 0 to 1; 
    # X[0,1] is 10 to 11. 
    # Correct: Y[1,0]=X[0,1].
    # Y[1,0] >= 10.0 is reachable (witness X[0,1]=10.0) -> SAT. 
    # (A solver that swapped axes would compute Y[1,0]=X[0,0] instead, which
    # CANNOT reach 10.0, and wrongly say UNSAT.)
    return """ (vnnlib-version <2.0>)
    
(declare-network N
    (declare-input X float32 [1, 2])
    (declare-output Y float32 [2,1])    
)

; Input Constraints
(assert (>= X[0,0] 0.0))
(assert (<= X[0,0] 1.0))
(assert (>= X[0,1] 10.0))
(assert (<= X[0,1] 11.0))

; Output Constraints
(assert (>= Y[1,0] 10.0))
(assert (<= Y[1,0] 100.0))

"""

def return_expected_output():
    return "SAT"

def unexpected_result_message():
    return "Expected SAT: under the correct permutation, Y[1,0] equals X[0,1], which is 10.0 to 11.0. So Y[1,0] >= 10.0 should be reachable (at X[0,1]=10.0). The solver said it isn't, likely because it mixed up the axes and used X[0,0] (which maxes at 1.0) instead of X[0,1]."