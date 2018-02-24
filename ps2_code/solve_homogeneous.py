import numpy as np

'''
This function solves homogeneous linear system Ax = 0.
Arguments:
    matrix A
Returns:
    vector x
'''
def solveHomogeneous(A):
    U, s, VT = np.linalg.svd(A);
    V = VT.T;
    x = V[:, -1];
    return x;

