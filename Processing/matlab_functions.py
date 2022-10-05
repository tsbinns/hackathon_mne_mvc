"""Methods for performing MATLAB-equivalent functions in Python.

METHODS
-------
linsolve_transa
-   Equivalent of calling MATLAB's 'linsolve' function with the 'TRANSA' option
    set to 'True', whereby instead of solving the equation A*X=B, the equation
    A'*X=B is solved (where ' is the complex conjugate transposed).

reshape
-   Equivalent to the MATLAB 'reshape' function, whereby the elements from the
    first axis onwards are taken in some order for the reshaping (i.e. from axis
    0 to n).

kron
-   Equivalent to the MATLAB 'kron' function, in which the Kronecker product is
    calculated.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np
import scipy as sp


def linsolve_transa(A: NDArray, B: NDArray) -> NDArray:
    """Equivalent of calling MATLAB's 'linsolve' function with the 'TRANSA'
    option set to 'True', whereby instead of solving the equation A*X=B, the
    equation A'*X=B is solved (where ' is the complex conjugate transposed).

    This is implemented in Python by calling scipy's 'linalg.solve' function
    with 'transposed' set to 'True'.

    PARAMETERS
    ----------
    A : numpy ndarray
    -   Variable A in the equation to solve.

    B : numpy ndarray
    -   Variable B in the equation to solve.

    RETURNS
    -------
    numpy array
    -   Variable X in the equation to solve.
    """
    return sp.linalg.solve(A, B, transposed=True)


def reshape(array: NDArray, dims: Union[int, tuple[int]]) -> NDArray:
    """Equivalent to the MATLAB 'reshape' function, whereby the elements from
    the first axis onwards are taken in some order for the reshaping (i.e. from
    axis 0 to n).

    This is different to numpy's method of taking elements from the last axis
    first, then the penultimate axis, and so on (i.e. from axis n to 0).

    PARAMETERS
    ----------
    array : numpy ndarray
    -   Array which will be reshaped.

    dims : int | tuple[int]
    -   The dimensions of the reshaped array.

    RETURNS
    -------
    numpy ndarray
    -   The reshaped array.

    NOTES
    -----
    -   This is equivalent to calling numpy.reshape(array, dims, order="F").
    """
    return np.reshape(array, dims, order="F")


def kron(A: NDArray, B: NDArray) -> NDArray:
    """Equivalent to the MATLAB 'kron' function, in which the Kronecker product
    is calculated.

    PARAMETERS
    ----------
    A : numpy ndarray
    -   A matrix to take the Kronecker product of with 'B'.

    B : numpy ndarray
    -   A matrix to take the Kronecker product of with 'A'.

    RETURNS
    -------
    K : numpy ndarray
    -   The Kronecker product of 'A' and 'B'.

    NOTES
    -----
    -   If the matrices are not sparse, the numpy 'kron' function is used. If
        either of the matrices are sparse, the scipy.sparse function 'kron' is
        used.
    """
    if not sp.sparse.issparse(A) and not sp.sparse.issparse(B):
        K = np.kron(A, B)
    else:
        K = sp.sparse.kron(A, B)

    return K
