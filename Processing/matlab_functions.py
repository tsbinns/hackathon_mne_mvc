"""Functions for performing MATLAB-equivalent functions in Python.

FUNCTIONS
---------
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
from numpy.typing import ArrayLike, NDArray
import numpy as np
import scipy as sp


def linsolve_transa(A: ArrayLike, B: ArrayLike) -> NDArray:
    """Equivalent of calling MATLAB's 'linsolve' function with the 'TRANSA'
    option set to 'True', whereby instead of solving the equation A*X=B, the
    equation A'*X=B is solved (where ' is the complex conjugate transposed).

    PARAMETERS
    ----------
    A : array-like
    -   Variable A in the equation to solve.

    B : array-like
    -   Variable B in the equation to solve.

    RETURNS
    -------
    X : numpy ndarray
    -   Variable X in the equation to solve.

    NOTES
    -----
    -   This is implemented in Python by calling scipy's 'linalg.solve' function
        with 'transposed' set to 'True'.
    """
    return sp.linalg.solve(A, B, transposed=True)


def reshape(array: ArrayLike, dims: Union[int, tuple[int]]) -> NDArray:
    """Equivalent to the MATLAB 'reshape' function, whereby the elements from
    the first axis onwards are taken in some order for the reshaping (i.e. from
    axis 0 to n).

    This is different to numpy's method of taking elements from the last axis
    first, then the penultimate axis, and so on (i.e. from axis n to 0).

    PARAMETERS
    ----------
    array : array-like
    -   Array which will be reshaped.

    dims : int | tuple of int
    -   The dimensions of the reshaped array.

    RETURNS
    -------
    reshaped_array : numpy ndarray
    -   The reshaped array.

    NOTES
    -----
    -   This is equivalent to calling numpy.reshape(array, dims, order="F").
    """
    return np.reshape(array, dims, order="F")


def kron(A: ArrayLike, B: ArrayLike) -> NDArray:
    """Equivalent to the MATLAB 'kron' function, in which the Kronecker product
    is calculated.

    PARAMETERS
    ----------
    A : array-like
    -   A matrix to take the Kronecker product of with 'B'.

    B : array-like
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
        return np.kron(A, B)
    return sp.sparse.kron(A, B)
