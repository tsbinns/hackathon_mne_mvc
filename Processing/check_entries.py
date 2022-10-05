"""Functions for checking the entries/values of objects.

FUNCTIONS
---------
check_posdef
-   Checks whether a matrix is positive-definite.

check_svd_params
-   Checks that the parameters used for a singular value decomposition (SVD) are
    compatible with the data being used.
"""

import numpy as np
from numpy.typing import ArrayLike


def check_posdef(A: ArrayLike) -> bool:
    """Checks whether a matrix is positive-definite.

    PARAMETERS
    ----------
    A : array-like
    -   The matrix to check the positive-definite nature of.

    RETURNS
    -------
    is_posdef : bool
    -   Whether or not the matrix is positive-definite.

    NOTES
    -----
    -   First checks if the matrix is symmetric, and then performs a Cholesky
        decomposition.
    -   If the matrix is not symmetric, the positive-definite nature is
        determined to be false.
    -   If the matrix is symmetric and the Cholesky decomposition fails, the
        positive-definite nature is determined to be false, otherwise the matrix
        is said to be positive-definite.
    """
    is_posdef = True
    if np.allclose(A, A.T):  # For differences due to floating-point errors
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            is_posdef = False
    else:
        is_posdef = False

    return is_posdef


def check_svd_params(n_signals: int, take_n_components: int) -> None:
    """Checks that the parameters used for a singular value decomposition (SVD)
    are compatible with the data being used.

    PARAMETERS
    ----------
    n_signals : int
    -   The number of signals in the data the SVD is being performed on. This is
        the maximum number of components that can be taken from the SVD.

    take_n_components : int
    -   The number of components being taken from the SVD.

    RAISES
    ------
    ValueError
    -   Raised if 0 components are being taken from the SVD, or the number of
        components being taken are greater than the number of signals (i.e. the
        maximum number of components available).
    """
    if take_n_components == 0:
        raise ValueError(
            "0 components are being taken from the singular value "
            "decomposition, but this must be at least 1."
        )
    if take_n_components > n_signals:
        raise ValueError(
            f"At most {n_signals} components can be taken from the singular "
            f"value decomposition, but {take_n_components} are being taken."
        )
