"""Functions for performing signal processing computations.

FUNCTIONS
---------
csd_to_autocov
-   Computes the autocovariance sequence from the cross-spectral density.

autocov_to_full_var
-   Computes the full vector autoregressive (VAR) model from an autocovariance
    sequence using Whittle's recursion.

whittle_lwr_recursion
-   Calculates regression coefficients and the residuals' covariance matrix from
    an autocovariance sequence by solving the Yule-Walker equations using
    Whittle's recursive Levinson, Wiggins, Robinson (LWR) algorithm.

full_var_to_iss
-   Computes innovations-form parameters for a state-space model from a full
    vector autoregressive (VAR) model using Aoki's method.

iss_to_usgc
-   Computes unconditional spectral Granger causality from innovations-form
    state-space model parameters.

iss_to_tf
-   Computes a transfer function (moving-average representation) for
    innovations-form state-space model parameters.

partial_covariance
-   Computes the partial covariance for use in spectral Granger causality (GC)
    calculations.

block_ifft
-   Performs a 'block' inverse fast Fourier transform on the data, involving an
    n-point inverse Fourier transform.

discrete_lyapunov
-   Solves the discrete-time Lyapunov equation via Schur decomposition with a
    column-by-column solution.

cross_spectra_svd
-   Performs dimensionality reduction on a cross-spectral density using
    singular value decomposition (SVD), if requested.

mim_mic_compute_e
-   Computes E as the imaginary part of the transformed connectivity matrix D
    derived from the original cross-spectral density between the seed and target
    signals.
"""

from typing import Union
import numpy as np
from scipy import linalg as spla
from numpy.typing import ArrayLike, NDArray
from Processing.check_entries import check_posdef, check_svd_params
from Processing.matlab_functions import reshape, kron, linsolve_transa


def csd_to_autocov(
    csd: NDArray,
    seeds: tuple[ArrayLike],
    targets: tuple[ArrayLike],
    n_seed_components: tuple[Union[int, None]],
    n_target_components: tuple[Union[int, None]],
    n_lags: int,
) -> tuple[list[NDArray], tuple[list[list[int]]]]:
    """Computes the autocovariance sequence from the cross-spectral density.

    PARAMETERS
    ----------
    csd : numpy array
    -   Three-dimensional matrix of the cross-spectral density.
    -   Expects a matrix with dimensions [signals x signals x frequencies],
        where the third dimension corresponds to different frequencies.

    seeds : tuple of array-like of int
    -   Indices of signals in "csd" to treat as seeds. Should be a tuple of
        arrays, where each array contains the signal indices that will be
        treated as a group of seeds.
    -   The number of sublists must match the number of sublists in "targets".

    targets : tuple of array-like of int
    -   Indices of signals in "csd" to treat as targets. Should be a tuple of
        arrays, where each array contains the signal indices that will be
        treated as a group of targets.
    -   The number of sublists must match the number of sublists in "seeds".

    n_seed_components : tuple of int or None
    -   The number of components that should be taken from the seed data after
        dimensionality reduction using singular value decomposition (SVD), with
        one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the seed data for
        any of the connectivity nodes.
    -   If some values in the tuple are 'None', no dimensionality reduction is
        performed on the seed data for the corresponding connectivity node.

    n_target_components : tuple of int or None
    -   The number of components that should be taken from the target data after
        dimensionality reduction using singular value decomposition (SVD), with
        one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the target data
        for any of the connectivity nodes.
    -   If some values in the tuple are 'None', no dimensionality reduction is
        performed on the target data for the corresponding connectivity node.

    n_lags : int
    -   Number of autocovariance lags to calculate.

    RETURNS
    -------
    autocov : list of numpy ndarray
    -   The computed autocovariance sequence for each node as arrays with
        dimensions [signals x signals x (lags + 1)]

    autocov_indices : tuple of list of list of int
    -   Indices of the seeds and targets in each node of "autocov".

    RAISES
    ------
    ValueError
    -   Raised if "n_lags" is greater than (n_freqs - 1) * 2.
    """
    n_freqs = csd.shape[2]
    freq_res = n_freqs - 1
    if n_lags > freq_res * 2:
        raise ValueError(
            f"The number of lags ({n_lags}) cannot be greater than the "
            "frequency resolution of the cross-spectral density "
            f"({freq_res})."
        )

    autocov = []
    autocov_indices = [[], []]
    node_i = 0
    for node_seeds, node_targets in zip(seeds, targets):
        node_idcs = [*node_seeds, *node_targets]
        node_csd = csd[np.ix_(node_idcs, node_idcs, np.arange(n_freqs))]

        node_csd_bar = []
        for freq_i in range(n_freqs):
            C_bar, U_bar_aa, _ = cross_spectra_svd(
                csd=node_csd[:, :, freq_i],
                n_seeds=len(node_seeds),
                n_seed_components=n_seed_components[node_i],
                n_target_components=n_target_components[node_i],
            )
            node_csd_bar.append(C_bar)
        node_csd = np.transpose(np.asarray(node_csd_bar), (1, 2, 0))
        autocov_indices[0].append(np.arange(U_bar_aa.shape[1]).tolist())
        autocov_indices[1].append(
            np.arange(start=U_bar_aa.shape[1], stop=node_csd.shape[0]).tolist()
        )

        circular_shifted_csd = np.concatenate(
            [np.flip(np.conj(node_csd[:, :, 1:]), axis=2), node_csd[:, :, :-1]],
            axis=2,
        )
        ifft_shifted_csd = block_ifft(
            data=circular_shifted_csd, n_points=freq_res * 2
        )

        lags_ifft_shifted_csd = reshape(
            ifft_shifted_csd[:, :, : n_lags + 1],
            (node_csd.shape[0] ** 2, n_lags + 1),
        )
        signs = [1] * (n_lags + 1)
        signs[1::2] = [x * -1 for x in signs[1::2]]
        sign_matrix = np.tile(np.asarray(signs), (node_csd.shape[0] ** 2, 1))

        autocov.append(
            np.real(
                reshape(
                    sign_matrix * lags_ifft_shifted_csd,
                    (node_csd.shape[0], node_csd.shape[0], n_lags + 1),
                )
            )
        )
        node_i += 1

    return autocov, autocov_indices


def autocov_to_full_var(autocov: NDArray) -> tuple[NDArray, NDArray]:
    """Computes the full vector autoregressive (VAR) model from an
    autocovariance sequence using Whittle's recursion.

    PARAMETERS
    ----------
    autocov : numpy ndarray
    -   An autocovariance sequence with dimensions [signals x signals x (lags +
        1)].

    RETURNS
    -------
    var_coeffs : numpy ndarray
    -   The coefficients of the full forward VAR model with dimensions
        [signals x signals x lags].

    residuals_cov : numpy ndarray
    -   The residuals' covariance matrix with dimensions [signals x signals].

    RAISES
    ------
    -   Raised if the residuals' covariance matrix is not positive-definite.

    NOTES
    -----
    -   For Whittle's recursion algorithm, see: Whittle P., 1963. Biometrika,
        doi: 10.1093/biomet/50.1-2.129.
    -   Additionally checks that the coefficients are all 'good', i.e. that all
        values are neither 'NaN' nor 'Inf'.
    -   If an error is raised over the non-positive-definite nature of the
        residuals' covariance matrix, try using only data that has full rank.
    """
    AF, V = whittle_lwr_recursion(autocov=autocov, enforce_coeffs_good=True)

    try:
        np.linalg.cholesky(V)
    except np.linalg.linalg.LinAlgError as np_error:
        raise ValueError(
            "The residuals' covariance matrix is not positive-definite. Try "
            "using only full-rank data when creating the autocovariance "
            "sequence."
        ) from np_error

    return AF, V


def whittle_lwr_recursion(
    autocov: NDArray, enforce_coeffs_good: bool = True
) -> tuple[NDArray, NDArray]:
    """Calculates regression coefficients and the residuals' covariance matrix
    from an autocovariance sequence by solving the Yule-Walker equations using
    Whittle's recursive Levinson, Wiggins, Robinson (LWR) algorithm.

    PARAMETERS
    ----------
    autocov : numpy ndarray
    -   The autocovariance sequence with dimensions [signals x signals x (lags +
        1)].

    enforce_coeffs_good : bool; default True
    -   Checks that the coefficients of the VAR model are all 'good', i.e. that
        they are all neither 'NaN' or 'Inf', which can happen if the regressions
        are rank-deficient or ill-conditioned.

    RETURNS
    -------
    var_coeffs : numpy ndarray
    -   The coefficients of the full forward VAR model with dimensions
        [signals x signals x lags].

    residuals_cov : numpy ndarray
    -   The residuals' covariance matrix with dimensions [signals x signals].

    RAISES
    ------
    ValueError
    -   Raised if 'enforce_coeffs_good' is 'True' and the VAR model coefficients
        are not all neither 'NaN' or 'Inf'.

    NOTES
    -----
    -   For Whittle's recursion algorithm, see: Whittle P., 1963, Biometrika,
        DOI: 10.1093/biomet/50.1-2.129.
    """
    G = autocov
    ### Initialise recursion
    n = G.shape[0]  # number of signals
    q = G.shape[2] - 1  # number of lags
    qn = n * q

    G0 = G[:, :, 0]  # covariance
    GF = (
        reshape(G[:, :, 1:], (n, qn)).conj().T
    )  # forward autocovariance sequence
    GB = reshape(
        np.flip(G[:, :, 1:], 2).transpose((0, 2, 1)), (qn, n)
    )  # backward autocovariance sequence

    AF = np.zeros((n, qn))  # forward coefficients
    AB = np.zeros((n, qn))  # backward coefficients

    k = 1  # model order
    r = q - k
    kf = np.arange(k * n)  # forward indices
    kb = np.arange(r * n, qn)  # backward indices

    # equivalent to calling A/B or linsolve(B',A',opts.TRANSA=true)' in MATLAB
    AF[:, kf] = linsolve_transa(G0.conj().T, GB[kb, :].conj().T).conj().T
    AB[:, kb] = linsolve_transa(G0.conj().T, GF[kf, :].conj().T).conj().T

    ### Recursion
    for k in np.arange(2, q + 1):
        # equivalent to calling A/B or linsolve(B,A',opts.TRANSA=true)' in
        # MATLAB
        var_A = GB[(r - 1) * n : r * n, :] - np.matmul(AF[:, kf], GB[kb, :])
        var_B = G0 - np.matmul(AB[:, kb], GB[kb, :])
        AAF = linsolve_transa(var_B, var_A.conj().T).conj().T
        var_A = GF[(k - 1) * n : k * n, :] - np.matmul(AB[:, kb], GF[kf, :])
        var_B = G0 - np.matmul(AF[:, kf], GF[kf, :])
        AAB = linsolve_transa(var_B, var_A.conj().T).conj().T

        AF_previous = AF[:, kf]
        AB_previous = AB[:, kb]

        r = q - k
        kf = np.arange(k * n)
        kb = np.arange(r * n, qn)

        AF[:, kf] = np.hstack((AF_previous - np.matmul(AAF, AB_previous), AAF))
        AB[:, kb] = np.hstack((AAB, AB_previous - np.matmul(AAB, AF_previous)))

    V = G0 - np.matmul(AF, GF)
    AF = reshape(AF, (n, n, q))

    if enforce_coeffs_good:
        if not np.isfinite(AF).all():
            raise ValueError(
                "The 'good' (i.e. non-NaN and non-infinite) nature of the VAR "
                "model coefficients is being enforced, but the coefficients "
                "are not all finite."
            )

    return AF, V


def full_var_to_iss(var_coeffs: NDArray) -> tuple[NDArray, NDArray]:
    """Computes innovations-form parameters for a state-space model from a full
    vector autoregressive (VAR) model using Aoki's method.

    For a non-moving-average full VAR model, the state-space parameters C (the
    observation matrix) and V (the innovations covariance matrix) are identical
    to AF and V of the VAR model, respectively.

    PARAMETERS
    ----------
    var_coeffs : numpy ndarray
    -   The coefficients of the full VAR model with dimensions [signals x
        (signals * lags)].

    RETURNS
    -------
    state_transition : numpy ndarray
    -   State transition matrix with dimensions [(signals * lags) x (signals *
        lags)]

    kalman_gain : numpy ndarray
    -   Kalman gain matix with dimensions [(signals * lags) x (signals)]

    NOTES
    -----
    -   Reference(s): [1] Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   Aoki's method for computing innovations-form parameters for a
        state-space model allows for zero-lag coefficients.
    """
    m = var_coeffs.shape[0]  # number of signals
    p = var_coeffs.shape[1] // m  # number of autoregressive lags

    Ip = np.eye(m * p)
    A = np.vstack(
        (var_coeffs, Ip[: (len(Ip) - m), :])
    )  # state transition matrix
    K = np.vstack(
        (np.eye(m), np.zeros(((m * (p - 1)), m)))
    )  # Kalman gain matrix

    return A, K


def iss_to_usgc(
    state_transition: NDArray,
    observation: NDArray,
    kalman_gain: NDArray,
    covariance: NDArray,
    freqs: list[Union[int, float]],
    seeds: ArrayLike,
    targets: ArrayLike,
) -> NDArray:
    """Computes unconditional spectral Granger causality from innovations-form
    state-space model parameters.

    PARAMETERS
    ----------
    state_transition : numpy ndarray
    -   State transition matrix with dimensions [(signals * lags) x (signals *
        lags)].

    observation : numpy ndarray
    -   Observation matrix with dimensions [signals x (signals * lags)].

    kalman_gain : numpy ndarray
    -   Kalman gain matrix with dimensions [(signals * lags) x signals].

    covariance : numpy ndarray
    -   Innovations covariance matrix with dimensions [signals x signals].

    freqs : list of int or float
    -   Frequencies of connectivity being analysed.

    seeds : array-like of int
    -   Seed indices. Cannot contain indices also in "targets".

    targets : array-like of int
    -   Target indices. Cannot contain indices also in "seeds".

    RETURNS
    -------
    f : numpy ndarray
    -   Spectral Granger causality from the seeds to the targets for each
        frequency.

    NOTES
    -----
    -   Reference(s): [1] Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    """

    f = np.zeros(len(freqs))
    z = np.exp(
        -1j * np.pi * np.linspace(0, 1, len(freqs))
    )  # points on a unit circle in the complex plane, one for each frequency
    H = iss_to_tf(
        state_transition, observation, kalman_gain, z
    )  # spectral transfer function
    VSQRT = np.linalg.cholesky(covariance)
    PVSQRT = np.linalg.cholesky(partial_covariance(covariance, seeds, targets))

    for freq_i in range(len(freqs)):
        HV = np.matmul(H[:, :, freq_i], VSQRT)
        S = np.matmul(
            HV, HV.conj().T
        )  # CSD of the projected state variable (Eq. 6)
        S_tt = S[np.ix_(targets, targets)]  # CSD between targets
        if len(PVSQRT) == 1:
            HV_ts = H[targets, seeds, freq_i] * PVSQRT
            HVH_ts = np.outer(HV_ts, HV_ts.conj().T)
        else:
            HV_ts = np.matmul(H[np.ix_(targets, seeds)][:, :, freq_i], PVSQRT)
            HVH_ts = np.matmul(HV_ts, HV_ts.conj().T)
        if len(targets) == 1:
            numerator = np.real(S_tt)
            denominator = np.real(S_tt - HVH_ts)
        else:
            numerator = np.real(np.linalg.det(S_tt))
            denominator = np.real(np.linalg.det(S_tt - HVH_ts))
        f[freq_i] = np.log(numerator) - np.log(denominator)  # Eq. 11

    return f


def iss_to_tf(
    state_transition: NDArray,
    observation: NDArray,
    kalman_gain: NDArray,
    back_shift: NDArray,
) -> NDArray:
    """Computes a transfer function (moving-average representation) for
    innovations-form state-space model parameters.

    PARAMETERS
    ----------
    state_transition : numpy ndarray
    -   State transition matrix with dimensions [(signals * lags) x (signals *
        lags)].

    observation : numpy ndarray
    -   Observation matrix with dimensions [signals x (signals * lags)].

    kalman_gain : numpy ndarray
    -   Kalman gain matrix with dimensions [(signals * lags) x signals].

    back_shift : numpy ndarray
    -   The back-shift operator with length equal to the number of frequencies.

    RETURNS
    -------
    transfer_function : numpy ndarray
    -   The transfer function with dimensions [signals x signals x frequencies].

    NOTES
    -----
    -   Reference: [1] Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   In the frequency domain, the back-shift operator, z, is a vector of
        points on a unit circle in the complex plane. z = e^-iw, where -pi < w
        <= pi. See Ref. 17 of [1].
    """
    A = state_transition
    C = observation
    K = kalman_gain
    z = back_shift

    h = len(z)
    n = C.shape[0]
    m = A.shape[0]
    I_n = np.eye(n)
    I_m = np.eye(m)
    H = np.zeros((n, n, h), dtype=complex)

    for k in range(h):
        H[:, :, k] = I_n + np.matmul(
            C, spla.lu_solve(spla.lu_factor(z[k] * I_m - A), K)  # Eq. 4
        )

    return H


def partial_covariance(
    covariance: NDArray, seeds: ArrayLike, targets: ArrayLike
) -> NDArray:
    """Computes the partial covariance for use in spectral Granger causality
    (GC) calculations.

    PARAMETERS
    ----------
    covariance : numpy ndarray
    -   A positive-definite, symmetric innovations covariance matrix.

    seeds : array-like of int
    -   Indices of entries in "covariance" that are seeds in the GC calculation.

    targets : array-like of int
    -   Indices of entries in "covariance" that are targets in the GC
        calculation.

    RETURNS
    -------
    partial_covariance : numpy ndarray
    -   The partial covariance matrix between the targets given the seeds.

    RAISES
    ------
    ValueError
    -   Raised if "covariance" is not a symmetric, positive-definite matrix.
    -   Raised if "seeds" and "targets" contain common indices.

    NOTES
    -----
    -   Reference: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
    -   Given a covariance matrix V, the partial covariance matrix of V between
        indices i and j, given k (V_ij|k), is equivalent to
        V_ij - V_ik * V_kk^-1 * V_kj. In this case, i and j are seeds, and k is
        the targets.
    """
    V = covariance

    if not check_posdef(V):
        raise ValueError(
            "'covariance' must be a positive-definite, symmetric matrix, but "
            "it is not."
        )
    common_idcs = set.intersection(set(seeds), set(targets))
    if common_idcs:
        raise ValueError(
            "There are common indices present in both sets of indices, but "
            f"this is not allowed.\n- Common indices: {common_idcs}"
        )

    if len(targets) == 1:
        W = (1 / np.sqrt(V[targets, targets])) * V[targets, seeds]
        W = np.outer(W.conj().T, W)
    else:
        W = np.linalg.solve(
            np.linalg.cholesky(V[np.ix_(targets, targets)]),
            V[np.ix_(targets, seeds)],
        )
        W = W.conj().T.dot(W)

    return V[np.ix_(seeds, seeds)] - W


def block_ifft(data: NDArray, n_points: int) -> NDArray:
    """Performs a 'block' inverse fast Fourier transform on the data, involving
    an n-point inverse Fourier transform.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   A three-dimensional matrix on which the inverse Fourier transform will
        be conducted, where the third dimension is assumed to correspond to
        different frequencies.

    n_points : int | None; default None
    -   The number of points to use for the inverse Fourier transform.

    RETURNS
    -------
    transformed_data : numpy ndarray
    -   A three-dimensional matrix of the transformed data.
    """
    data_shape = data.shape
    two_dim_data = reshape(
        data, (data_shape[0] * data_shape[1], data_shape[2])
    ).T
    ifft_data = np.fft.ifft(two_dim_data, n=n_points, axis=0).T

    return reshape(ifft_data, (data_shape[0], data_shape[1], data_shape[2]))


def discrete_lyapunov(A: NDArray, Q: NDArray) -> NDArray:
    """Solves the discrete-time Lyapunov equation via Schur decomposition with a
    column-by-column solution.

    PARAMETERS
    ----------
    A : numpy ndarray
    -   A square matrix with a spectral radius of < 1.

    Q : numpy ndarray
    -   A symmetric, positive-definite matrix with the same dimensions as "A".

    RETURNS
    -------
    X : numpy ndarray
    -   The solution of the discrete-time Lyapunov equation.

    NOTES
    -----
    -   The Lyapunov equation takes the form X = A*X*conj(A)'+Q
    -   References: [1] Kitagawa G., 1977, International Journal of Control,
        DOI: 10.1080/00207177708922266; [2] Hammarling S.J., 1982, IMA Journal
        of Numerical Analysis, DOI: 10.1093/imanum/2.3.303.
    """
    n = A.shape[0]
    T, U = spla.schur(A)
    Q = np.matmul(np.matmul(-U.conj().T, Q), U)

    # Solve the equation column-by-column
    X = np.zeros((n, n))
    j = n - 1
    while j > 0:
        j1 = j + 1

        # Check Schur block size
        if T[j, j - 1] != 0:
            bsiz = 2
            j = j - 1
        else:
            bsiz = 1
        bsizn = bsiz * n

        Ajj = kron(T[j:j1, j:j1], T) - np.eye(bsizn)
        rhs = reshape(Q[:, j:j1], (bsizn, 1))

        if j1 < n:
            rhs = rhs + reshape(
                np.matmul(
                    T,
                    np.matmul(X[:, j1:n], T[j:j1, j1:n].conj().T),
                ),
                (bsizn, 1),
            )

        v = spla.lu_solve(spla.lu_factor(-Ajj), rhs)
        X[:, j] = v[:n].flatten()

        if bsiz == 2:
            X[:, j1 - 1] = v[n:bsizn].flatten()

        j = j - 1

    return np.matmul(
        U, np.matmul(X, U.conj().T)
    )  # Convert back to original coordinates


def cross_spectra_svd(
    csd: NDArray,
    n_seeds: int,
    n_seed_components: Union[int, None] = None,
    n_target_components: Union[int, None] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Performs dimensionality reduction on a cross-spectral density using
    singular value decomposition (SVD), if requested.

    PARAMETERS
    ----------
    csd : numpy ndarray
    -   A 2D array of cross-spectral values between all possible connections of
        seeds and targets, for a single frequency. Has the dimensions [signals x
        signals], where n_signals = n_seeds + n_targets.

    n_seeds : int
    -   Number of seed signals. Entries in both dimensions of "csd" from [0 :
        n_seeds] are taken as the coherency values for seed signals. Entries
        from [n_seeds : end] are taken as the coherency values for target
        signals.

    n_seed_components : int | None; default None
    -   The number of components to take from the SVD of the seed data.
    -   If 'None', no SVD is performed on the seed data.

    n_target_components : int | None; default None
    -   The number of components to take from the SVD of the target data.
    -   If 'None', no SVD is performed on the target data.

    RETURNS
    -------
    C_bar : numpy ndarray
    -   The transformed cross-spectral values with dimensions [signals x
        signals], where n_signals = n_seed_components + n_target_components.
    -   If both "n_seed_components" and "n_target_components" are None, the
        original cross-spectra is returned.

    U_bar_aa : numpy ndarray
    -   The real part of the matrix U from the SVD on the seed data with
        dimensions [seeds x seed_components].
    -   If "n_seed_components" is None, an identity matrix for a matrix of
        size [seeds x seeds] is returned.

    U_bar_bb : numpy ndarray
    -   The real part of the matrix U from the SVD on the target data with
        dimensions [targets x target_components].
    -   If "n_target_components" is None, an identity matrix for a matrix of
        size [targets x targets] is returned.
    """
    C_aa = csd[:n_seeds, :n_seeds]
    C_ab = csd[:n_seeds, n_seeds:]
    C_bb = csd[n_seeds:, n_seeds:]
    C_ba = csd[n_seeds:, :n_seeds]

    # Eq. 32
    if n_seed_components is not None:
        check_svd_params(n_signals=n_seeds, take_n_components=n_seed_components)
        U_aa, _, _ = np.linalg.svd(np.real(C_aa), full_matrices=False)
        U_bar_aa = U_aa[:, :n_seed_components]
    else:
        U_bar_aa = np.identity(C_aa.shape[0])
    if n_target_components is not None:
        check_svd_params(
            n_signals=csd.shape[0] - n_seeds,
            take_n_components=n_target_components,
        )
        U_bb, _, _ = np.linalg.svd(np.real(C_bb), full_matrices=False)
        U_bar_bb = U_bb[:, :n_target_components]
    else:
        U_bar_bb = np.identity(C_bb.shape[0])

    # Eq. 33
    C_bar_aa = np.matmul(U_bar_aa.T, np.matmul(C_aa, U_bar_aa))
    C_bar_ab = np.matmul(U_bar_aa.T, np.matmul(C_ab, U_bar_bb))
    C_bar_bb = np.matmul(U_bar_bb.T, np.matmul(C_bb, U_bar_bb))
    C_bar_ba = np.matmul(U_bar_bb.T, np.matmul(C_ba, U_bar_aa))
    C_bar = np.vstack(
        (np.hstack((C_bar_aa, C_bar_ab)), np.hstack((C_bar_ba, C_bar_bb)))
    )

    return C_bar, U_bar_aa, U_bar_bb


def mim_mic_compute_e(csd: NDArray, n_seeds: int) -> NDArray:
    """Computes E as the imaginary part of the transformed cross-spectra D
    derived from the original cross-spectra "csd" between the seed and target
    signals.

    Designed for use with the functions "max_imaginary_coherence" and
    "multivariate_interaction_measure".

    PARAMETERS
    ----------
    csd : numpy ndarray
    -   Cross-spectral density between all possible connections of seeds and
        targets, for a single frequency. Has the dimensions [signals x signals].

    n_seeds : int
    -   Number of seed signals. Entries in both dimensions of "csd" from
        [0 : n_seeds] are taken as the cross-spectral values for seed signals.
        Entries from [n_seeds : end] are taken as the cross-spcetral values for
        target signals.

    RETURNS
    -------
    transformed_csd : numpy ndarray
    -   E, the imaginary part of the transformed cross-spectra D between seed
        and target signals.

    NOTES
    -----
    -   References: [1] Ewald et al., 2012, NeuroImage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    """
    # Equation 3
    T = np.zeros(csd.shape)
    T[:n_seeds, :n_seeds] = spla.fractional_matrix_power(
        np.real(csd[:n_seeds, :n_seeds]), -0.5
    )  # real(C_aa)^-1/2
    T[n_seeds:, n_seeds:] = spla.fractional_matrix_power(
        np.real(csd[n_seeds:, n_seeds:]), -0.5
    )  # real(C_bb)^-1/2

    # Equation 4
    D = np.matmul(T, np.matmul(csd, T))

    # E as the imaginary part of D between seeds and targets
    E = np.imag(D[:n_seeds, n_seeds:])

    return E
