"""Methods for computing connectivity metrics.

FUNCTIONS
---------
multivariate_interaction_measure
-   Computes the multivariate interaction measure between two groups of signals.

max_imaginary_coherence
-   Computes the maximised imaginary coherence between two groups of signals.

granger_causality
-   Computes frequency-domain Granger causality between two groups of signals.

autocov_to_gc
-   Computes frequency-domain Granger causality from an autocovariance sequence.
"""

from typing import Union
import numpy as np
from numpy.typing import ArrayLike, NDArray
from Processing.matlab_functions import reshape
from Processing.signal_processing import (
    autocov_to_full_var,
    csd_to_autocov,
    full_var_to_iss,
    iss_to_usgc,
    mim_mic_compute_e,
    cross_spectra_svd,
)


def multivariate_interaction_measure(
    csd: NDArray,
    seeds: tuple[ArrayLike],
    targets: tuple[ArrayLike],
    n_seed_components: tuple[Union[int, None]],
    n_target_components: tuple[Union[int, None]],
) -> NDArray:
    """Computes the multivariate interaction measure between two sets of
    signals.

    PARAMETERS
    ----------
    csd : numpy ndarray
    -   Cross-spectral density with dimensions [signals x signals x
        frequencies].

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

    RETURNS
    -------
    mim : numpy ndarray
    -   Array containing connectivity values for each node with dimensions
        [nodes x frequencies].

    NOTES
    -----
    -   References: [1] Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    """
    n_nodes = len(seeds)
    n_freqs = csd.shape[2]

    mim = np.zeros((n_nodes, n_freqs))
    node_i = 0
    for seed_idcs, target_idcs in zip(seeds, targets):
        node_idcs = [*seed_idcs, *target_idcs]
        node_csd = csd[np.ix_(node_idcs, node_idcs, np.arange(n_freqs))]
        for freq_i in range(n_freqs):
            # Eqs. 32 & 33
            C_bar, U_bar_aa, _ = cross_spectra_svd(
                csd=node_csd[:, :, freq_i],
                n_seeds=len(seed_idcs),
                n_seed_components=n_seed_components[node_i],
                n_target_components=n_target_components[node_i],
            )

            # Eqs. 3 & 4
            E = mim_mic_compute_e(csd=C_bar, n_seeds=U_bar_aa.shape[1])

            # Equation 14
            mim[node_i, freq_i] = np.trace(np.matmul(E, np.conj(E).T))
        node_i += 1

    return mim


def max_imaginary_coherence(
    csd: NDArray,
    seeds: tuple[ArrayLike],
    targets: tuple[ArrayLike],
    n_seed_components: tuple[Union[int, None]],
    n_target_components: tuple[Union[int, None]],
) -> NDArray:
    """Computes the maximised imaginary coherence between two groups of signals.

    PARAMETERS
    ----------
    csd : numpy ndarray
    -   Cross-spectral density with dimensions [signals x signals x
        frequencies].

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

    RETURNS
    -------
    mic : numpy ndarray
    -   Array containing connectivity values for each node with dimensions
        [nodes x frequencies].

    NOTES
    -----
    -   References: [1] Ewald et al. (2012), NeuroImage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    """
    n_nodes = len(seeds)
    n_freqs = csd.shape[2]

    mic = np.zeros((n_nodes, n_freqs))
    node_i = 0
    for seed_idcs, target_idcs in zip(seeds, targets):
        n_seeds = len(seed_idcs)
        node_idcs = [*seed_idcs, *target_idcs]
        node_csd = csd[np.ix_(node_idcs, node_idcs, np.arange(n_freqs))]
        for freq_i in range(n_freqs):
            # Eqs. 32 & 33
            C_bar, U_bar_aa, _ = cross_spectra_svd(
                csd=node_csd[:, :, freq_i],
                n_seeds=n_seeds,
                n_seed_components=n_seed_components[node_i],
                n_target_components=n_target_components[node_i],
            )

            # Eqs. 3 & 4
            E = mim_mic_compute_e(csd=C_bar, n_seeds=U_bar_aa.shape[1])

            # Weights for signals in the groups
            w_a, V_a = np.linalg.eigh(np.matmul(E, np.conj(E).T))
            w_b, V_b = np.linalg.eigh(np.matmul(np.conj(E).T, E))
            alpha = V_a[:, w_a.argmax()]
            beta = V_b[:, w_b.argmax()]

            # Eq. 7
            mic[node_i, freq_i] = (
                np.matmul(np.conj(alpha).T, np.matmul(E, beta))
                / np.linalg.norm(alpha)
                * np.linalg.norm(beta)
            )
        node_i += 1

    return mic


def granger_causality(
    csd: NDArray,
    freqs: list[Union[int, float]],
    method: str,
    seeds: tuple[ArrayLike],
    targets: tuple[ArrayLike],
    n_seed_components: tuple[Union[int, None]],
    n_target_components: tuple[Union[int, None]],
    n_lags: int,
) -> NDArray:
    """Computes frequency-domain Granger causality (GC) between two sets of
    signals.

    PAREMETERS
    ----------
    csd : numpy ndarray
    -   Matrix containing the cross-spectral density (CSD) between signals, with
        dimensions [n_signals x n_signals x n_frequencies].

    freqs : list[int | float]
    -   Frequencies in "csd".

    method : str
    -   Which form of GC to compute.
    -   Supported inputs are: "gc" for GC from seeds to targets; "net_gc" for
        net GC, i.e. GC from seeds to targets minus GC from targets to seeds;
        "trgc" for time-reversed GC (TRGC) from seeds to targets; and "net_trgc"
        for net TRGC, i.e. TRGC from seeds to targets minus TRGC from targets to
        seeds.

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
    -   Number of lags to use when computing the autocovariance sequence from
        the cross-spectra.

    RETURNS
    -------
    connectivity : numpy ndarray
    -   Granger causality values in a matrix with dimensions [nodes x
        frequencies], where the nodes correspond to seed-target pairs.

    NOTES
    -----
    -   Net TRGC is the recommended method for maximum robustness.
    -   Each group of seeds and targets cannot contain the same indices.
    """
    autocov, autocov_indices = csd_to_autocov(
        csd=csd,
        seeds=seeds,
        targets=targets,
        n_seed_components=n_seed_components,
        n_target_components=n_target_components,
        n_lags=n_lags,
    )
    connectivity = autocov_to_gc(
        autocov=autocov, freqs=freqs, method=method, indices=autocov_indices
    )

    return connectivity


def autocov_to_gc(
    autocov: list[NDArray],
    freqs: list[Union[int, float]],
    method: str,
    indices: tuple[list[list[int]]],
) -> NDArray:
    """Computes frequency-domain Granger causality from an autocovariance
    sequence.

    PARAMETERS
    ----------
    autocov : list[numpy ndarray]
    -   An autocovariance sequence for each node as arrays with dimensions
        [signals x signals x (lags + 1)]

    freqs : list[int | float]
    -   Frequencies of the data being analysed.

    method : str
    -   Which form of GC to compute.
    -   Supported inputs are: "gc" for GC from seeds to targets; "net_gc" for
        net GC, i.e. GC from seeds to targets minus GC from targets to seeds;
        "trgc" for time-reversed GC (TRGC) from seeds to targets; and "net_trgc"
        for net TRGC, i.e. TRGC from seeds to targets minus TRGC from targets to
        seeds.

    indices : tuple of list of list of int
    -   Connectivity indices for each set of seeds and targets, respectively,
        based on the position of the signals in the first two dimensions of each
        node of "autocov".

    RETURNS
    -------
    connectivity : numpy ndarray
    -   Granger causality values in a matrix with dimensions [nodes x
        frequencies], where the nodes correspond to seed-target pairs.
    """
    connectivity = np.zeros((len(autocov), len(freqs)))
    for node_i, node_autocov in enumerate(autocov):
        var_coeffs, residuals_cov = autocov_to_full_var(node_autocov)
        var_coeffs_2d = reshape(
            var_coeffs,
            (var_coeffs.shape[0], var_coeffs.shape[0] * var_coeffs.shape[2]),
        )
        A, K = full_var_to_iss(var_coeffs=var_coeffs_2d)
        connectivity[node_i, :] = iss_to_usgc(
            state_transition=A,
            observation=var_coeffs_2d,
            kalman_gain=K,
            covariance=residuals_cov,
            freqs=freqs,
            seeds=indices[0][node_i],
            targets=indices[1][node_i],
        )

        if "net" in method:
            connectivity[node_i, :] -= iss_to_usgc(
                state_transition=A,
                observation=var_coeffs_2d,
                kalman_gain=K,
                covariance=residuals_cov,
                freqs=freqs,
                seeds=indices[1][node_i],
                targets=indices[0][node_i],
            )

    if "trgc" in method:
        if method == "trgc":
            tr_method = "gc"
        elif method == "net_trgc":
            tr_method = "net_gc"
        connectivity -= autocov_to_gc(
            autocov=[
                np.transpose(node_autocov, (1, 0, 2))
                for node_autocov in autocov
            ],
            freqs=freqs,
            method=tr_method,
            indices=indices,
        )

    return np.asarray(connectivity)
