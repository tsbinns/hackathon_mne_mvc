"""Methods for computing connectivity metrics.

METHODS
-------
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
from numpy.typing import NDArray
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
    seeds: list[list[Union[int, float]]],
    targets: list[list[Union[int, float]]],
    n_seed_components: Union[list[Union[int, None]], None] = None,
    n_target_components: Union[list[Union[int, None]], None] = None,
) -> NDArray:
    """Computes the multivariate interaction measure between two sets of
    signals.

    PARAMETERS
    ----------
    csd : numpy ndarray
    -   Cross-spectral density with dimensions [signals x signals x
        frequencies].

    seeds : list[list[int]]
    -   Indices of signals in 'csd' to treat as seeds. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of seeds.
    -   The number of sublists must match the number of sublists in 'targets'.

    targets : list[list[int]]
    -   Indices of signals in 'csd' to treat as targets. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of targets.
    -   The number of sublists must match the number of sublists in 'seeds'.

    n_seed_components : list[int | None] | None; default None
    -   The number of components that should be taken from the seed data after
        dimensionality reduction using singular value decomposition (SVD), with
        one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the seed data for
        any of the connectivity nodes.
    -   If some values in the list are 'None', no dimensionality reduction is
        performed on the seed data for the corresponding connectivity node.

    n_target_components : list[int | None] | None; default None
    -   The number of components that should be taken from the target data after
        dimensionality reduction using singular value decomposition (SVD), with
        one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the target data
        for any of the connectivity nodes.
    -   If some values in the list are 'None', no dimensionality reduction is
        performed on the target data for the corresponding connectivity node.

    RETURNS
    -------
    mim : numpy ndarray
    -   Array containing connectivity values for each node with dimensions
        [nodes x frequencies].

    RAISES
    ------
    ValueError
    -   Raised if 'C' is not three-dimensional.
    -   Raised if the first two dimensions of 'C' are not identical.

    NOTES
    -----
    -   References: [1] Ewald et al., 2012, Neuroimage. DOI:
        10.1016/j.neuroimage.2011.11.084.
    -   Based on the 'compute_mim' MATLAB function.
    """
    if len(csd.shape) != 3:
        raise ValueError(
            "The cross-spectral density must have three dimensions, but "
            f"has {len(csd.shape)}."
        )
    if csd.shape[0] != csd.shape[1]:
        raise ValueError(
            "The cross-spectral density must have the same first two "
            f"dimensions, but these are {csd.shape[0]} and {csd.shape[1]}, "
            "respectively."
        )
    if len(seeds) != len(targets):
        raise ValueError(
            f"The number of seed and target groups ({len(seeds)} and "
            f"{len(targets)}, respectively) do not match."
        )
    n_nodes = len(seeds)
    if n_seed_components is None:
        n_seed_components = [None] * n_nodes
    if n_target_components is None:
        n_target_components = [None] * n_nodes

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
            E = mim_mic_compute_e(C=C_bar, n_seeds=U_bar_aa.shape[1])

            # Equation 14
            mim[node_i, freq_i] = np.trace(np.matmul(E, np.conj(E).T))
        node_i += 1

    return mim


def max_imaginary_coherence(
    csd: NDArray,
    seeds: list[list[Union[int, float]]],
    targets: list[list[Union[int, float]]],
    n_seed_components: Union[list[Union[int, None]], None] = None,
    n_target_components: Union[list[Union[int, None]], None] = None,
    return_topographies: bool = True,
) -> Union[NDArray, tuple[NDArray]]:
    """Computes the maximised imaginary coherence between two groups of signals.

    PARAMETERS
    ----------
    csd : numpy ndarray
    -   Cross-spectral density with dimensions [signals x signals x
        frequencies].

    seeds : list[list[int]]
    -   Indices of signals in 'csd' to treat as seeds. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of seeds.
    -   The number of sublists must match the number of sublists in 'targets'.

    targets : list[list[int]]
    -   Indices of signals in 'csd' to treat as targets. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of targets.
    -   The number of sublists must match the number of sublists in 'seeds'.

    n_seed_components : list[int | None] | None; default None
    -   The number of components that should be taken from the seed data after
        dimensionality reduction using singular value decomposition (SVD), with
        one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the seed data for
        any of the connectivity nodes.
    -   If some values in the list are 'None', no dimensionality reduction is
        performed on the seed data for the corresponding connectivity node.

    n_target_components : list[int | None] | None; default None
    -   The number of components that should be taken from the target data after
        dimensionality reduction using singular value decomposition (SVD), with
        one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the target data
        for any of the connectivity nodes.
    -   If some values in the list are 'None', no dimensionality reduction is
        performed on the target data for the corresponding connectivity node.

    return_topographies : bool; default True
    -   Whether or not to return spatial topographies of connectivity for the
        signals.

    RETURNS
    -------
    mic : numpy ndarray
    -   Array containing connectivity values for each node with dimensions
        [nodes x frequencies].

    topographies : tuple(numpy object array, numpy object array)
    -   Spatial topographies of connectivity for seeds and targets,
        respectively. The entries for seeds and targets have dimensions [nodes x
        signals x frequencies], where signals correspond to the number of seed
        and target signals in each node, respectively.
    -   Returned only if 'return_topographies' is 'True'.

    RAISES
    ------
    ValueError
    -   Raised if the 'C is not a three-dimensional array.
    -   Raised if the first two dimensions of 'C' is not a square matrix with
        lengths equal to the combined number of seed and target signals.

    NOTES
    -----
    -   References: [1] Ewald et al. (2012), NeuroImage. DOI:
        10.1016/j.neuroimage.2011.11.084; [2] Nikulin et al. (2011), NeuroImage,
        DOI: 10.1016/j.neuroimage.2011.01.057.
    -   Spatial topographies are computed using the weight vectors alpha and
        beta (see [1]) by multiplying the real part of 'C' by the weight
        vectors, as in Eq. 20 of [2]. If dimensionality reduction is performed,
        weight vectors are recovered in the original sensor space using Eqs. 46
        & 47 of [1].
    -   Based on the 'compute_mic' MATLAB function.
    """
    if len(csd.shape) != 3:
        raise ValueError(
            "The cross-spectral density must have three dimensions, but "
            f"has {len(csd.shape)}."
        )
    if csd.shape[0] != csd.shape[1]:
        raise ValueError(
            "The cross-spectral density must have the same first two "
            f"dimensions, but these are {csd.shape[0]} and {csd.shape[1]}, "
            "respectively."
        )
    if len(seeds) != len(targets):
        raise ValueError(
            f"The number of seed and target groups ({len(seeds)} and "
            f"{len(targets)}, respectively) do not match."
        )
    n_nodes = len(seeds)
    if n_seed_components is None:
        n_seed_components = [None] * n_nodes
    if n_target_components is None:
        n_target_components = [None] * n_nodes

    n_freqs = csd.shape[2]
    mic = np.zeros((n_nodes, n_freqs))
    topographies_a = []
    topographies_b = []
    node_i = 0
    for seed_idcs, target_idcs in zip(seeds, targets):
        topographies_a.append([])
        topographies_b.append([])
        n_seeds = len(seed_idcs)
        node_idcs = [*seed_idcs, *target_idcs]
        node_csd = csd[np.ix_(node_idcs, node_idcs, np.arange(n_freqs))]
        for freq_i in range(n_freqs):
            # Eqs. 32 & 33
            C_bar, U_bar_aa, U_bar_bb = cross_spectra_svd(
                csd=node_csd[:, :, freq_i],
                n_seeds=n_seeds,
                n_seed_components=n_seed_components[node_i],
                n_target_components=n_target_components[node_i],
            )

            # Eqs. 3 & 4
            E = mim_mic_compute_e(C=C_bar, n_seeds=U_bar_aa.shape[1])

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

            # Eqs. 46 & 47
            if return_topographies:
                topographies_a[node_i].append(
                    np.matmul(
                        np.real(node_csd[:n_seeds, :n_seeds, freq_i]),
                        U_bar_aa.dot(alpha),
                    ),
                )  # C_aa * U_bar_aa * alpha
                topographies_b[node_i].append(
                    np.matmul(
                        np.real(node_csd[n_seeds:, n_seeds:, freq_i]),
                        U_bar_bb.dot(beta),
                    ),
                )  # C_bb * U_bar_bb * beta
        if return_topographies:
            topographies_a[node_i] = np.transpose(
                topographies_a[node_i], (1, 0)
            )  # [signals x frequencies]
            topographies_b[node_i] = np.transpose(
                topographies_b[node_i], (1, 0)
            )  # [signals x frequencies]

        node_i += 1

    if return_topographies:
        topographies_a = np.asarray(topographies_a, dtype=object)
        topographies_b = np.asarray(topographies_b, dtype=object)
        return mic, (topographies_a, topographies_b)
    else:
        return mic


def granger_causality(
    csd: NDArray,
    freqs: list[Union[int, float]],
    method: str,
    seeds: list[list[int]],
    targets: list[list[int]],
    n_seed_components: Union[list[int], None] = None,
    n_target_components: Union[list[int], None] = None,
    n_lags: int = 20,
) -> NDArray:
    """Computes frequency-domain Granger causality (GC) between two sets of
    signals.

    PAREMETERS
    ----------
    csd : numpy ndarray
    -   Matrix containing the cross-spectral density (CSD) between signals, with
        dimensions [n_signals x n_signals x n_frequencies].

    freqs : list[int | float]
    -   Frequencies in 'csd'.

    method : str
    -   Which form of GC to compute.
    -   Supported inputs are: "gc" for GC from seeds to targets; "net_gc" for
        net GC, i.e. GC from seeds to targets minus GC from targets to seeds;
        "trgc" for time-reversed GC (TRGC) from seeds to targets; and "net_trgc"
        for net TRGC, i.e. TRGC from seeds to targets minus TRGC from targets to
        seeds.

    seeds : list[list[int]]
    -   Indices of signals in 'csd' to treat as seeds. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of seeds.
    -   The number of sublists must match the number of sublists in 'targets'.

    targets : list[list[int]]
    -   Indices of signals in 'csd' to treat as targets. Should be a list of
        sublists, where each sublist contains the signal indices that will be
        treated as a group of targets.
    -   The number of sublists must match the number of sublists in 'seeds'.

    n_seed_components : list[int | None] | None; default None
    -   The number of components that should be taken from the seed entries of
        the CSD after dimensionality reduction using singular value
        decomposition (SVD), with one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the seed data for
        any of the connectivity nodes.
    -   If some values in the list are 'None', no dimensionality reduction is
        performed on the seed data for the corresponding connectivity node.

    n_target_components : list[int | None] | None; default None
    -   The number of components that should be taken from the target entries of
        the CSD after dimensionality reduction using singular value
        decomposition (SVD), with one value for each of the connectivity nodes.
    -   If 'None', no dimensionality reduction is performed on the target data
        for any of the connectivity nodes.
    -   If some values in the list are 'None', no dimensionality reduction is
        performed on the target data for the corresponding connectivity node.

    n_lags : int; default 20
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

    RAISES
    ------
    NotImplementedError
    -   Raised if "method" is not supported.

    NOTES
    -----
    -   Based on the 'data2sctrgcmim' MATLAB function.
    """
    supported_methods = ["gc", "net_gc", "trgc", "net_trgc"]
    if method not in supported_methods:
        raise NotImplementedError(
            f"The method '{method}' for computing frequency-domain Granger "
            "causality is not recognised. Supported methods are "
            f"{supported_methods}."
        )
    if len(indices[0]) != len(indices[1]):
        raise ValueError(
            f"The length of the seed and target indices ({len(indices[0])} and "
            f"{len(indices[1])}, respectively) do not match."
        )

    connectivity = np.zeros((len(autocov), len(freqs)))
    for node_i, node_autocov in enumerate(autocov):
        var_coeffs, residuals_cov = autocov_to_full_var(node_autocov)
        var_coeffs_2d = reshape(
            var_coeffs,
            (var_coeffs.shape[0], var_coeffs.shape[0] * var_coeffs.shape[2]),
        )
        A, K = full_var_to_iss(AF=var_coeffs_2d, V=residuals_cov)
        connectivity[node_i, :] = iss_to_usgc(
            A=A,
            C=var_coeffs_2d,
            K=K,
            V=residuals_cov,
            freqs=freqs,
            seeds=indices[0][node_i],
            targets=indices[1][node_i],
        )

        if "net" in method:
            connectivity[node_i, :] -= iss_to_usgc(
                A=A,
                C=var_coeffs_2d,
                K=K,
                V=residuals_cov,
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
