"""Wrapper functions for dealing with MNE objects and functions."""

from copy import deepcopy
from typing import Union
from numpy.typing import ArrayLike, NDArray
from mne import BaseEpochs
from mne.time_frequency import (
    CrossSpectralDensity,
    csd_fourier,
    csd_array_fourier,
    csd_multitaper,
    csd_array_multitaper,
    csd_morlet,
    csd_array_morlet,
)
from mne_connectivity import SpectralConnectivity
import numpy as np
from Processing.connectivity_classes import Topographies
from Processing.connectivity_computations import (
    max_imaginary_coherence,
    multivariate_interaction_measure,
    granger_causality,
)


def multivar_spectral_connectivity_epochs(
    data: Union[ArrayLike, BaseEpochs],
    indices: tuple[tuple[ArrayLike]],
    ch_names: Union[list, None] = None,
    ch_coords: Union[ArrayLike, None] = None,
    method: Union[str, list[str]] = "mic",
    sfreq: float = 6.283185307179586,
    mode: str = "multitaper",
    t0: float = 0.0,
    tmin: Union[float, None] = None,
    tmax: Union[float, None] = None,
    fmt_fmin: float = 0.0,
    fmt_fmax: float = np.inf,
    cwt_freqs: list[float] = None,
    fmt_n_fft: Union[int, None] = None,
    cwt_use_fft: bool = True,
    mt_bandwidth: Union[float, None] = None,
    mt_adaptive: bool = False,
    mt_low_bias: bool = True,
    cwt_n_cycles: Union[float, list[float]] = 7.0,
    cwt_decim: Union[int, slice] = 1,
    n_seed_components: Union[tuple[Union[int, None]], None] = None,
    n_target_components: Union[tuple[Union[int, None]], None] = None,
    mic_return_topographies: bool = False,
    gc_n_lags: int = 20,
    n_jobs: int = 1,
    verbose: Union[bool, str, int, None] = None,
) -> Union[
    Union[SpectralConnectivity, list[SpectralConnectivity]],
    tuple[
        Union[SpectralConnectivity, list[SpectralConnectivity]], Topographies
    ],
]:
    """Compute frequency-domain multivariate connectivity measures.

    The connectivity method(s) are specified using the "method" parameter. All
    methods are based on estimates of the cross-spectral densities (CSD) Sxy.

    Based on the "spectral_connectivity_epochs" function of the
    "mne-connectivity" package.

    PARAMETERS
    ----------
    data : BaseEpochs | array
    -   Data to compute connectivity on. If an array, must have the dimensions
        [epochs x signals x timepoints].

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    ch_names : list | None; default None
    -   Names of the channels in the data. If "data" is an Epochs object, these
        channel names will override those in the object.

    ch_coords : array of array of float | None; default None
    -   X-, y-, and z-axis coordinates stored in arrays for each channel in the
        data. If "data" is an Epochs object, these coordinates will override
        those in the object.

    method : str | list of str; default "mic"
    -   Connectivity measure(s) to compute. These can be ['mic', 'mim', 'gc',
        'net_gc', 'trgc', 'net_trgc'].

    sfreq : float; default 6.283185307179586
    -   Sampling frequency of the data. Only used if "data" is an array.

    mode : str; default "multitaper"
    -   Cross-spectral estimation method. Can be 'fourier', 'multitaper', or
        'cwt_wavelet'.

    t0 : float; default 0.0
    -   Time of the first sample relative to the onset of the epoch, in seconds.
        Only used if "data" is an array.

    tmin : float | None; default None
    -   The time at which to start computing connectivity, in seconds. If None,
        starts from the first sample.

    tmax : float | None; default None
    -   The time at which to stop computing connectivity, in seconds. If None,
        ends with the final sample.

    fmt_fmin : float; default 0.0
    -   Minumum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    fmt_fmax : float; default infinity
    -   Maximum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    cwt_freqs : list of float | None; default None
    -   The frequencies of interest, in Hz. If "mode" is 'cwt_morlet', this
        cannot be None. Only used if "mode" if 'cwt_morlet'.

    fmt_n_fft : int | None; default None
    -   Length of the FFT. If None, the exact number of samples between "tmin"
        and "tmax" will be used. Only used if "mode" is 'fourier' or
        'multitaper'.

    cwt_use_fft : bool; default True
    -   Whether to use FFT-based convolution to compute the wavelet transform.
        Only used if "mode" is 'cwt_morlet'.

    mt_bandwidth : float | None; default None
    -   Bandwidth of the multitaper windowing function, in Hz. Only used if
        "mode" if 'multitaper'.

    mt_adaptive : bool; default False
    -   Whether or not to use adaptive weights to combine the tapered spectra
        into the power spectral density. Only used if "mode" if 'multitaper'.

    mt_low_bias : bool; default True
    -   Whether or not to only use tapers with over 90% spectral concentration
        within the bandwidth. Only used if "mode" if 'multitaper'.

    cwt_n_cycles : float | list of float; default 7.0
    -   Number of cycles to use when constructing the Morlet wavelets. Can be a
        single number, or one per frequency. Only used if "mode" if
        'cwt_morlet'.

    cwt_decim : int | slice; default 1
    -   To redice memory usage, decimation factor during time-frequency
        decomposition. Default to 1 (no decimation). If int, uses
        tfr[..., ::"decim"]. If slice, used tfr[..., "decim"]. Only used if
        "mode" is 'cwt_morlet'.

    n_seed_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node. If None, or if an individual
        entry is None, no dimensionality reduction is performed.

    n_target_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of target
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node. If None, or if an individual
        entry is None, no dimensionality reduction is performed.

    mic_return_topographies : bool; default False
    -   Whether or not to return spatial topographies for the connectivity. Only
        used if "method" is 'mic'.

    gc_n_lags : int; default 20
    -   The number of lags to use when computing the autocovariance sequence
        from the cross-spectral density. Only used if "method" is 'gc',
        'net_gc', 'trgc', or 'net_trgc'.

    n_jobs : int; default 1
    -   Number of jobs to run in parallel when computing the cross-spectral
        density.

    verbose : bool | str | int | None; default None
    -   Whether or not to print information about the status of the connectivity
        computations. See MNE's logging information for further details.

    RETURNS
    -------
    results : SpectralConnectivity | list[SpectralConnectivity] |
    tuple(SpectralConnectivity, numpy object array) |
    tuple(list[SpectralConnectivity], numpy object array)
    -   The connectivity (and optionally spatial topography) results.
    -   If "mic_return_topographies" is True and 'mic' is present in "method", a
        tuple of length two is returned, where the first entry is the
        connectivity results, and the second entry are the spatial topographies
        for maximmised imaginary coherence. If 'mic' is not present in "method",
        only the connectivity results are returned.
    -   If "method" contains multiple entries, the connectivity results will be
        a list of SpectralConnectivity object, where each object is the results
        for the corresponding method in "method".
    """
    data = deepcopy(data)

    _check_inputs(
        data=data,
        indices=indices,
        ch_names=ch_names,
        ch_coords=ch_coords,
        method=method,
        mode=mode,
        n_seed_components=n_seed_components,
        n_target_components=n_target_components,
    )

    (
        ch_names,
        ch_coords,
        method,
        picks,
        n_seed_components,
        n_target_components,
    ) = _sort_inputs(
        data=data,
        ch_names=ch_names,
        ch_coords=ch_coords,
        method=method,
        indices=indices,
        n_seed_components=n_seed_components,
        n_target_components=n_target_components,
    )

    data, ch_names, ch_coords, indices = _pick_channels(
        data=data,
        ch_names=ch_names,
        ch_coords=ch_coords,
        picks=picks,
        indices=indices,
    )
    n_epochs = _get_n_epochs(data)

    csd = _compute_csd(
        data=data,
        sfreq=sfreq,
        mode=mode,
        ch_names=ch_names,
        t0=t0,
        tmin=tmin,
        tmax=tmax,
        fmt_fmin=fmt_fmin,
        fmt_fmax=fmt_fmax,
        cwt_freqs=cwt_freqs,
        fmt_n_fft=fmt_n_fft,
        cwt_use_fft=cwt_use_fft,
        mt_bandwidth=mt_bandwidth,
        mt_adaptive=mt_adaptive,
        mt_low_bias=mt_low_bias,
        cwt_n_cycles=cwt_n_cycles,
        cwt_decim=cwt_decim,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return _compute_connectivity(
        csd=csd,
        indices=indices,
        method=method,
        n_seed_components=n_seed_components,
        n_target_components=n_target_components,
        mic_return_topographies=mic_return_topographies,
        gc_n_lags=gc_n_lags,
        mode=mode,
        n_epochs=n_epochs,
        ch_names=ch_names,
        ch_coords=ch_coords,
        verbose=verbose,
    )


def _check_inputs(
    data: Union[BaseEpochs, ArrayLike],
    indices: tuple[tuple[ArrayLike]],
    ch_names: Union[list, None],
    ch_coords: Union[ArrayLike, None],
    method: Union[str, list[str]],
    mode: str,
    n_seed_components: Union[tuple[Union[int, None]], None] = None,
    n_target_components: Union[tuple[Union[int, None]], None] = None,
) -> None:
    """Checks the values of the input parameters to the
    "multivar_spectral_connectivity_epochs" function.

    PARAMETERS
    ----------
    data : Epochs | array
    -   The data on which connectivity is being computed.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    ch_coords : array of array of float | None
    -   X-, y-, and z-axis coordinates stored in arrays for each channel in the
        data.

    method : list[str]
    -   Connectivity measure(s) to compute.

    n_seed_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node. If None, will be replaced
        with a tuple of None.

    n_target_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of target
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node. If None, will be replaced
        with a tuple of None.

    RAISES
    ------
    TypeError
    -   Raised if the data is neither an MNE Epochs object or an array.

    NotImplementedError
    -   Raised if any entries of "method" are not recognised.
    -   Raised if the "mode" is not recognised.

    ValueError
    -   Raised if the length of "ch_names" and the number of channels in the
        data do not match.
    -   Raised if the length of "ch_coords" and the number of channels in the
        data do not match.
    -   Raised if the length of "indices" is not equal to two.
    -   Raised if the length of the seed and target values in "indices" do not
        match.
    -   Raised if the length of "n_seed_components" and "n_target_components" do
        not match.
    -   Raised if the number of components in "n_seed_components" or
        "n_target_components" is greater than the number of channels at the
        corresponding connectivity node for each seed or target, respectively.
    """
    if not isinstance(data, (BaseEpochs, list, tuple, np.ndarray)):
        raise TypeError(
            "The data must be either an MNE Epochs object, or an array."
        )

    if isinstance(data, BaseEpochs):
        n_chs = len(data.info["ch_names"])
    else:
        n_chs = np.shape(data)[1]
    if ch_coords and n_chs != len(ch_coords):
        raise ValueError(
            f"The number of channels ({n_chs}) and provided coordinates "
            f"({len(ch_coords)}) do not match."
        )
    if ch_names and n_chs != len(ch_names):
        raise ValueError(
            f"The number of channels ({n_chs}) and provided channel names "
            f"({len(ch_coords)}) do not match."
        )

    supported_methods = ["mic", "mim", "gc", "net_gc", "trgc", "net_trgc"]
    if isinstance(method, str):
        if method not in supported_methods:
            raise NotImplementedError(
                "One or more methods are not supported connectivity methods."
            )
    else:
        if not all(check in supported_methods for check in method):
            raise NotImplementedError(
                "One or more methods are not supported connectivity methods."
            )

    supported_modes = ["fourier", "multitaper", "morlet"]
    if mode not in supported_modes:
        raise NotImplementedError(
            "The mode is not a supported manner for computing the "
            "cross-spectral density."
        )

    if len(indices) != 2:
        raise ValueError(
            "The indices should have a length of two, consisting of index "
            "values for the seeds and index values for the targets, "
            f"respectively, but has length {len(indices)}."
        )
    if len(indices[0]) != len(indices[1]):
        raise ValueError(
            "The sets of index values for seeds and targets must have the same "
            f"length, but have lengths {len(indices[0])} and "
            f"{len(indices[1])}, respectively."
        )

    if (
        n_seed_components
        and n_target_components
        and len(n_seed_components) != len(n_target_components)
    ):
        raise ValueError(
            "The number of provided seed and target components must match, but "
            f"are {len(n_seed_components)} and {len(n_target_components)}, "
            "respectively."
        )

    if n_seed_components:
        if len(n_seed_components) != len(indices[0]):
            raise ValueError(
                "A value for the number of seed components to take "
                f"({len(n_seed_components)}) must be provided for each set of "
                f"seeds ({len(indices[0])})."
            )
        for n_comps, chs in zip(n_seed_components, indices[0]):
            if n_comps > len(chs):
                raise ValueError(
                    f"The number of components to take ({n_comps}) cannot be "
                    "greater than the number of channels in that seed "
                    f"({len(chs)})."
                )

    if n_target_components:
        if len(n_target_components) != len(indices[1]):
            raise ValueError(
                "A value for the number of target components to take "
                f"({len(n_target_components)}) must be provided for each set "
                f"of targets ({len(indices[1])})."
            )
        for n_comps, chs in zip(n_target_components, indices[1]):
            if n_comps > len(chs):
                raise ValueError(
                    f"The number of components to take ({n_comps}) cannot be "
                    "greater than the number of channels in that target "
                    f"({len(chs)})."
                )


def _sort_inputs(
    data: Union[BaseEpochs, ArrayLike],
    ch_names: Union[list, None],
    ch_coords: Union[ArrayLike, None],
    method: Union[str, list[str]],
    indices: tuple[tuple[ArrayLike]],
    n_seed_components: Union[tuple[Union[int, None]], None] = None,
    n_target_components: Union[tuple[Union[int, None]], None] = None,
) -> tuple[
    list,
    Union[ArrayLike, None],
    list[str],
    NDArray[np.int],
    tuple[Union[int, None], tuple[Union[int, None]]],
]:
    """Sorts the format of the input parameters to the
    "multivar_spectral_connectivity_epochs" function.

    PARAMETERS
    ----------
    data : Epochs | array
    -   The data on which connectivity is being computed.

    ch_names : list | None
    -   Names of the channels in the data. If None and "data" is an Epochs
        object, the names from the Epochs object will be used. If None and
        "data" is an array, the names will be a list of strings from '0' to the
        number of channels in the data.

    ch_coords : array of array of float | None
    -   X-, y-, and z-axis coordinates stored in arrays for each channel in the
        data.

    method : str | list of str
    -   Connectivity measure(s) to compute. If a str, will be converted to a
        list of str.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    n_seed_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node. If None, will be replaced
        with a tuple of None.

    n_target_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of target
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node. If None, will be replaced
        with a tuple of None.

    RETURNS
    -------
    ch_names : list
    -   Names of the channels in the data.

    ch_coords : array of array of float | None
    -   X-, y-, and z-axis coordinates stored in arrays for each channel in the
        data.

    method : list[str]
    -   Connectivity measure(s) to compute.

    picks : array of int
    -   Indices of the channels being used in the connectivity computations.

    n_seed_components : tuple of int or None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node.

    n_target_components : tuple of int or None
    -   Dimensionality reduction parameter specifying the number of target
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node.
    """
    if not ch_names:
        if isinstance(data, BaseEpochs):
            ch_names = data.info["ch_names"]
        else:
            ch_names = [str(i) for i in range(np.shape(data)[1])]

    if not ch_coords:
        if isinstance(data, BaseEpochs):
            ch_coords = data._get_channel_positions(picks=ch_names)

    if isinstance(method, str):
        method = [method]

    picks = np.unique(
        [
            *[seed for seeds in indices[0] for seed in seeds],
            *[target for targets in indices[1] for target in targets],
        ]
    )

    if not n_seed_components:
        n_seed_components = tuple([None] * len(indices[0]))
    if not n_target_components:
        n_target_components = tuple([None] * len(indices[1]))

    return (
        ch_names,
        ch_coords,
        method,
        picks,
        n_seed_components,
        n_target_components,
    )


def _pick_channels(
    data: Union[ArrayLike, BaseEpochs],
    ch_names: list,
    ch_coords: Union[ArrayLike, None],
    picks: NDArray[np.int],
    indices: tuple[tuple[ArrayLike]],
) -> tuple[
    Union[ArrayLike, BaseEpochs],
    list,
    Union[ArrayLike, None],
    tuple[tuple[ArrayLike]],
]:
    """Selects only the channels being used in the connectivity computations
    from the data.

    PARAMETERS
    ----------
    data : Epochs | array
    -   The data on which connectivity is being computed. If an array, should
        have the shape [n_epochs, n_signals, n_times].

    ch_names : list
    -   Names of the channels in the data.

    ch_coords : array of array of float | None
    -   X-, y-, and z-axis coordinates stored in arrays for each channel in the
        data.

    picks : array of int
    -   Indices of the channels being used in the connectivity computation.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    RETURNS
    -------
    data : Epochs | array
    -   The data with only the channels being used in the connectivity
        computations.

    ch_names : list
    -   Names of the channels being used in the connectivity computations.

    ch_coords : array of array of float | None
    -   X-, y-, and z-axis coordinates stored in arrays for each channel in the
        data.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity updated to reflect any changes resulting from dropping
        channels.
    """
    if isinstance(data, BaseEpochs):
        data = data.pick_channels(
            ch_names=[data.info["ch_names"][pick] for pick in picks],
            ordered=True,
        )
    else:
        data = data[:, picks, :]

    if ch_coords is not None:
        ch_coords = ch_coords[picks, :]
    if ch_names is not None:
        ch_names = [name for ch_i, name in enumerate(ch_names) if ch_i in picks]

    indices = _update_indices(indices=indices, picks=picks)

    return data, ch_names, ch_coords, indices


def _update_indices(
    indices: tuple[tuple[ArrayLike]], picks: NDArray[np.int]
) -> tuple[tuple[ArrayLike]]:
    """Updates the connectivity indices in the event that channels are dropped
    from the data.

    PARAMETERS
    ----------
    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    picks : array of int
    -   Indices of the channels being used in the connectivity computation.

    RETURNS
    -------
    updated_indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.
    """
    remapping = {}
    update_required = False
    for pick_i, ch_i in enumerate(picks):
        remapping[ch_i] = pick_i
        if not update_required and pick_i != ch_i:
            update_required = True

    updated_indices = deepcopy(indices)
    if update_required:
        for seed_i, seed_idcs in enumerate(indices[0]):
            indices[0][seed_i] = [remapping[idx] for idx in seed_idcs]
        for target_i, target_idcs in enumerate(indices[1]):
            indices[1][target_i] = [remapping[idx] for idx in target_idcs]

    return updated_indices


def _get_n_epochs(data: Union[BaseEpochs, ArrayLike]) -> int:
    """Finds the number of epochs in the data.

    PARAMETERS
    ----------
    data : Epochs | array
    -   The data on which connectivity is being computed. If an array, must
        have shape [n_epochs, n_signals, n_freqs].

    RETURNS
    -------
    int
    -   The number of epochs in the data.
    """
    if isinstance(data, BaseEpochs):
        return data.get_data(picks=[0]).shape[0]
    else:
        return data.shape[0]


def _compute_csd(
    data: Union[ArrayLike, BaseEpochs],
    sfreq: float,
    mode: str,
    ch_names: list,
    t0: float,
    tmin: Union[float, None],
    tmax: Union[float, None],
    fmt_fmin: Union[float, tuple[float]],
    fmt_fmax: Union[float, tuple[float]],
    cwt_freqs: Union[ArrayLike, None],
    fmt_n_fft: Union[int, None],
    cwt_use_fft: bool,
    mt_bandwidth: Union[float, None],
    mt_adaptive: bool,
    mt_low_bias: bool,
    cwt_n_cycles: Union[float, ArrayLike],
    cwt_decim: Union[int, slice],
    n_jobs: int,
    verbose: Union[bool, str, int, None],
) -> CrossSpectralDensity:
    """Computes the cross-spectral density of the data.

    PARAMETERS
    ----------
    data : BaseEpochs | array
    -   Data to compute connectivity on. If an array, must have the dimensions
        [epochs x signals x timepoints].

    sfreq : float; default 6.283185307179586
    -   Sampling frequency of the data. Only used if "data" is an array.

    mode : str; default "multitaper"
    -   Cross-spectral estimation method. Can be 'fourier', 'multitaper', or
        'cwt_wavelet'.

    ch_names : list | None; default None
    -   Names of the channels in the data. If "data" is an Epochs object, these
        channel names will override those in the object.

    t0 : float; default 0.0
    -   Time of the first sample relative to the onset of the epoch, in seconds.
        Only used if "data" is an array.

    tmin : float | None; default None
    -   The time at which to start computing connectivity, in seconds. If None,
        starts from the first sample.

    tmax : float | None; default None
    -   The time at which to stop computing connectivity, in seconds. If None,
        ends with the final sample.

    fmt_fmin : float; default 0.0
    -   Minumum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    fmt_fmax : float; default infinity
    -   Maximum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    cwt_freqs : list of float | None; default None
    -   The frequencies of interest, in Hz. If "mode" is 'cwt_morlet', this
        cannot be None. Only used if "mode" if 'cwt_morlet'.

    fmt_n_fft : int | None; default None
    -   Length of the FFT. If None, the exact number of samples between "tmin"
        and "tmax" will be used. Only used if "mode" is 'fourier' or
        'multitaper'.

    cwt_use_fft : bool; default True
    -   Whether to use FFT-based convolution to compute the wavelet transform.
        Only used if "mode" is 'cwt_morlet'.

    mt_bandwidth : float | None; default None
    -   Bandwidth of the multitaper windowing function, in Hz. Only used if
        "mode" if 'multitaper'.

    mt_adaptive : bool; default False
    -   Whether or not to use adaptive weights to combine the tapered spectra
        into the power spectral density. Only used if "mode" if 'multitaper'.

    mt_low_bias : bool; default True
    -   Whether or not to only use tapers with over 90% spectral concentration
        within the bandwidth. Only used if "mode" if 'multitaper'.

    cwt_n_cycles : float | list of float; default 7.0
    -   Number of cycles to use when constructing the Morlet wavelets. Can be a
        single number, or one per frequency. Only used if "mode" if
        'cwt_morlet'.

    cwt_decim : int | slice; default 1
    -   To redice memory usage, decimation factor during time-frequency
        decomposition. Default to 1 (no decimation). If int, uses
        tfr[..., ::"decim"]. If slice, used tfr[..., "decim"]. Only used if
        "mode" is 'cwt_morlet'.

    n_jobs : int; default 1
    -   Number of jobs to run in parallel when computing the cross-spectral
        density.

    verbose : bool | str | int | None; default None
    -   Whether or not to print information about the status of the connectivity
        computations. See MNE's logging information for further details.

    RETURNS
    -------
    CrossSpectralDensity
    -   The cross-spectral density of the data between all channels.
    """
    if isinstance(data, BaseEpochs):
        if mode == "fourier":
            return csd_fourier(
                epochs=data,
                fmin=fmt_fmin,
                fmax=fmt_fmax,
                tmin=tmin,
                tmax=tmax,
                picks=None,
                n_fft=fmt_n_fft,
                projs=None,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        if mode == "multitaper":
            return csd_multitaper(
                epochs=data,
                fmin=fmt_fmin,
                fmax=fmt_fmax,
                tmin=tmin,
                tmax=tmax,
                picks=None,
                n_fft=fmt_n_fft,
                bandwidth=mt_bandwidth,
                adaptive=mt_adaptive,
                low_bias=mt_low_bias,
                projs=None,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        return csd_morlet(
            epochs=data,
            frequencies=cwt_freqs,
            tmin=tmin,
            tmax=tmax,
            picks=None,
            n_cycles=cwt_n_cycles,
            use_fft=cwt_use_fft,
            decim=cwt_decim,
            projs=None,
            n_jobs=n_jobs,
            verbose=None,
        )
    else:
        if mode == "fourier":
            return csd_array_fourier(
                X=data,
                sfreq=sfreq,
                t0=t0,
                fmin=fmt_fmin,
                fmax=fmt_fmax,
                tmin=tmin,
                tmax=tmax,
                ch_names=ch_names,
                n_fft=fmt_n_fft,
                projs=None,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        if mode == "multitaper":
            return csd_array_multitaper(
                X=data,
                sfreq=sfreq,
                t0=t0,
                fmin=fmt_fmin,
                fmax=fmt_fmax,
                tmin=tmin,
                tmax=tmax,
                ch_names=ch_names,
                n_fft=fmt_n_fft,
                bandwidth=mt_bandwidth,
                adaptive=mt_adaptive,
                low_bias=mt_low_bias,
                projs=None,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        return csd_array_morlet(
            X=data,
            sfreq=sfreq,
            frequencies=cwt_freqs,
            t0=t0,
            tmin=tmin,
            tmax=tmax,
            ch_names=ch_names,
            n_cycles=cwt_n_cycles,
            use_fft=cwt_use_fft,
            decim=cwt_decim,
            projs=None,
            n_jobs=n_jobs,
            verbose=verbose,
        )


def _compute_connectivity(
    csd: CrossSpectralDensity,
    indices: tuple[tuple[ArrayLike]],
    method: list[str],
    n_seed_components: Union[tuple[Union[int, None]], None],
    n_target_components: Union[tuple[Union[int, None]], None],
    mic_return_topographies: bool,
    gc_n_lags: int,
    mode: str,
    n_epochs: int,
    ch_names: list,
    ch_coords: Union[ArrayLike, None],
    verbose: bool,
) -> Union[
    Union[SpectralConnectivity, list[SpectralConnectivity]],
    tuple[
        Union[SpectralConnectivity, list[SpectralConnectivity]], Topographies
    ],
]:
    """Computes connectivity results from the cross-spectral density.

    PARAMETERS
    ----------
    csd : CrossSpectralDensity
    -   The cross-spectral density of the data between all channels.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    method : list of str
    -   Connectivity measure(s) to compute.

    mic_return_topographies : bool; default False
    -   Whether or not to return spatial topographies for the connectivity. Only
        used if the method is 'mic'.

    n_seed_components : tuple of int or None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node.

    n_target_components : tuple of int or None
    -   Dimensionality reduction parameter specifying the number of target
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node.

    gc_n_lags : int; default 20
    -   The number of lags to use when computing the autocovariance sequence
        from the cross-spectral density. Only used if the method is 'gc',
        'net_gc', 'trgc', or 'net_trgc'.

    mode : str
    -   The name of the mode used to compute the cross-spectral density.

    n_epochs : int
    -   The number of epochs used to compute the cross-spectral density.

    ch_names : list
    -   The names of the channels in the data.

    ch_coords : array of array of float | None
    -   The x-, y-, and z-axis coordinates stored in arrays for each channel in
        the data.

    verbose: bool
    -   Whether or not to print information about the connectivity computations.

    RETURNS
    -------
    connectivity : SpectralConnectivity | list[SpectralConnectivity] |
    tuple(SpectralConnectivity, Topographies) |
    tuple(list[SpectralConnectivity], Topographies)
    -   The connectivity (and optionally spatial topography) results.
    -   If "mic_return_topographies" is True and 'mic' is present in "method", a
        tuple of length two is returned, where the first entry is the
        connectivity results, and the second entry are the spatial topographies
        for maximmised imaginary coherence. If 'mic' is not present in "method",
        only the connectivity results are returned.
    -   If "method" contains multiple entries, the connectivity results will be
        a list of SpectralConnectivity object, where each object is the results
        for the corresponding method in "method".
    """
    connectivity = []
    topographies = None
    csd_matrix = np.transpose(
        np.asarray([csd.get_data(freq) for freq in csd.frequencies]), (1, 2, 0)
    )

    for con_method in method:
        if verbose:
            print(f"\nComputing connectivity for the method '{con_method}'.")
        if con_method == "mic":
            con = max_imaginary_coherence(
                csd=csd_matrix,
                seeds=indices[0],
                targets=indices[1],
                n_seed_components=n_seed_components,
                n_target_components=n_target_components,
                return_topographies=mic_return_topographies,
            )
            if mic_return_topographies:
                topographies = deepcopy(con[1])
                con = con[0]
        elif con_method == "mim":
            con = multivariate_interaction_measure(
                csd=csd_matrix,
                seeds=indices[0],
                targets=indices[1],
                n_seed_components=n_seed_components,
                n_target_components=n_target_components,
            )
        else:
            con = granger_causality(
                csd=csd_matrix,
                freqs=csd.frequencies,
                method=con_method,
                seeds=indices[0],
                targets=indices[1],
                n_seed_components=n_seed_components,
                n_target_components=n_target_components,
                n_lags=gc_n_lags,
            )
        connectivity.append(con)

    connectivity = _connectivity_to_mne(
        data=connectivity,
        freqs=csd.frequencies,
        indices=indices,
        ch_names=ch_names,
        method=method,
        spec_method=mode,
        n_epochs_used=n_epochs,
    )
    if topographies:
        topographies = _handle_topographies(
            data=topographies,
            freqs=csd.frequencies,
            indices=indices,
            ch_names=ch_names,
            ch_coords=ch_coords,
            spec_method=mode,
            n_epochs_used=n_epochs,
        )

    if len(method) == 1:
        connectivity = connectivity[0]
    if topographies:
        connectivity = [connectivity, topographies]

    return connectivity


def _connectivity_to_mne(
    data: list[NDArray],
    freqs: NDArray[np.float],
    indices: tuple[tuple[ArrayLike]],
    ch_names: list,
    method: list[str],
    spec_method: str,
    n_epochs_used: int,
) -> list[SpectralConnectivity]:
    """Stores the connectivity results in an MNE SpectralConnectivity object.

    PARAMETERS
    ----------
    data : list of array
    -   The connectivity results, with one array in the list for each entry of
        "method".

    freqs : array of float
    -   The frequencies in the connectivity results.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections used to compute
        connectivity.

    ch_names : list
    -   Names of the channels used to compute connectivity.

    method : list of str
    -   Computed connectivity measure(s).

    spec_method : str
    -   Method used to compute the cross-spectral density on which the
        connectivity results are based.

    n_epochs_used : int
    -   Number of epochs used to compute the cross-spectral density on which the
        connectivity results are based.

    RETURNS
    -------
    connectivity : list[SpectralConnectivity]
    -   List of connectivity results stored in MNE objects, with one object in
        the list for each computed connectivity measure.
    """
    connectivity = []
    for method_i, con_data in enumerate(data):
        connectivity.append(
            SpectralConnectivity(
                data=con_data,
                freqs=freqs,
                n_nodes=len(ch_names),
                names=ch_names,
                indices=indices,
                method=method[method_i],
                spec_method=spec_method,
                n_epochs_used=n_epochs_used,
            )
        )

    return connectivity


def _handle_topographies(
    data: tuple[NDArray],
    freqs: NDArray,
    indices: tuple[tuple[ArrayLike]],
    ch_names: list[str],
    ch_coords: Union[ArrayLike, None],
    spec_method: str,
    n_epochs_used: int,
) -> Topographies:
    """Stores the spatial topography results in a placeholder object.

    PARAMETERS
    ----------
    data : tuple(numpy object array, numpy object array)
    -   Spatial topographies of connectivity for seeds and targets,
        respectively. The entries for seeds and targets have dimensions [nodes x
        signals x frequencies], where signals correspond to the number of seed
        and target signals in each node, respectively.

    freqs : array of float
    -   The frequencies in the connectivity results.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections used to compute
        connectivity.

    ch_names : list
    -   Names of the channels used to compute connectivity.

    ch_coords : array of array of float | None
    -   The x-, y-, and z-axis coordinates stored in arrays for each channel in
        the data.

    spec_method : str
    -   Method used to compute the cross-spectral density on which the
        connectivity results are based.

    n_epochs_used : int
    -   Number of epochs used to compute the cross-spectral density on which the
        connectivity results are based.

    RETURNS
    -------
    Topographies
    -   The spatial topography results stored in a placeholder object.
    """
    return Topographies(
        data=data,
        freqs=freqs,
        n_nodes=len(ch_names),
        names=ch_names,
        coords=ch_coords,
        indices=indices,
        method="mic",
        spec_method=spec_method,
        n_epochs_used=n_epochs_used,
    )
