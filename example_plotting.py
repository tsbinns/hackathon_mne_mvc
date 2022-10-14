from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np

import json
from mne import read_epochs
from Processing.helper_functions import multivar_seed_target_indices
from Processing.mne_wrapper_functions import (
    multivar_spectral_connectivity_epochs,
)

## Load data stored in an MNE Epochs object
data = read_epochs("Data/real_data-epo.fif")

## Load analysis settings
with open("Settings/pipeline_settings.json", encoding="utf-8") as settings_file:
    settings = json.load(settings_file)

## Compute connectivity
indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
wrapper_results = multivar_spectral_connectivity_epochs(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=settings["method"],
    mode=settings["mode"],
    tmin=settings["tmin"],
    tmax=settings["tmax"],
    fmt_fmin=settings["fmt_fmin"],
    fmt_fmax=settings["fmt_fmax"],
    cwt_freqs=settings["cwt_freqs"],
    fmt_n_fft=settings["fmt_n_fft"],
    cwt_use_fft=settings["cwt_use_fft"],
    mt_bandwidth=settings["mt_bandwidth"],
    mt_adaptive=settings["mt_adaptive"],
    mt_low_bias=settings["mt_low_bias"],
    cwt_n_cycles=settings["cwt_n_cycles"],
    cwt_decim=settings["cwt_decim"],
    n_seed_components=settings["n_seed_components"],
    n_target_components=settings["n_target_components"],
    gc_n_lags=settings["gc_n_lags"],
    n_jobs=settings["n_jobs"],
    verbose=settings["verbose"],
)

from mne_connectivity import (
    multivar_seed_target_indices,
    multivar_spectral_connectivity_epochs
)


## Load analysis settings
with open("Settings/pipeline_settings_mne_mvc.json", encoding="utf-8") as settings_file:
    settings = json.load(settings_file)

## Compute connectivity
indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
integrated_results = multivar_spectral_connectivity_epochs(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=settings["method"],
    mode=settings["mode"],
    tmin=settings["tmin"],
    tmax=settings["tmax"],
    fmin=settings["fmt_fmin"],
    fmax=settings["fmt_fmax"],
    cwt_freqs=np.asarray(settings["cwt_freqs"]),
    mt_bandwidth=settings["mt_bandwidth"],
    mt_adaptive=settings["mt_adaptive"],
    mt_low_bias=settings["mt_low_bias"],
    cwt_n_cycles=settings["cwt_n_cycles"],
    n_seed_components=tuple(settings["n_seed_components"]),
    n_target_components=tuple(settings["n_target_components"]),
    n_jobs=settings["n_jobs"],
    block_size=1000,
    verbose=settings["verbose"]
)

freqs = wrapper_results[0].freqs

# Plot results
fig, axs = plt.subplots(1, 2)

axs[0].plot(freqs[:81], np.abs(wrapper_results[0].get_data()[0,:81]), label="MNE Wrapper")
axs[0].plot(freqs[:81], np.abs(integrated_results[0].get_data()[0, 1:82]), label="MNE-integrated")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Abs(connectivity) (A.U.)")
axs[0].set_title("Maximised Imaginary Coherence")

axs[1].plot(freqs[:81], np.abs(wrapper_results[1].get_data()[0,:81]))
axs[1].plot(freqs[:81], np.abs(integrated_results[1].get_data()[0, 1:82]))
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Abs(connectivity) (A.U.)")
axs[1].set_title("Multivariate Interaction Measure")

fig.legend()
plt.show()

# Check similarity of results
if np.allclose(np.abs(wrapper_results[0].get_data()[0,:]),np.abs(integrated_results[0].get_data()[0, 1:])):
    print("MIC results are near-identical across implementations.")
if np.allclose(np.abs(wrapper_results[1].get_data()[0,:]),np.abs(integrated_results[1].get_data()[0, 1:])):
    print("MIM results are near-identical across implementations.")


print("Finished!")