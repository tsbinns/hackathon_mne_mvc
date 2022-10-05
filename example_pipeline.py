"""An example pipeline for analysing multivariate connectivity."""

from copy import deepcopy
import json
from mne import read_epochs
from Processing.helper_functions import multivar_seed_target_indices
from Processing.mne_wrapper_functions import (
    multivar_spectral_connectivity_epochs,
)

## Load data stored in an MNE Epochs object
data = read_epochs("Data/epochs-epo.fif")

## Load analysis settings
with open("Settings/pipeline_settings.json", encoding="utf-8") as settings_file:
    settings = json.load(settings_file)

## Compute connectivity
indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
results = multivar_spectral_connectivity_epochs(
    data=data,
    indices=indices,
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
    mic_return_topographies=settings["mic_return_topographies"],
    gc_n_lags=settings["gc_n_lags"],
    n_jobs=settings["n_jobs"],
    verbose=settings["verbose"],
)

## Extract connectivity results
connectivity = {}
if "mic" in settings["method"] and settings["mic_return_topographies"]:
    connectivity["mic_topographies"] = deepcopy(results[1])
    results = results[0]
for con, method in zip(results, settings["method"]):
    connectivity[method] = con

print("Finished!")
