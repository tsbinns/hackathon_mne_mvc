"""An example pipeline for analysing multivariate connectivity."""

import json
from mne import read_epochs
from mne_connectivity import (
    multivar_seed_target_indices,
    multivar_spectral_connectivity_epochs
)

## Load data stored in an MNE Epochs object
data = read_epochs("Data/epochs-epo.fif")

## Load analysis settings
with open("Settings/pipeline_settings_mne.json", encoding="utf-8") as settings_file:
    settings = json.load(settings_file)

## Compute connectivity
indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
results = multivar_spectral_connectivity_epochs(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=settings["method"],
    mode=settings["mode"],
    tmin=settings["tmin"],
    tmax=settings["tmax"],
    fmin=settings["fmt_fmin"],
    fmax=settings["fmt_fmax"],
    cwt_freqs=settings["cwt_freqs"],
    mt_bandwidth=settings["mt_bandwidth"],
    mt_adaptive=settings["mt_adaptive"],
    mt_low_bias=settings["mt_low_bias"],
    cwt_n_cycles=settings["cwt_n_cycles"],
    n_seed_components=settings["n_seed_components"],
    n_target_components=settings["n_target_components"],
    n_jobs=settings["n_jobs"],
    verbose=settings["verbose"]
)