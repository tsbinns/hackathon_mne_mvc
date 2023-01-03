"""Pipeline for analysing multivariate connectivity integrated in MNE."""

import json
import numpy as np
from mne import read_epochs
from mne_connectivity import (
    multivariate_seed_target_indices,
    multivariate_spectral_connectivity_epochs,
)

## Load data stored in an MNE Epochs object
data = read_epochs("Data/epochs-epo.fif")

## Load analysis settings
with open("Settings/pipeline_settings_mne_mvc.json", encoding="utf-8") as settings_file:
    settings = json.load(settings_file)

## Compute connectivity
indices = multivariate_seed_target_indices(settings["seeds"], settings["targets"])
results = multivariate_spectral_connectivity_epochs(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=settings["method"],
    mode=settings["mode"],
    tmin=settings["tmin"],
    tmax=settings["tmax"],
    fmin=settings["fmin"],
    fmax=settings["fmax"],
    cwt_freqs=np.asarray(settings["cwt_freqs"]),
    mt_bandwidth=settings["mt_bandwidth"],
    mt_adaptive=settings["mt_adaptive"],
    mt_low_bias=settings["mt_low_bias"],
    cwt_n_cycles=settings["cwt_n_cycles"],
    n_seed_components=settings["n_seed_components"],
    n_target_components=settings["n_target_components"],
    gc_n_lags=settings["gc_n_lags"],
    n_jobs=settings["n_jobs"],
    block_size=settings["block_size"],
    verbose=settings["verbose"]
)

print("Finished!")