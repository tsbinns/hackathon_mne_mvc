"""Plots results of MIC/MIM from MNE-integrated and wrapper implementations."""

from matplotlib import pyplot as plt
import numpy as np
from mne import read_epochs
from Processing.mne_wrapper_functions import multivar_spectral_connectivity_epochs as wrapper_mvc
from mne_connectivity import multivar_spectral_connectivity_epochs as integrated_mvc

## Load data stored in an MNE Epochs object
data = read_epochs("Data/epochs-epo.fif")

## Compute connectivity
methods = ["mic", "mim"]
indices = tuple([[[0, 1, 2, 3, 4]], [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]])
import time
start = time.time()
wrapper_results = wrapper_mvc(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=methods,
    mode="multitaper",
    fmt_fmin=0,
    fmt_fmax=50.5,
    mt_bandwidth=5,
    mt_adaptive=True,
    mt_low_bias=True,
    n_seed_components=None,
    n_target_components=None,
    n_jobs=3
)
wrapper_time = time.time()-start


## Compute connectivity
start=time.time()
integrated_results = integrated_mvc(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=methods,
    mode="multitaper",
    fmin=0.5,
    fmax=50.0,
    mt_bandwidth=5,
    mt_adaptive=True,
    mt_low_bias=True,
    n_seed_components=None,
    n_target_components=None,
    n_jobs=3
)
int_time = time.time()-start
print(f"Wrapper time = {wrapper_time}; Integrated time = {int_time}")
freqs = wrapper_results[0].freqs

# Plot results
fig, axs = plt.subplots(1, 2)
fig.suptitle("Cortex-STN connectivity")

axs[0].plot(freqs, np.abs(wrapper_results[0].get_data()[0, :]), label="MNE Wrapper", linewidth=5)
axs[0].plot(freqs, np.abs(integrated_results[0].get_data()[0, :]), label="MNE-integrated", linestyle="--", linewidth=5)
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Abs(connectivity) (A.U.)")
axs[0].set_title("Maximised Imaginary Coherence")

axs[1].plot(freqs[:], np.abs(wrapper_results[1].get_data()[0, :]), linewidth=5)
axs[1].plot(freqs[:], np.abs(integrated_results[1].get_data()[0, :]), linestyle="--", linewidth=5)
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Abs(connectivity) (A.U.)")
axs[1].set_title("Multivariate Interaction Measure")

fig.legend()
plt.show()

# Check similarity of results
for method_i, method in enumerate(methods):
    if np.allclose(
        wrapper_results[method_i].get_data(),
        integrated_results[method_i].get_data(),
        atol=1e-5
    ):
        print(f"{method} results are near-identical across implementations.")
    else:
        raise ValueError(
            f"{method} results are different across implementations."
        )


print("Finished!")