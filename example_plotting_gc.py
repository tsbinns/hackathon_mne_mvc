from matplotlib import pyplot as plt
import numpy as np
from mne import read_epochs
from Processing.mne_wrapper_functions import (
    multivar_spectral_connectivity_epochs,
)

## Load data stored in an MNE Epochs object
data = read_epochs("Data/epochs-epo.fif")

## Compute connectivity
methods = ["gc", "net_gc", "trgc", "net_trgc"]
indices = tuple([[[0, 1, 2, 3, 4]], [[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]])
import time
start = time.time()
wrapper_results = multivar_spectral_connectivity_epochs(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=methods,
    mode="multitaper",
    fmt_fmin=0,
    mt_bandwidth=5,
    mt_adaptive=True,
    mt_low_bias=True,
    n_seed_components=[5],
    n_target_components=[7],
    gc_n_lags=10,
    n_jobs=3
)
wrapper_time = time.time()-start


from mne_connectivity import multivar_spectral_connectivity_epochs
## Compute connectivity
start=time.time()
integrated_results = multivar_spectral_connectivity_epochs(
    data=data,
    indices=indices,
    names=data.info["ch_names"],
    method=methods,
    mode="multitaper",
    fmin=0.5,
    mt_bandwidth=5,
    mt_adaptive=True,
    mt_low_bias=True,
    n_seed_components=[5],
    n_target_components=[7],
    gc_n_lags=10,
    n_jobs=3
)
int_time = time.time()-start
print(wrapper_time, int_time)
freqs = wrapper_results[0].freqs

# Plot results
fig, axs = plt.subplots(2, 2)
fig.suptitle("Cortex-STN connectivity")

axs[0, 0].plot(freqs[:81], wrapper_results[0].get_data()[0, :81], label="MNE Wrapper", linewidth=5)
axs[0, 0].plot(freqs[:81], integrated_results[0].get_data()[0, :81], label="MNE-integrated", linestyle="--", linewidth=5)
axs[0, 0].set_xlabel("Frequency (Hz)")
axs[0, 0].set_ylabel("Connectivity (A.U.)")
axs[0, 0].set_title("Granger Causality")

axs[0, 1].plot(freqs[:81], wrapper_results[1].get_data()[0, :81], linewidth=5)
axs[0, 1].plot(freqs[:81], integrated_results[1].get_data()[0, :81], linestyle="--", linewidth=5)
axs[0, 1].set_xlabel("Frequency (Hz)")
axs[0, 1].set_ylabel("Connectivity (A.U.)")
axs[0, 1].set_title("Net Granger Causality")

axs[1, 0].plot(freqs[:81], wrapper_results[2].get_data()[0, :81], linewidth=5)
axs[1, 0].plot(freqs[:81], integrated_results[2].get_data()[0, :81], linestyle="--", linewidth=5)
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_ylabel("Connectivity (A.U.)")
axs[1, 0].set_title("Time-Reversed Granger Causality")

axs[1, 1].plot(freqs[:81], wrapper_results[3].get_data()[0, :81], linewidth=5)
axs[1, 1].plot(freqs[:81], integrated_results[3].get_data()[0, :81], linestyle="--", linewidth=5)
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Connectivity (A.U.)")
axs[1, 1].set_title("Net Time-Reversed Granger Causality")

fig.legend()
plt.show()

# Check similarity of results
if np.allclose(wrapper_results[0].get_data(), integrated_results[0].get_data()):
    print("GC results are near-identical across implementations.")
else:
    raise ValueError("GC results are different across implementations.")
if np.allclose(wrapper_results[1].get_data(), integrated_results[1].get_data()):
    print("Net GC results are near-identical across implementations.")
else:
    raise ValueError("Net GC results are different across implementations.")
if np.allclose(wrapper_results[2].get_data(), integrated_results[2].get_data()):
    print("TRGC results are near-identical across implementations.")
else:
    raise ValueError("TRGC results are different across implementations.")
if np.allclose(wrapper_results[3].get_data(), integrated_results[3].get_data()):
    print("Net TRGC results are near-identical across implementations.")
else:
    raise ValueError("Net TRGC results are different across implementations.")


print("Finished!")