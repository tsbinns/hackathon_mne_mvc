# hackathon_mne_mvc
Multivariate connectivity methods implemented in Python and based on functions and objects available in MNE.

The methods include:
- Maximised imaginary coherence [[1]](#References)
- Multivariate interaction measure [[1]](#References)
- Granger causality based on state space models [[2 & 3]](#References) with optional time-reversal [[4]](#References)

## Requirements
1. The base [Anaconda](https://www.anaconda.com/) package.
2. The [MNE](https://mne.tools/stable/index.html) and [MNE-connectivity](https://mne.tools/mne-connectivity/stable/index.html) packages.

## Use
Use the _example_pipeline.py_ script to generate the different multivariate results based on the _pipeline_settings.json_ file in the _Settings_ folder.

## References
[1] Ewald _et al._ (2012). _NeuroImage_. DOI: [10.1016/j.neuroimage.2011.11.084](https://doi.org/10.1016/j.neuroimage.2011.11.084).

[2] Barnett & Seth (2014). _Journal of Neuroscience Methods_. DOI: [10.1016/j.jneumeth.2013.10.018](https://doi.org/10.1016/j.jneumeth.2013.10.018).

[3] Barnett & Seth (2015). _Physical Review E_. DOI: [10.1103/PhysRevE.91.040101](https://doi.org/10.1103/PhysRevE.91.040101).

[4] Winkler _et al._ (2016). _IEEE Transactions on Signal Processing_. DOI: [10.1109/TSP.2016.2531628](https://doi.org/10.1109/TSP.2016.2531628).
