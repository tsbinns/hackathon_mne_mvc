"""Minor functions for computing connectivity.

FUNCTIONS
---------
multivar_seed_target_indices
-   Generates indices parameter for seed-based multivariate connectivity
    analysis.
"""

from numpy.typing import ArrayLike


def multivar_seed_target_indices(
    seeds: ArrayLike, targets: ArrayLike
) -> tuple[ArrayLike]:
    """Generates indices parameter for seed-based multivariate connectivity
    analysis.

    PARAMETERS
    ----------
    seeds : array of array of int
    -   Seed indices, consisting of an array composed of sub-arrays, where each
        each sub-array contains the indices of the channels for the seeds of a
        single connectivity node.

    targets : array of array of int
    -   Target indices, consisting of an array composed of sub-arrays, where
        each each sub-array contains the indices of the channels for the targets
        of a single connectivity node. For each "seeds" sub-array, a
        corresponding entry for each of the "targets" sub-arrays will be added
        to the indices.

    RETURNS
    -------
    indices : tuple of array of array of int
    -   The indices paramater used for multivariate connectivity computation.
        Consists of two arrays corresponding to the seeds and targets,
        respectively, where each array is composed of sub-arrays, where each
        sub-array contains the indices of the channels for the seeds/targets of
        a single connectivity node.
    """
    indices = [[], []]
    for seed in seeds:
        for target in targets:
            indices[0].append(seed)
            indices[1].append(target)

    return tuple(indices)
