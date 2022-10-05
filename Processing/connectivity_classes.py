"""Classes for storing connectivity results."""

from numpy.typing import ArrayLike, NDArray


class Topographies:
    """Placeholder class for storing spatial topography results.

    PARAMETERS
    ----------
    data : tuple(numpy object array, numpy object array)
    -   Spatial topographies of connectivity for seeds and targets,
        respectively. The entries for seeds and targets have dimensions [nodes x
        signals x frequencies], where signals correspond to the number of seed
        and target signals in each node, respectively.

    freqs : array of float
    -   The frequencies in the connectivity results.

    n_nodes : int
    -   The number of channels used to compute connectivity.

    indices : tuple of tuple of array of int
    -   Two tuples of arrays with indices of connections used to compute
        connectivity.

    names : list
    -   Names of the channels used to compute connectivity, which the index
        values in indices correspond to.

    coords : array of array of float | None
    -   The x-, y-, and z-axis coordinates stored in arrays for each channel in
        the data.

    method : str
    -   The connectivity method from which the topographies were derived.

    spec_method : str
    -   Method used to compute the cross-spectral density on which the
        connectivity results are based.

    n_epochs_used : int
    -   Number of epochs used to compute the cross-spectral density on which the
        connectivity results are based.
    """

    def __init__(
        self,
        data: tuple[NDArray],
        freqs: NDArray,
        n_nodes: int,
        indices: tuple[tuple[ArrayLike]],
        names: list[str] = None,
        coords: ArrayLike = None,
        method: str = None,
        spec_method: str = None,
        n_epochs_used: int = None,
    ) -> None:
        self.data = data
        self.freqs = freqs
        self.n_nodes = n_nodes
        self.names = names
        self.ch_coords = coords
        self.indices = indices
        self.method = method
        self.spec_method = spec_method
        self.n_epochs_used = n_epochs_used

        self._sort_inputs()

    def _sort_inputs(self) -> None:
        """Ensures that the inputs are in the appropriate format.

        RAISES
        ------
        ValueError
        -   Raised if the number of channel names and nodes do not match.
        -   Raised if the number of channel coordinates and nodes do not match.
        -   Raised if any set of coordinates does not have three values.
        """
        if self.names is not None:
            if len(self.names) != self.n_nodes:
                raise ValueError(
                    "The number of channels names and nodes do not match."
                )
        else:
            self.names = [str(ch_i) for ch_i in self.n_nodes]

        if self.ch_coords is not None:
            if len(self.ch_coords) != self.n_nodes:
                raise ValueError(
                    "The number of channel coordinates and nodes do not match."
                )
            for coord in self.ch_coords:
                if len(coord) != 3:
                    raise ValueError(
                        "Each set of coordinates must have three values, "
                        "corresponding to the x-, y-, and z-axis coordinates, "
                        "respectively."
                    )
