from data import ModelParameters

class Model:
    """Contains a Chern Insulator model evaluated at a single pair kx, ky."""

    def __init__(self, params: ModelParameters) -> None:
        """Initialises the instance.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters to use for this model.
        """

        self.__params = params