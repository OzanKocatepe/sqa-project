from data import ModelParameters, Fourier
import numpy as np

class CorrelationSolver:
    """Contains the relevant code for solving the single- and double- time correlations."""

    def __init__(self, params: ModelParameters) -> None:
        """
        Initialises the instance.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the model we are solving the correlations for.
        """

        self.__params = params

    def FourierMatrix(self) -> np.ndarray[complex]:
        """
        Creates the matrix M in the equation Mx = b, where x is a vector containing
        the Fourier series coefficients for sigma_-, sigma_+, sigma_z, and
        b contains the Fourier series coefficients for the components of the inhomogenous
        term in the main ODE.

        Returns
        -------
        ndarray[complex]
            The matrix M, which has shape (3 * (2 * maxN + 1), 3 * (2 * maxN + 1)).
        """

        n = self.__params.maxN
        fullN = 2 * n + 1
        M = np.zeros((3 * fullN, 3 * fullN))

        # Convolution matrix for taking the derivative of a series.
        deriv = 1j * self.__params.angularFreq * np.diag(np.arange(-n, n + 1))
        # HzConv = Fourier.BuildConvolutionMatrix()