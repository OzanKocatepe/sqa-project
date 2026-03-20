from data import ModelParameters, Fourier
from Hamiltonian import Hamiltonian
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
        self.__hamiltonian = Hamiltonian(params)

    def __SingleTimeFourierMatrix(self) -> np.ndarray[complex]:
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

        # Defines a bunch of shorthand.
        n = self.__params.maxN
        fullN = 2 * n + 1
        gamma = self.__params.decayConstant

        # Creates the empty matrix.
        M = np.zeros((3 * fullN, 3 * fullN), dtype=complex)

        # Convolution matrix for taking the derivative of a series.
        deriv = 1j * self.__params.angularFreq * np.diag(np.arange(-n, n + 1))

        # Builds the convolution matrices for Hm, Hp, Hz.
        HmConv = Fourier(self.__params.drivingFreq, self.__hamiltonian.Hmn(np.arange(-n, n + 1))) \
            .BuildConvolutionMatrix()

        HpConv = Fourier(self.__params.drivingFreq, self.__hamiltonian.Hpn(np.arange(-n, n + 1))) \
            .BuildConvolutionMatrix()

        HzConv = Fourier(self.__params.drivingFreq, self.__hamiltonian.Hzn(np.arange(-n, n + 1))) \
            .BuildConvolutionMatrix()

        # First row, equation for sigma_-.
        M[0:fullN, 0:fullN] = deriv + 2j * HzConv + 0.5 * gamma * np.eye(fullN)
        M[0:fullN, 2 * fullN:] = -1j * HpConv

        # Second row, equation for sigma_+.
        M[fullN:2 * fullN, fullN:2 * fullN] = deriv - 2j * HzConv + 0.5 * gamma * np.eye(fullN)
        M[fullN:2 * fullN, 2 * fullN:] = 1j * HmConv

        # Third row, equation for sigma_z.
        M[2 * fullN:, 0:fullN] = -2j * HmConv
        M[2 * fullN:, fullN:2 * fullN] = 2j * HpConv
        M[2 * fullN:, 2 * fullN:] = deriv + gamma * np.eye(fullN)

        return M
    
    def SolveSingleTimeCorrelations(self) -> list[Fourier]:
        """Solves the single time correlations as Fourier series.
        
        Returns
        -------
        list[Fourier]:
            A list of the Fourier series for sigma_-, sigma_+, sigma_z respectively.
        """

        n = self.__params.maxN
        fullN = 2 * n + 1
        M = self.__SingleTimeFourierMatrix()

        # Creates the right hand side of the equation Mx = b.
        b = np.zeros((3 * fullN), dtype=complex)
        # Corresponds to the inhomogenous part -gamma in the ODEs.
        b[2 * fullN + n] = -self.__params.decayConstant

        sigmaCoeffs = np.linalg.solve(M, b)

        return [
            Fourier(
                freq = self.__params.drivingFreq,
                coeffs = sigmaCoeffs[0: fullN]
            ),

            Fourier(
                freq = self.__params.drivingFreq,
                coeffs = sigmaCoeffs[fullN:2 * fullN]
            ),

            Fourier(
                freq = self.__params.drivingFreq,
                coeffs = sigmaCoeffs[2 * fullN:]
            ) 
        ]