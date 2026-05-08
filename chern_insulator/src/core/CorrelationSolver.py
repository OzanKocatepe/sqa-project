import numpy as np
import matplotlib.pyplot as plt
from functools import cache
from scipy import integrate
from tqdm import tqdm

from data import ModelParameters, Fourier, AxisData
from operators import Hamiltonian
from .Dynamics import Dynamics

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
        self.__dynamics = Dynamics(params, self.__hamiltonian)

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
        HmConv = Fourier(self.__params.drivingFreq, self.__hamiltonian.fourier_minus(np.arange(-n, n + 1))) \
            .BuildConvolutionMatrix()

        HpConv = Fourier(self.__params.drivingFreq, self.__hamiltonian.fourier_plus(np.arange(-n, n + 1))) \
            .BuildConvolutionMatrix()

        HzConv = Fourier(self.__params.drivingFreq, self.__hamiltonian.fourier_z(np.arange(-n, n + 1))) \
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

        # plt.plot(sigmaCoeffs[0:fullN] - np.conjugate(sigmaCoeffs[fullN:2 * fullN]))
        # plt.title(r"$\sigma_-$ coeffs - $\sigma_+^*$ coeffs.")
        # plt.show()

        return [
            Fourier(
                freq = self.__params.drivingFreq,
                coeffs = sigmaCoeffs[0 : fullN]
            ),

            Fourier(
                freq = self.__params.drivingFreq,
                # coeffs = np.conjugate(sigmaCoeffs[0: fullN])
                coeffs = sigmaCoeffs[fullN : 2*fullN]
            ),

            Fourier(
                freq = self.__params.drivingFreq,
                coeffs = sigmaCoeffs[2 * fullN:]
            ) 
        ]

    def SolveDoubleTimeCorrelations(self, tAxis: np.ndarray[float], tauAxis: np.ndarray[float], singleTimeFourier: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the double-time correlations.

        Parameters
        ----------
        tAxis : ndarray[float], shape (n,)
            The tAxis, as stored in AxisData, in seconds.
        tauAxis : ndarray[float], shape(n,)
            The tauAxis, as stored in AxisData, in seconds.
        singleTimeFourier : list[Fourier]
            A list containing the single-time fourier series, in the order sigma_-, sigma_+, sigma_z.

        Returns
        -------
        ndarray[complex]:
            The double-time correlations as an array of shape (3, 3, t.size, tau.size). The first and second axes
            correspond to the left and right operator respectively, with indices 0, 1, and 2 corresponding
            to sigma_-, sigma_+, and sigma_z for both axes. The third axis and fourth axis correspond to the times
            t and tau that the correlation has been evaluated at. Remember that the correlation functions are functions
            sigma_i(t) sigma_j(t + tau).
        """

        doubleTimeCorrelations = np.zeros((3, 3, tAxis.size, tauAxis.size), dtype=complex)
        # Calculates the inhomogenous parts as an array of shape (3, tAxis.size),
        # with first axis corresponding to the left-operator we chose.
        # Remember inhom part of equation is -gamma * <sigma_i(t)>
        inhomParts = -self.__params.decayConstant * np.array([
            singleTimeFourier[0].Evaluate(tAxis),
            singleTimeFourier[1].Evaluate(tAxis),
            singleTimeFourier[2].Evaluate(tAxis)
        ])

        initialConds = self.__DoubleTimeInitialConditions(tAxis, singleTimeFourier)

        # Creates an iterable that contains tuples (leftIndex, tIndex) for
        # every possible combination of the index of the left operator (0, 1, 2)
        # and the index of the 'initial' time t (anything in tAxis.size).
        leftIndexGrid, tIndexGrid = np.meshgrid(range(3), range(tAxis.size))
        indicesIterable = zip(leftIndexGrid.flatten(), tIndexGrid.flatten())

        # Index of the left operator. and index of the 'initial' time t.
        for leftIndex, tIndex in tqdm(indicesIterable,
                                      desc = 'Solving double-time correlations',
                                      position = 1,
                                      leave = False,
                                      total = 3 * tAxis.size):
        # for leftIndex, tIndex in indicesIterable:
            # For the given left-operator and time t, calculates
            # the correlation functions for all right operators
            # and all times t + tau.
            doubleTimeCorrelations[leftIndex, :, tIndex, :] = integrate.solve_ivp(
                fun = self.__dynamics.EquationsOfMotion,
                t_span = (tAxis[tIndex], tAxis[tIndex] + np.max(tauAxis)),
                y0 = initialConds[leftIndex, :, tIndex],
                t_eval = tAxis[tIndex] + tauAxis,
                rtol = 1e-11,
                atol = 1e-12,
                vectorized = True,
                # Uses -<sigma_i(t)\rangle gamma_- as the inhomogenous
                # z-component.
                args = (inhomParts[leftIndex, tIndex],)
            ).y

        return doubleTimeCorrelations

    def __DoubleTimeInitialConditions(self, t: np.ndarray[float], singleTimeFourier: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the double-time initial conditions at some time t.
        
        To solve the double-time correlations, our initial conditions are
        sigma_i (t) sigma_j (t) for some time t in the steady-state period.
        Since these are evaluated at the same time, we can use the Pauli algebra
        to turn these into equations using single-time correlations.
        
        Hence, given some left-operator and start time t, we can calculate the initial
        conditions sigma_i(t) sigma_j(t) for the double-time correlation functions.
        
        We calculate and return the entire initial-conditions matrix for all times in the
        tAxis so that we can call this function only once - indexing the array outside of this function
        is faster than calling the function every single time for different t.

        Parameters
        ----------
        t : ndarray[float], shape (n,)
            The tAxis, as stored in AxisData.
        singleTimeFourier : list[Fourier]
            A list containing the single-time fourier series, in the order sigma_-, sigma_+, sigma_z.
        
        Returns
        -------
        ndarray[complex]:
            An array of shape (3, 3, time.size) (where the third axis can potentially have
            size 1, though in our implementation this should never happen). The first and second axes correspond to the left and right operator
            respectively, with indices 0, 1, and 2 corresponding to sigma_-, sigma_+, and sigma_z for both axes.
            The third axis corresponds to the time t, within the tAxis, that we have evaluated the conditions at.
        """

        sigmam = singleTimeFourier[0].Evaluate(t)        
        sigmap = singleTimeFourier[1].Evaluate(t)
        sigmaz = singleTimeFourier[2].Evaluate(t)

        return np.array([
            # Left operator is sigma_-
            [
                np.zeros(t.size, dtype=complex),
                -0.5 * (sigmaz - 1),
                sigmam
            ],

            # Left operator is sigma_+
            [
                0.5 * (sigmaz + 1),
                np.zeros(t.size, dtype=complex),
                -sigmap
            ],

            # Left operator is sigma_z
            [
                -sigmam,
                sigmap,
                np.ones(t.size, dtype=complex)
            ]
        ])