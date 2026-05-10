import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from data import ModelParameters, Fourier, AxisData
from operators import Hamiltonian
from .Dynamics import Dynamics

class CorrelationSolver:
    """Contains the relevant code for solving the single- and double- time correlations."""

    def __init__(self, params: ModelParameters, hamiltonian: Hamiltonian) -> None:
        """
        Initialises the instance.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the model we are solving the correlations for.
        hamiltonian: Hamiltonian
            The hamiltonian of the system.
        """

        self.__params = params
        self.__hamiltonian = hamiltonian
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
    
    # def __DoubleTimeFourierMatrix(self, t_prime: float) -> np.ndarray[complex]:
        """Creates the matrix used to solve for the Fourier series of the double-time correlations.
        
        The full ODE for the double-time correlation functions is a function of t and t'. By making our
        double-time correlation functions into Fourier series' in t, we only need to find the 2N + 1
        coefficients of the Fourier series'. However, each coefficient is a function of t', so we each coefficient
        into a Fourier series in t', and then solve the matrix ODE to find the Fourier series.
        
        Hence, this matrix lets us find the coefficients of the Fourier series' for """
    
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

    def SolveDoubleTimeCorrelationsFourier(self, tAxis: np.ndarray[float], tauAxis: np.ndarray[float], singleTimeFourier: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the double-time correlations.

        WARNING: CURRENTLY WORK IN PROGRESS.

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

        n = self.__params.maxN
        fullN = 2 * n + 1
        tPlusTauAxis = np.add.outer(tAxis, tauAxis)
        # We use the exact same B matrix as the single-time case, since we are still expanding the same
        # ODE matrix w.r.t. t'.
        M = self.__SingleTimeFourierMatrix()

        # Creates the right hand side of the equation Mx = b.
        b = np.zeros((3 * fullN), dtype=complex)

        doubleTimeCorrelations = np.zeros((3, 3, tAxis.size, tauAxis.size), dtype=complex)

        # Loops through the left-opeator.
        for leftOperatorIndex in range(3):
            # Stores the coefficients for the corresponding right operator. 
            # For example, rightOperatorCoeffs[1, 5] contains the coefficient of the 5th harmonic
            # in the Fourier series

            # Loops through all the coefficients for this operator.
            for m in range(-n, n + 1):
                # Corresponds to the inhomogenous part -gamma in the ODEs.
                b[2 * fullN + n] = -self.__params.decayConstant * singleTimeFourier[leftOperatorIndex][m]

                # This stores the coefficients for the Fourier series which makes up the nth coefficient of
                # the full Fourier series for our functions. The first, second, and third portions of this contain
                # the coefficients for sigma_i sigma_-, sigma_i sigma_+, and sigma_i sigma_z respectively.
                nth_coefficient_coefficients = np.linalg.solve(M, b)

                # Fourier series for the nth coefficient, as a function of t'.
                c_m_coeffs = [
                    Fourier(
                        freq = self.__params.drivingFreq,
                        coeffs = nth_coefficient_coefficients[0 : fullN]
                    ),

                    Fourier(
                        freq = self.__params.drivingFreq,
                        coeffs = nth_coefficient_coefficients[fullN : 2*fullN]
                    ),

                    Fourier(
                        freq = self.__params.drivingFreq,
                        coeffs = nth_coefficient_coefficients[2 * fullN:]
                    ) 
                ]

                for rightOperatorIndex in range(3):
                    # Calculates the mth coefficient for this right operator at time(s) t' = t + tau.
                    c_m = c_m_coeffs[rightOperatorIndex].Evaluate(tPlusTauAxis)
                    doubleTimeCorrelations[leftOperatorIndex, rightOperatorIndex, :, :] += c_m * np.exp(1j * m * self.__params.angularFreq * tAxis)[:, np.newaxis]

        return doubleTimeCorrelations

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

        for tIndex in tqdm(range(tAxis.size),
            disable = True,
            desc = "Solving double-time correlations",
            position = 1,
            leave = False
        ):
            # Calculates the correlation, but bc of the way matrix multiplication works
            # the input has to be given indexed as [rightOperator, leftOperator], while
            # we want our final array to have input [leftOperator, rightOperator].

            # Hence, we transpose the initial conditions we put in, and then
            # transpose the final results we get out.
            correlation = integrate.solve_ivp(
                fun = self.__dynamics.EquationsOfMotion,
                t_span = (tAxis[tIndex], tAxis[tIndex] + np.max(tauAxis)),
                # We want our matrix to have its columns have the same left operator
                # (i.e. each column should act like a non-batched input),
                # so we need to tranpose the axes of the initial conditions.
                y0 = initialConds[:, :, tIndex].T.ravel(),
                method = 'DOP853',
                t_eval = tAxis[tIndex] + tauAxis,
                rtol = 1e-3,
                atol = 1e-6,
                vectorized = True,
                # Uses -<sigma_i(t)\rangle gamma_- as the inhomogenous
                # z-component.
                args = (inhomParts[:, tIndex],)
            ).y.reshape(3, 3, -1)

            # Here is where the last transpose happens. The last matrix stays the same, but the first
            # two axes are swapped, so the matrix at each time is transposed.
            doubleTimeCorrelations[:, :, tIndex, :] = np.transpose(correlation, (1, 0, 2))

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