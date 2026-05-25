import numpy as np
import diffrax
import jax
import jax.numpy as jnp

from data import ModelParameters, Fourier, DoubleTimeODEParams
from operators import hamiltonian
from . import Dynamics

def single_time_fourier_matrix(params: ModelParameters) -> np.ndarray[complex]:
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
    n = params.maxN
    fullN = 2 * n + 1
    gamma = params.decayConstant

    # Creates the empty matrix.
    M = np.zeros((3 * fullN, 3 * fullN), dtype=complex)

    # Convolution matrix for taking the derivative of a series.
    deriv = 1j * params.angularFreq * np.diag(np.arange(-n, n + 1))

    # Builds the convolution matrices for Hm, Hp, Hz.
    HmConv = Fourier(params.drivingFreq, hamiltonian.fourier_minus(np.arange(-n, n + 1))) \
        .BuildConvolutionMatrix()

    HpConv = Fourier(params.drivingFreq, hamiltonian.fourier_plus(np.arange(-n, n + 1))) \
        .BuildConvolutionMatrix()

    HzConv = Fourier(params.drivingFreq, hamiltonian.fourier_z(np.arange(-n, n + 1))) \
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
    
def solve_single_time_correlations(params: ModelParameters) -> list[Fourier]:
    """Solves the single time correlations as Fourier series.
    
    Returns
    -------
    list[Fourier]:
        A list of the Fourier series for sigma_-, sigma_+, sigma_z respectively.
    """

    n = params.maxN
    fullN = 2 * n + 1
    M = single_time_fourier_matrix()

    # Creates the right hand side of the equation Mx = b.
    b = np.zeros((3 * fullN), dtype=complex)
    # Corresponds to the inhomogenous part -gamma in the ODEs.
    b[2 * fullN + n] = -params.decayConstant

    sigmaCoeffs = np.linalg.solve(M, b)

    # plt.plot(sigmaCoeffs[0:fullN] - np.conjugate(sigmaCoeffs[fullN:2 * fullN]))
    # plt.title(r"$\sigma_-$ coeffs - $\sigma_+^*$ coeffs.")
    # plt.show()

    return [
        Fourier(
            freq = params.drivingFreq,
            coeffs = sigmaCoeffs[0 : fullN]
        ),

        Fourier(
            freq = params.drivingFreq,
            # coeffs = np.conjugate(sigmaCoeffs[0: fullN])
            coeffs = sigmaCoeffs[fullN : 2*fullN]
        ),

        Fourier(
            freq = params.drivingFreq,
            coeffs = sigmaCoeffs[2 * fullN:]
        ) 
    ]

def solve_double_time_correlations_fourier(tAxis: np.ndarray[float], tauAxis: np.ndarray[float], singleTimeFourier: list[Fourier]) -> np.ndarray[complex]:
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

    raise PendingDeprecationWarning("SolveDoubleTimeCorrelationsFourier is incomplete and may be deleted.")

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

def solve_double_time_correlations(params: ModelParameters, tAxis: np.ndarray[float], tauAxis: np.ndarray[float], singleTimeFourier: list[Fourier]) -> np.ndarray[complex]:
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

    tAxis = jnp.array(tAxis)
    tauAxis = jnp.array(tauAxis)

    doubleTimeCorrelations = jnp.zeros((3, 3, tAxis.size, tauAxis.size), dtype=complex)
    # Calculates the inhomogenous parts as an array of shape (3, tAxis.size),
    # with first axis corresponding to the left-operator we chose.
    # Remember inhom part of equation is -gamma * <sigma_i(t)>
    inhomParts = -params.decayConstant * jnp.array([
        singleTimeFourier[0].Evaluate(tAxis),
        singleTimeFourier[1].Evaluate(tAxis),
        singleTimeFourier[2].Evaluate(tAxis)
    ])

    initialConds = double_time_initial_conditions(tAxis, singleTimeFourier)
    odeParams = DoubleTimeODEParams.build_from_params(params, hamiltonian)

    # The following code is very messy, but it speeds up the code by like 20% so I'm leaving it
    # until I can clean it up.

    def rhs(t, c, inhom_part):
        return Dynamics.EquationsOfMotion(t, c, inhom_part, odeParams)

    def SolveSingleTPoint(initialConditions, inhom_part, initialTime):
        return diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Dopri8(),
            t0 = initialTime,
            t1 = initialTime + jnp.max(tauAxis),
            dt0 = None,
            y0 = initialConditions,
            args = inhom_part,
            saveat = diffrax.SaveAt(ts = initialTime + tauAxis),
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        ).ys

    # vmap over all the taxis initial points.
    solve_all = jax.vmap(SolveSingleTPoint)
    results = solve_all(
        initialConds.transpose(2, 1, 0).reshape(tAxis.size, 9, 1),
        inhomParts.T,
        tAxis
    )

    doubleTimeCorrelations = np.array(results.reshape(tAxis.size, tauAxis.size, 3, 3).transpose(3, 2, 0, 1))

    # for tIndex in tqdm(range(tAxis.size),
    #     disable = True,
    #     desc = "Solving double-time correlations",
    #     position = 1,
    #     leave = False
    # ):
    #     # Calculates the correlation, but bc of the way matrix multiplication works
    #     # the input has to be given indexed as [rightOperator, leftOperator], while
    #     # we want our final array to have input [leftOperator, rightOperator].

    #     # Hence, we transpose the initial conditions we put in, and then
    #     # transpose the final results we get out.
    #     correlation = integrate.solve_ivp(
    #         fun = self.__dynamics.EquationsOfMotion,
    #         t_span = (tAxis[tIndex], tAxis[tIndex] + np.max(tauAxis)),
    #         # We want our matrix to have its columns have the same left operator
    #         # (i.e. each column should act like a non-batched input),
    #         # so we need to tranpose the axes of the initial conditions.
    #         y0 = initialConds[:, :, tIndex].T.ravel(),
    #         method = 'DOP853',
    #         t_eval = tAxis[tIndex] + tauAxis,
    #         rtol = 1e-3,
    #         atol = 1e-6,
    #         vectorized = True,
    #         max_step = 1 / (20 * self.__params.drivingFreq),
    #         # Uses -<sigma_i(t)\rangle gamma_- as the inhomogenous
    #         # z-component.
    #         args = (inhomParts[:, tIndex],)
    #     ).y.reshape(3, 3, -1)

    #     # Here is where the last transpose happens. The last matrix stays the same, but the first
    #     # two axes are swapped, so the matrix at each time is transposed.
    #     doubleTimeCorrelations[:, :, tIndex, :] = np.transpose(correlation, (1, 0, 2))

    return doubleTimeCorrelations

def double_time_initial_conditions(t: np.ndarray[float], singleTimeFourier: list[Fourier]) -> np.ndarray[complex]:
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

    return jnp.array([
        # Left operator is sigma_-
        [
            np.zeros(t.size, dtype=jnp.complex64),
            -0.5 * (sigmaz - 1),
            sigmam
        ],

        # Left operator is sigma_+
        [
            0.5 * (sigmaz + 1),
            np.zeros(t.size, dtype=jnp.complex64),
            -sigmap
        ],

        # Left operator is sigma_z
        [
            -sigmam,
            sigmap,
            np.ones(t.size, dtype=jnp.complex64)
        ]
    ])