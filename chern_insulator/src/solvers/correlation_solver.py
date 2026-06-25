import numpy as np
import diffrax
import jax
import jax.numpy as jnp

from data import ModelParameters, Fourier
from physics import hamiltonian, band_basis_projector
from ..physics import dynamics

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

    band_basis = hamiltonian.get_band_basis(params)

    # Builds the convolution matrices for Hm, Hp, Hz.
    HmConv = Fourier(
        params.drivingFreq,
        band_basis_projector.rotated_minus_coeff(band_basis, hamiltonian.lattice_fourier_coefficient(params, np.arange(-n, n + 1)))
    ).BuildConvolutionMatrix()

    HpConv = Fourier(
        params.drivingFreq,
        band_basis_projector.rotated_plus_coeff(band_basis, hamiltonian.lattice_fourier_coefficient(params, np.arange(-n, n + 1)))
    ).BuildConvolutionMatrix()

    HzConv = Fourier(
        params.drivingFreq,
        band_basis_projector.rotated_z_coeff(band_basis, hamiltonian.lattice_fourier_coefficient(params, np.arange(-n, n + 1)))
    ).BuildConvolutionMatrix()

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
    M = single_time_fourier_matrix(params)

    # Creates the right hand side of the equation Mx = b.
    b = np.zeros((3 * fullN), dtype=complex)
    # Corresponds to the inhomogenous part -gamma in the ODEs.
    b[2 * fullN + n] = -params.decayConstant

    sigmaCoeffs = np.linalg.solve(M, b)

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

def solve_double_time_correlations(
        params: ModelParameters,
        tAxis: np.ndarray[float],
        tauAxis: np.ndarray[float],
        singleTimeFourier: list[Fourier]
    ) -> np.ndarray[complex]:
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

    # The following code is very messy, but it speeds up the code by like 20% so I'm leaving it
    # until I can clean it up.

    def rhs(t, c, inhom_part):
        return dynamics.equation_of_motion(t, c, inhom_part, params)

    def solve_single_t_point(initialConditions, inhom_part, initialTime):
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
    solve_all = jax.vmap(solve_single_t_point)
    results = solve_all(
        initialConds.transpose(2, 1, 0).reshape(tAxis.size, 9, 1),
        inhomParts.T,
        tAxis
    )

    doubleTimeCorrelations = np.array(results.reshape(tAxis.size, tauAxis.size, 3, 3).transpose(3, 2, 0, 1))
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