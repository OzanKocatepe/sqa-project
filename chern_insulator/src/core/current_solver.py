import numpy as np
from scipy import integrate

from data import ModelParameters, Fourier
from operators import ParamagneticCurrentX, ParamagneticCurrentY, DiamagneticCurrentXX, band_basis_projector, hamiltonian
from LengthGauge import LengthGauge

def calculate_paramagnetic_current(
        params: ModelParameters,
        time: float | np.ndarray[float],
        fourierSeries: list[Fourier]
    ) -> np.ndarray[complex]:
    """Calculates the paramagnetic current.
    
    Parameters
    ----------
    params : ModeLParameters
        The parameters of the model.
    time : float | ndarray[float]
        The points in time, in seconds, to evaluate the paramagnetic current operator at.
    fourierSeries : list[Fourier]
        The list containing the Fourier series for sigma_-, sigma_+,
        and sigma_z, in that order.

    Returns
    -------
    ndarray[complex]:
        The value of the paramagnetic current operator at the corresponding times.
        Has shape (2, time.size), where the first dimension corresponds to the
        current in the x-dimension and y-dimension for indices 0 and 1 respectively.
    """

    current = np.zeros((2, time.size), dtype=complex)

    sigmam = fourierSeries[0].Evaluate(time)
    sigmap = fourierSeries[1].Evaluate(time)
    sigmaz = fourierSeries[2].Evaluate(time)

    jpx = ParamagneticCurrentX.lattice_basis(params, time)
    jpy = ParamagneticCurrentY.lattice_basis(params)

    basis = hamiltonian.get_band_basis(params)

    current[0, :] = (
        band_basis_projector.rotated_minus_coeff(basis, jpx) * sigmam
        + band_basis_projector.rotated_plus_coeff(basis, jpx) * sigmap
        + band_basis_projector.rotated_z_coeff(basis, jpx) * sigmaz
    )

    current[1, :] = (
        band_basis_projector.rotated_minus_coeff(basis, jpy) * sigmam
        + band_basis_projector.rotated_plus_coeff(basis, jpy) * sigmap
        + band_basis_projector.rotated_z_coeff(basis, jpy) * sigmaz
    )
 
    return current

def calculate_diamagnetic_current(
        params: ModelParameters,
        time : float | np.ndarray[float],
        fourierSeries: list[Fourier]
    ) -> np.ndarray[complex]:
    """Calculates the xx-diamagnetic current.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    time : float | ndarray[float]
        The points in time, in seconds, to evaluate the diamagnetic current operator at.
    fourierSeries : list[Fourier]
        The list containing the Fourier series for sigma_-, sigma_+,
        and sigma_z, in that order.

    Returns
    -------
    ndarray[complex]:
        The value of the diamagnetic current operator at the corresponding times.
        Has shape (time.size,), since we only need the xx-component of the diamagnetic
        current in our case.
    """

    sigmam = fourierSeries[0].Evaluate(time)
    sigmap = fourierSeries[1].Evaluate(time)
    sigmaz = fourierSeries[2].Evaluate(time)

    jdxx = DiamagneticCurrentXX.lattice_basis(params, time)
    basis = hamiltonian.get_band_basis(params)
    
    return (
        band_basis_projector.rotated_minus_coeff(basis, jdxx) * sigmam
        + band_basis_projector.rotated_plus_coeff(basis, jdxx) * sigmap
        + band_basis_projector.rotated_z_coeff(basis, jdxx) * sigmaz
    )

def calculate_total_current(
    Ax: np.ndarray[float],
    paramagnetic: np.ndarray[complex],
    diamagnetic: np.ndarray[complex]
) -> np.ndarray[complex]:
    """Calculates the total current given the paramagnetic and diamagnetic current.
    
    Parameters
    ----------
    Ax : ndarray[float], shape (n,)
        The driving vector potential calculated at each time in the tauAxis, such that
        each index here corresponds to the vector potential that was applied at the same index
        (corresponding to the same time) in the paramagentic and diamagnetic currents.
    paramagnetic : ndarray[complex], shape(2, n)
        The paramagentic current array, as stored in CurrentData and calculated in
        CalculateParamagneticCurrent.
    diamagnetic : ndarray[complex], shape(n,)
        The diamagnetic current array, as stored in CurrentData and calculated in
        CalculateDiamagneticCurrent.

    Returns
    -------
    ndarray[complex]:
        An array of shape (2, n) containing the total current in each direction. In the x-direcition
        this is a combination of the paramagnetic and diamagnetic currents. In the y-direction, this is
        just the paramagnetic current.
    """

    totalCurrent = np.zeros(paramagnetic.shape, dtype=complex)

    # Total x-current depends on paramagnetic and diamagnetic x-currents, along with vector potential.
    totalCurrent[0, :] = paramagnetic[0, :] + diamagnetic * Ax
    # Total y-current remains just the paramagnetic y-current.
    totalCurrent[1, :] = paramagnetic[1, :]

    return totalCurrent

def calculate_double_time_current(
    params: ModelParameters,
    tAxis: np.ndarray[float],
    tauAxis: np.ndarray[float],
    singleTimeFourier: list[Fourier],
    doubleTimeCorrelators: np.ndarrayp[complex]
) -> np.ndarray[complex]:
    """Calculates the double-time current correlators.

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    tAxis : ndarray[float], shape (n,)
        The tAxis, as stored in AxisData, in seconds.
    tauAxis : ndarray[float], shape(n,)
        The tauAxis, as stored in AxisData, in seconds.
    singleTimeFourier : list[Fourier]
        The single-time correlation Fourier series.
    doubleTimeCorrelators : ndarray[complex], shape(3, 3, tAxis.size, tauAxis.size)
        The double-time correlators.
    
    Returns
    -------
    ndarray[complex]
        An array of shape (2, 2, tAxis.size, tauAxis.size) containing the double-time current correlations.
        The first and second axes correspond to the direction of the current operator, with indices
        0 and 1 corresponding to the x- and y- current respectively.
        The last two axes correspond to the double-time correlator j_alpha(t) j_beta(t + tau) at times
        t and t + tau.
    """

    doubleCurrentCorrelations = np.zeros((2, 2, tAxis.size, tauAxis.size), dtype=complex)
    tPlusTauAxis = np.add.outer(tAxis, tauAxis)

    currentOperators = [
        ParamagneticCurrentX,
        ParamagneticCurrentY
    ]

    basis = hamiltonian.get_band_basis(params)

    for leftDirection in range(2):
        for rightDirection in range(2):

            # Now that we know the direction of our operators, we can calculate
            # the coefficients individually.
            firstOperatorCoefficients = np.array([
                band_basis_projector.rotated_minus_coeff(basis, currentOperators[leftDirection].lattice_basis(params, tAxis)),
                band_basis_projector.rotated_plus_coeff(basis, currentOperators[leftDirection].lattice_basis(params, tAxis)),
                band_basis_projector.rotated_z_coeff(basis, currentOperators[leftDirection].lattice_basis(params, tAxis)),
            ])

            secondOperatorCoefficients = np.array([
                band_basis_projector.rotated_minus_coeff(basis, currentOperators[rightDirection].lattice_basis(params, tPlusTauAxis.flatten())).reshape(tPlusTauAxis.shape),
                band_basis_projector.rotated_plus_coeff(basis, currentOperators[rightDirection].lattice_basis(params, tPlusTauAxis.flatten())).reshape(tPlusTauAxis.shape),
                band_basis_projector.rotated_z_coeff(basis, currentOperators[rightDirection].lattice_basis(params, tPlusTauAxis.flatten())).reshape(tPlusTauAxis.shape),
            ])

            # Now we can loop through each of the possible combinations of these coefficients,
            # and multiply by the respective connected correlator.
            for firstCoeff in range(3):
                for secondCoeff in range(3):
                    # Calculates the product term in the connected correlator.
                    prod = (singleTimeFourier[firstCoeff].Evaluate(tAxis)[:, np.newaxis]
                            * singleTimeFourier[secondCoeff].Evaluate(tPlusTauAxis))
                    # Calculates the connected correlator.
                    connected = doubleTimeCorrelators[firstCoeff, secondCoeff, :, :] - prod
                    # Multiplies the first and second coefficient, resulting in an array of shape
                    # (t.size, tau.size). Then, multiplies that by the connected correlator at each point.
                    # Adds this onto the connected current correlator for this direction, so by the end
                    # of the loop we've added on every contribution for this direction.
                    doubleCurrentCorrelations[leftDirection, rightDirection, :, :] += (
                        firstOperatorCoefficients[firstCoeff][:, np.newaxis]
                        * secondOperatorCoefficients[secondCoeff]
                        * connected
                    )

    return doubleCurrentCorrelations 

def calculate_length_gauge_current(time: float | np.ndarray[float]) -> np.ndarray[complex]:
    """Calculates the expectation value of the current in the length gauge.
    
    Parameters
    ----------
    time : float | ndarray[float]
        The points in time, in seconds, to evaluate the current operator at.

    
    Returns
    -------
    ndarray[complex]:
        The value of the length gauge current operator at the corresponding times.
        Has shape (2, time.size), where the first dimension corresponds to the
        current in the x-dimension and y-dimension for indices 0 and 1 respectively.
    """

    raise DeprecationWarning("This function is no longer supported.")

    current = np.zeros((2, time.size), dtype=complex)

    lg = LengthGauge(self.__params, self.__hamiltonian)

    # Solves the ODE for our density matrix at the desired times.
    rho = integrate.solve_ivp(
        fun = lg.DensityMatrixODE,
        t_span = (0, np.max(time)),
        y0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex),
        t_eval = time,
        rtol=1e-9,
        atol=1e-12
    ).y.T

    # Reshapes rho into a matrix rather than a flattened array.
    rho = rho.reshape((time.size, 2, 2))

    # Calculates our current operators at each desired time.
    jx = lg.jxLengthGauge(time)
    jy = lg.jyLengthGauge(time)

    # Calculates the average current.
    current[0, :] = np.trace(jx @ rho, axis1=1, axis2=2)
    current[1, :] = np.trace(jy @ rho, axis1=1, axis2=2)

    return current

def integrate_second_order_current(drivingFreq: float, tAxis: np.ndarray[float], doubleTimeCurrent: np.ndarray[complex]) -> np.ndarray[complex]:
    """Integrates the second-order current functions over the t-axis.

    Parameters
    ----------
    drivingFreq: float
        The driving frequency, in Hz, of the pumping, which is also the frequency of the system
        in steady state.
    tAxis : ndarray[float]
        The tAxis in seconds.
    doubleTimeCurrent : ndarray[complex]
        The second-order current values, of shape (2, 2, t.size, tau.size), where
        the first two axes are the left and right current direction.

    Returns
    -------
    ndarray[complex]
        The doubleTimeCurrent, integrated over the t-axis (axis 2) and divided by the driving period,
        in order to get the mean current at each tau, resulting in an array of shape (2, 2, tau.size).
    """

    return drivingFreq * np.trapezoid(
        y = doubleTimeCurrent,
        x = tAxis,
        axis = 2
    )

def calculate_spectral_noise_tensor(
    params: ModelParameters,
    tauAxis: np.ndarray[float],
    doubleTimeCurrent: np.ndarray[complex],
    n: int
) -> np.ndarray[complex]:
    """Calculates the Fourier transform at the harmonics of the second-order current w.r.t. tau.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    tauAxis : ndarray[float]
        The tauAxis in seconds.
    doubleTimeCurrent : ndarray[complex]
        The second order current, with shape (2, 2, t.size, tau.size).
    n : int
        The number of harmonics to calculate.

    Returns
    -------
    ndarray[complex]
        The spectral correlation tensor, with shape (2, 2, 2 * n + 1, t.size), where the third axis corresponds to the harmonics
        at -n to n times the driving frequency and the fourth corresponds to the t-axis. This is a function of t.
    """
    
    # Creates harmonics [omega_{-n}, ..., omega_n], where omega_i = i * angularFreq
    harmonic_freqs = np.arange(-n, n + 1) * params.angularFreq
    # Creates array of shape (2n + 1, tauAxis.size), containing exponents -1j * omega_n * tau.
    exponentials = -1j * np.multiply.outer(harmonic_freqs, tauAxis)
    exponentials = np.exp(exponentials)

    # Final shape is of integrand is (2, 2, 2n+1, t.size, tauAxis.size), corresponding to indices of correlation tensor,
    # chosen harmonic, and then the t and tau axes.
    # The correlation tensor part of the integrand is the same for all harmonics, so we insert a new axis on the harmonic axis.
    # The exponential part of the integrand is the same for all indices of the correlation tensor, and the chosen initial condition,
    # so we add axes for the correlation tensor indices and the t axis.
    integrand = doubleTimeCurrent[:, :, np.newaxis, :, :] * exponentials[np.newaxis, np.newaxis, :, np.newaxis, :]

    amps = 1 / (2 * np.arange(-n, n + 1)**3 * params.angularFreq**2 * params.decayConstant)

    return amps[np.newaxis, np.newaxis, :, np.newaxis] * np.trapezoid(
        y = integrand,
        x = tauAxis,
        axis = 4
    )

def calculate_current_fourier_coefficients(
    params: ModelParameters,
    current: np.ndarray[complex],
    tauAxisSec: np.ndarray[float],
    n : int
) -> np.ndarray[complex]:
    """Calculates the fourier coefficients for the current.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters for this model.
    current : ndarray[complex]
        An array of shape (2, t.size,) storing x- and y- currents.
    tauAxisSec : ndarray[float]
        The points on the tau axis, in seconds.
    n : int
        The maximum index to calculate.

    Returns
    -------
    ndarray[complex]
        An array of shape (2, n) which stores the fourier coefficients of the
        x- and y- current, from indices 1 to n.
    """

    # Restricts our view to only 3 driving periods, since we're already in the steady state anyways.
    time = tauAxisSec[tauAxisSec <= 3 / params.drivingFreq]

    # Forms array containing exp(i omega_m t) of shape (2n + 1, t.size).
    harmonics_terms = np.arange(1, n + 1) * params.angularFreq
    exponentials = np.exp(
        np.multiply.outer(-1j * tauAxisSec, harmonics_terms)
    )

    integrand = current[:, :, np.newaxis] * exponentials[np.newaxis, :, :]

    return (1 / 3) * params.drivingFreq * np.trapezoid(
        y = integrand,
        x = tauAxisSec,
        axis = 1
    )

def calculate_semiclassical_intracavity_field_amplitude(
    params: ModelParameters,
    current_coefficients: np.ndarray[complex],
    tau_axis_sec: np.ndarray[float],
    n: int
) -> np.ndarray[complex]:
    """Calculates the semiclassical intracavity field amplitude.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    current_coefficients : ndarray[complex]
        An array of shape (2, n) that contains the current fourier coefficients.
    tau_axis_sec : ndarray[float]
        The time axis, in seconds.
    n : int
        The maximum coefficient to calculate.

    Returns
    -------
    ndarray[complex]
        An array of shape (2, n, t) where the third axis corresponds to the indices of the first driving period
        of tau_axis_sec.
    """

    # Restricts the tau axis to the first driving period. Since our formula is periodic in time
    # with the period given by the chosen harmonic, there's no point in using more data than necessary.
    two_periods_axis = tau_axis_sec[tau_axis_sec <= 1 / params.drivingFreq]

    omega_k = np.arange(1, n + 1) * params.angularFreq
    gamma_k = np.arange(1, n + 1) * params.decayConstant

    # Using the formula from Denis' paper, this creates an array of shape (n + 1, n + 1)
    # where the first and second indices correspond to the indices labelled m and p respectively.
    denominator = gamma_k[:, np.newaxis] + 1j * np.subtract.outer(omega_k, omega_k)
    # We set the m = 0, p = 0 index to 1 temporarily to avoid a division by zero.
    # denominator[0, 0] = 1

    # The exponentials have shape (n + 1, t), corresponding to harmonics and time.
    exponentials = np.exp(
        -1j * np.multiply.outer(omega_k, two_periods_axis)
    )
    numerator = current_coefficients[:, :, np.newaxis].conj() * exponentials[np.newaxis, :, :]

    # Divides numerator by denominator. The [0, 0] term we artificially set the denominator for
    # will now just be the average current, which should always be zero (or close to it).
    # Indexing occurs b/c numerator has indices mu, p, t and denominator has indices m, p.
    # So new array has indices mu, m, p, t.
    field_amp = numerator[:, np.newaxis, :, :] / denominator[np.newaxis, :, :, np.newaxis]
    # Sum over p.
    field_amp = np.sum(field_amp, axis = 2)
    # Multiply by coefficient 1 / sqrt(omega_m).
    field_amp = -1j * (1 / np.sqrt(omega_k[np.newaxis, :, np.newaxis])) * field_amp

    return field_amp

def calculate_semiclassical_mode_population(
    params: ModelParameters,
    tau_axis_sec : np.ndarray[float],
    field_amplitude : np.ndarray[complex]
) -> np.ndarray[float]:
    """Calculates the semiclassical population of each mode.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    tau_axis_sec : ndarray[float]
        The tau axis, in seconds.
    field_amplitude : ndarray[complex]
        The semiclassical intracavity field amplitude, of shape (2, m, t),
        with axes corresponding to direction, harmonic, and time.

        Should only be defined on the first driving period of the tau axis.

    Returns
    -------
    ndarray[float]
        The semiclassical population of each mode, an array of shape (2, m).
    """

    return params.drivingFreq * np.trapezoid(
        y = np.abs(field_amplitude)**2,
        x = tau_axis_sec[tau_axis_sec <= 1 / params.drivingFreq],
        axis = 2
    )

def calculate_second_order_correlation_function(
    params: ModelParameters,
    current_coefficients: np.ndarray[complex],
) -> np.ndarray[float]:
    """Calculates the second order current correlation function, g2(0).
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    current_coefficients : ndarray[complex]
        An array of shape (2, 2m - 1) containing the Fourier coefficients
        for the first-order current, containing indices 1, ..., 2m - 1.

    Returns
    -------
    ndarray[float]
        The second-order correlation function for the current as an array
        of shape (2, m).
    """

    m = int((current_coefficients.shape[1] + 1) / 2)
    # Defines gamma m, omega m, and omega p1-p3 to each have shapes corresponding to (mu, m, p1, p2, p3).
    gamma_m = params.decayConstant * np.arange(1, m + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    omega_m = params.angularFreq * np.arange(1, m + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    omega_p1 = params.angularFreq * np.arange(1, m + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    omega_p2 = params.angularFreq * np.arange(1, m + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    omega_p3 = params.angularFreq * np.arange(1, m + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    # Calculates omega_{p1 + p2 - p3} with shape (mu, m, p1, p2, p3)
    omega_p1_p2_p3 = params.angularFreq * (
        np.arange(1, m + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        + np.arange(1, m + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        - np.arange(1, m + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )

    # Similarly, defines jp1, jp2, jp3 with the same shapes.
    jp1 = current_coefficients[:, np.newaxis, :m, np.newaxis, np.newaxis]
    jp2 = current_coefficients[:, np.newaxis, np.newaxis, :m, np.newaxis]
    jp3 = current_coefficients[:, np.newaxis, np.newaxis, np.newaxis, :m]

    # Contains the indices (p1 + p2 - p3) for shape (p1, p2, p3).
    # Offset by one since the current coefficients are zero indexed, so index 0 to m - 1
    # corresponds to coefficient 1 to m, as desired.
    p1_p2_p3 = (np.arange(0, m)[:, np.newaxis, np.newaxis]
        + np.arange(0, m)[np.newaxis, :, np.newaxis]
        - np.arange(0, m)[np.newaxis, np.newaxis, :]
    )

    # Creates array of shape (mu, m, p1, p2, p3) containing j_{p1 + p2 - p3}.
    jp1_p2_p3 = np.zeros((2, m, p1_p2_p3.shape[0], p1_p2_p3.shape[1], p1_p2_p3.shape[2]), dtype=complex)
    jp1_p2_p3[0, :, :, :, :] = current_coefficients[0, p1_p2_p3][np.newaxis, :, :, :]
    jp1_p2_p3[1, :, :, :, :] = current_coefficients[1, p1_p2_p3][np.newaxis, :, :, :]

    # Calculates each term in the numerator with the same shape.
    numerator = (
        jp1 / (gamma_m + 1j * (omega_p1 - omega_m))
    ) * (
        jp2 / (gamma_m + 1j * (omega_p2 - omega_m))
    ) * (
        jp3.conj() / (gamma_m + 1j * (omega_m - omega_p3))
    ) * (
        jp1_p2_p3.conj() / (gamma_m + 1j * (omega_m - omega_p1_p2_p3))
    )

    # Sums over p1, p2, p3. Remaining shape is (mu, m)
    numerator = np.sum(numerator, axis=(2, 3, 4))

    # Calculates denominator similarly, however much simpler.
    # Of shape (mu, m, p)
    jp = current_coefficients[:, np.newaxis, :m]
    gamma_m = params.decayConstant * np.arange(1, m + 1)[np.newaxis, :, np.newaxis]
    omega_m = params.angularFreq * np.arange(1, m + 1)[np.newaxis, :, np.newaxis]
    omega_p = params.angularFreq * np.arange(1, m + 1)[np.newaxis, np.newaxis, :]

    denominator = np.abs(jp)**2 / (
        gamma_m**2 + (omega_m - omega_p)**2
    )
    # Sums over p. Final shape is (mu, m).
    denominator = np.sum(denominator, axis = 2)

    return numerator / denominator**2