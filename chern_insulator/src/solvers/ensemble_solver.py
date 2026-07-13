import numpy as np

from data import ModelParameters, AxisData

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
        An array of shape (2, 2n + 1) which stores the fourier coefficients of the
        x- and y- current, from indices -n to n.
    """

    # Restricts our view to only 3 driving periods, since we're already in the steady state anyways.
    mask = tauAxisSec <= 3 / params.drivingFreq

    # Forms array containing exp(i omega_m t) of shape (2n + 1, t.size).
    harmonics_terms = np.arange(-n, n + 1) * params.angularFreq
    exponentials = np.exp(
        np.multiply.outer(-1j * tauAxisSec[mask], harmonics_terms)
    )

    integrand = current[:, mask, np.newaxis] * exponentials[np.newaxis, :, :]

    return (params.drivingFreq / 3) * np.trapezoid(
        y = integrand,
        x = tauAxisSec[mask],
        axis = 1
    )

def calculate_semiclassical_mode_population(
    params: ModelParameters,
    current_coefficients : np.ndarray[complex]
) -> np.ndarray[float]:
    """Calculates the semiclassical population of each mode.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    current_coefficients : ndarray[complex]
        An array of shape (2, 2m + 1) containing the Fourier coefficients
        for the first-order current, containing indices -m to m.

    Returns
    -------
    ndarray[float]
        The semiclassical population of each mode, an array of shape (2, m), containing
        n^(cl)_{1, ... m}.
    """

    # Of shape (mu, m, p)
    m = int((current_coefficients.shape[1] - 1) / 2)
    jp = current_coefficients[:, np.newaxis, :]

    gamma_m = params.decayConstant * (np.arange(1, m + 1)**2)[np.newaxis, :, np.newaxis]
    omega_m = params.angularFreq * np.arange(1, m + 1)[np.newaxis, :, np.newaxis]
    omega_p = params.angularFreq * np.arange(-m, m + 1)[np.newaxis, np.newaxis, :]

    mode_population = (params.matter_light_coupling**2 / omega_m) * np.abs(jp)**2 / (
        gamma_m**2 + (omega_m - omega_p)**2
    )
    # Sums over p. Final shape is (mu, m).
    return np.sum(mode_population, axis = 2)

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
        An array of shape (2, 6m + 1) containing the Fourier coefficients
        for the first-order current, containing indices -3m, ..., 3m.

    Returns
    -------
    ndarray[float]
        The second-order correlation function for the current as an array
        of shape (2, m) containing m = 1, ..., m.
    """

    m = int((current_coefficients.shape[1] - 1) / 6)
    # Defines gamma m, omega m, and omega p1-p3 to each have shapes corresponding to (mu, m, p1, p2, p3).
    gamma_m = params.decayConstant * (np.arange(1, m + 1)**2)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    omega_m = params.angularFreq * np.arange(1, m + 1)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    omega_p1 = params.angularFreq * np.arange(-m, m + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    omega_p2 = params.angularFreq * np.arange(-m, m + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    omega_p3 = params.angularFreq * np.arange(-m, m + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    # Calculates omega_{p1 + p2 - p3} with shape (mu, m, p1, p2, p3)
    omega_p1_p2_p3 = params.angularFreq * (
        np.arange(-m, m + 1)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        + np.arange(-m, m + 1)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
        - np.arange(-m, m + 1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )

    # Similarly, defines jp1, jp2, jp3 with the same shapes. Since p1 only go from -m to m,
    # we only include those coefficients out of the total -2m to 2m that we have,
    # since those extra ones are only for p1 + p2 - p3.
    start = -m + 3 * m
    end = m + 3 * m
    jp1 = current_coefficients[:, np.newaxis, start:end + 1, np.newaxis, np.newaxis]
    jp2 = current_coefficients[:, np.newaxis, np.newaxis, start:end + 1, np.newaxis]
    jp3 = current_coefficients[:, np.newaxis, np.newaxis, np.newaxis, start:end + 1]

    # Contains the indices (p1 + p2 - p3) for shape (p1, p2, p3).
    # We want each p1, p2, p3 to be in the range -m to m.
    p1_p2_p3 = (np.arange(-m, m + 1)[:, np.newaxis, np.newaxis]
        + np.arange(-m, m + 1)[np.newaxis, :, np.newaxis]
        - np.arange(-m, m + 1)[np.newaxis, np.newaxis, :]
    )

    # However these are indices to the current coefficients *array*,
    # so these correspond to each index must be offset by 2m in order to get the correct
    # index for the *array*.
    p1_p2_p3 += 3 * m

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

    # Sums over p1, p2, p3. Remaining shape is (mu, m).
    numerator = np.sum(numerator, axis=(2, 3, 4))

    # Calculates denominator similarly, however much simpler.
    # Of shape (mu, m). When calculating, this function only sums over the
    # given current coefficients. Could be made more accurate by giving it all the
    # coefficients and then considering only the first m denomiantors *after* calculation,
    # but it felt wrong to calculate the denominator using more terms in the sum than the numerator.
    denominator = calculate_semiclassical_mode_population(
        params,
        current_coefficients[:, start:end + 1]
    ) * omega_m[:, :, 0, 0, 0]

    return numerator / denominator**2

def calculate_squeezing_weak_laser(
    params : ModelParameters,
    time_averaged_generalised_noise_tensor : np.ndarray[complex]
) -> np.ndarray[float]:
    """Calculates the maximal squeezing at each direction and mode.
    
    Parameters
    ----------
    params : modelParameters
        The parameters of the model.
    time_averaged_generalised_noise_tensor : ndarray[complex]
        The time-averaged generalised noise correlation tensor, which should have
        shape (2, m) corresponding to direction and mode.

    Returns
    -------
    ndarray[float]
        The maximal squeezing at weak laser power at each direction and mode, as an array of shape (2, m, M), corresponding
        to direction, mode, and scattering rate.
    """

    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :, np.newaxis]
    gamma_m = params.decayConstant * (np.arange(1, params.maxN + 1)**2)[np.newaxis, :, np.newaxis]
    Q_cm = omega_m / (2 * gamma_m)

    # Defines noise tensor divided by the amplitude factor.
    # parameterised_noise_tensor = time_averaged_generalised_noise_tensor * omega_m**2 / (Q_cm * params.matter_light_coupling**2)
    parameterised_noise_tensor = (omega_m / params.matter_light_coupling**2) * time_averaged_generalised_noise_tensor

    return -10 * np.log10(
        1 + 4 * Q_cm * (params.matter_light_coupling / omega_m)**2
        * (parameterised_noise_tensor.real - np.abs(parameterised_noise_tensor) / np.sqrt(1 + 4 * Q_cm**2))
    )

def calculate_angular_momentum_operator(
        params : ModelParameters,
        current_coefficients: np.ndarray[complex]
) -> np.ndarray[complex]:
    """Calculates the angular momentum operator at each mode.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters for each model.
    current_coefficients : ndarray[complex]
        The fourier coefficients of the current, with shape (2, 2n + 1) containing the indices from
        -n to n.

    Returns
    -------
    ndarray[complex]
        An array of shape (n,) containing the angular momentum at modes 1 to n.
    """

    # Use shape (m, p)
    p = (current_coefficients.shape[1] - 1) // 2
    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[:, np.newaxis]
    gamma_m = params.decayConstant * (np.arange(1, params.maxN + 1)**2)[:, np.newaxis]
    omega_p = params.angularFreq * np.arange(-p, p + 1)[np.newaxis, :]

    summand = (4 * gamma_m * current_coefficients[0, np.newaxis, :] * current_coefficients[1, np.newaxis, :].conj()
        / ( (omega_p**2 + gamma_m**2 - omega_m**2)**2 + 4 * gamma_m**2 * omega_m**2 )
    )

    return np.sum(summand, axis=1)

def calculate_spectral_noise_tensor(
    params: ModelParameters,
    tauAxis: np.ndarray[float],
    doubleTimeCurrent: np.ndarray[complex],
    n: int
) -> np.ndarray[complex]:
    """Calculates the spectral noise tensor.
    
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
        The spectral correlation tensor, with shape (2, 2, n, t.size), where the third axis corresponds to the harmonics
        at 1 to n and the fourth corresponds to the t-axis. This is a function of t.
    """
    
    # Creates harmonics [omega_{-n}, ..., omega_n], where omega_i = i * angularFreq
    omega_m = np.arange(1, n + 1) * params.angularFreq
    # Creates array of shape (2n + 1, tauAxis.size), containing exponents -1j * omega_n * tau.
    exponentials = -1j * np.multiply.outer(omega_m, tauAxis)
    exponentials = np.exp(exponentials)

    # Final shape is of integrand is (2, 2, n, t.size, tauAxis.size), corresponding to indices of correlation tensor,
    # chosen harmonic, and then the t and tau axes.
    # The correlation tensor part of the integrand is the same for all harmonics, so we insert a new axis on the harmonic axis.
    # The exponential part of the integrand is the same for all indices of the correlation tensor, and the chosen initial condition,
    # so we add axes for the correlation tensor indices and the t axis.
    integrand = doubleTimeCurrent[:, :, np.newaxis, :, :] * exponentials[np.newaxis, np.newaxis, :, np.newaxis, :]

    gamma_m = np.arange(1, n + 1)**2 * params.decayConstant
    amps = params.matter_light_coupling**2 / omega_m

    return amps[np.newaxis, np.newaxis, :, np.newaxis] * np.trapezoid(
        y = integrand,
        x = tauAxis,
        axis = 4
    )

def calculate_dc_population_variance(
    params: ModelParameters,
    tauAxis: np.ndarray[float],
    tAxis: np.ndarray[float],
    doubleTimeCurrent: np.ndarray[complex],
    n: int
) -> np.ndarray[complex]:
    """Calculates the dc population variance in a very similar method to the spectral noise tensor.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    tauAxis : ndarray[float]
        The tauAxis in seconds.
    tAxis : ndarray[float]
        The tAxis in seconds.
    doubleTimeCurrent : ndarray[complex]
        The second order current, with shape (2, 2, t.size, tau.size).
    n : int
        The number of harmonics to calculate.

    Returns
    -------
    ndarray[complex]
        The dc population variance, with shape (2, 2, n), where the third axis corresponds to the harmonics
        at 1 to n.
    """
    
    # Creates harmonics [omega_{-n}, ..., omega_n], where omega_i = i * angularFreq
    omega_m = np.arange(1, n + 1) * params.angularFreq
    # Creates array of shape (2n + 1, tauAxis.size), containing exponents -1j * omega_n * tau.
    exponentials = -1j * np.multiply.outer(omega_m, tauAxis)
    exponentials = np.exp(exponentials)

    # Final shape is of integrand is (2, 2, n, t.size, tauAxis.size), corresponding to indices of correlation tensor,
    # chosen harmonic, and then the t and tau axes.
    # The correlation tensor part of the integrand is the same for all harmonics, so we insert a new axis on the harmonic axis.
    # The exponential part of the integrand is the same for all indices of the correlation tensor, and the chosen initial condition,
    # so we add axes for the correlation tensor indices and the t axis.

    averaged_current = params.drivingFreq * np.trapezoid(
        y = doubleTimeCurrent,
        x = tAxis,
        axis = 2
    )

    integrand = averaged_current[:, :, np.newaxis, :] * exponentials[np.newaxis, np.newaxis, :, :]

    gamma_m = np.arange(1, n + 1)**2 * params.decayConstant
    amps = params.matter_light_coupling**2 / (2 * gamma_m * omega_m)

    return amps[np.newaxis, np.newaxis, :] * np.trapezoid(
        y = integrand,
        x = tauAxis,
        axis = 3
    )

def calculate_generalised_noise_tensor(
        params: ModelParameters,
        axes : AxisData,
        spectral_noise_tensor: np.ndarray[complex],
        diamagnetic_current: np.ndarray[complex]
) -> np.ndarray[complex]:
    """Calculates the generalised noise tensor using the current correlation tensor.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    axes : AxisData
        The axes of the model.
    spectral_noise_tensor : ndarray[complex]
        An array of shape (2, 2, n, t), with the axes corresponding to the first and second indices,
        the harmonic, and the time t. This should be the resulting tensor from averaging over the entire Brillouin Zone.
    diamagnetic_current : ndarray[complex]
        An array of shape (2, t) where the first axis corresponds to xx or yy and the second corresponds to time t.
        This is the expectation of the total diamagnetic current, averaged over the BZ.
    
    Returns
    -------
    ndarray[complex]
        The generalised noise tensor, with shape (2, n, t) corresponding to direction, harmonic,
        and time respectively.
    """

    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :, np.newaxis]
    gamma_m = params.decayConstant * (np.arange(1, params.maxN + 1)**2)[np.newaxis, :, np.newaxis]
    Q_cm = omega_m / (2 * gamma_m)

    # Creates index which will select (0, 0) and (1, 1) from first two indices of spectral_noise_tensor.
    idx = np.arange(2)
    
    # Offsets t-axis so that we find interpolated *steady state* diamagnetic current in next step,
    # rather than finding the transient diamagnetic current.
    steady_state_t_axis = axes.t_axis_sec + axes.tau_axis_sec[-1000]

    # Interpolates the diamagnetic current to find its values within one steady state (i.e. at points along the t-axis).
    # This makes sure it has the right shape to broadcast with spectral_noise_tensor.
    interpolated_diamagnetic_current = diamagnetic_current[:, 
        np.around(steady_state_t_axis * axes.tau_axis_sec.size / np.max(axes.tau_axis_sec), 0).astype(int)
    ]
    
    return ( (params.matter_light_coupling**2 / omega_m) * 0.5j * interpolated_diamagnetic_current[:, np.newaxis, :]
        + spectral_noise_tensor[idx, idx, :, :] )

def calculate_squeezing(
        params: ModelParameters,
        axes : AxisData,
        generalised_noise_tensor: np.ndarray[complex]
) -> np.ndarray[float]:
    """Calculates the squeezing of the system.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    axes : AxisData
        The axes of the model.
    generalised_noise_tensor : ndarray[complex]
        The generalised noise tensor of the system, with shape (2, n, t) corresponding to direction,
        harmonic, and time.

    Returns
    -------
    ndarray[float]
        The squeezing as an array of shape (2, n) in each direction and at each harmonic.
    """
    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :]
    gamma_m = params.decayConstant * (np.arange(1, params.maxN + 1)**2)[np.newaxis, :]
    Q_cm = omega_m / (2 * gamma_m)

    averaged_noise_tensor = params.drivingFreq * np.trapezoid(
        y = generalised_noise_tensor,
        x = axes.t_axis_sec,
        axis = 2
    )

    parameterised_noise_tensor = (omega_m / params.matter_light_coupling**2) * averaged_noise_tensor

    return -10 * np.log10(
        1 + 4 * Q_cm * (params.matter_light_coupling / omega_m)**2 * (parameterised_noise_tensor.real - np.abs(parameterised_noise_tensor) / np.sqrt(1 + 4 * Q_cm**2))
    )

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