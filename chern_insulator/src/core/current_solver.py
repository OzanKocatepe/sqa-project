import numpy as np
from scipy import integrate

from data import ModelParameters, Fourier, AxisData
from operators import ParamagneticCurrentX, ParamagneticCurrentY, DiamagneticCurrentXX, DiamagneticCurrentYY, band_basis_projector, hamiltonian
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
        Has shape (2, time.size,), giving the xx- and yy- diamagnetic current at all times.
    """

    sigmam = fourierSeries[0].Evaluate(time)
    sigmap = fourierSeries[1].Evaluate(time)
    sigmaz = fourierSeries[2].Evaluate(time)

    jdxx = DiamagneticCurrentXX.lattice_basis(params, time)
    jdyy = DiamagneticCurrentYY.lattice_basis(params, time)
    basis = hamiltonian.get_band_basis(params)
    
    return np.array([
        band_basis_projector.rotated_minus_coeff(basis, jdxx) * sigmam
        + band_basis_projector.rotated_plus_coeff(basis, jdxx) * sigmap
        + band_basis_projector.rotated_z_coeff(basis, jdxx) * sigmaz,

        band_basis_projector.rotated_minus_coeff(basis, jdyy) * sigmam
        + band_basis_projector.rotated_plus_coeff(basis, jdyy) * sigmap
        + band_basis_projector.rotated_z_coeff(basis, jdyy) * sigmaz
    ])

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
    totalCurrent[0, :] = paramagnetic[0, :] + diamagnetic[0, :] * Ax
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
    amps = params.matter_light_coupling**2 / (2 * gamma_m * omega_m)

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

def calculate_dc_population_variance_weak_laser_power(
        params: ModelParameters,
        scattering_rate: float | np.ndarray[float]
) -> np.ndarray[complex]:
    """Calculates the dc population variance at some weak laser power.

    Note that this must be calculated at each momentum individually, since
    the summand is non-linear in the momentum.

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    scattering_rate : float | ndarray[float]
        The scattering rate of the system. Can be given as a float or a vector
        of scattering rates. Note that this is the value of gamma_M, NOT the
        parameterised value of gamma_M / Delta.

    Returns
    -------
    ndarray[complex]
        The dc population variance at weak laser power for m = 1, ..., maxN. Has shape (2, m), where the
        first axis corresponds to direction and m corresponds to mode. If scattering_rate is given as a vector,
        then the shape is (2, m, scattering_rate.size).
    ndarray[complex]
        The same as above, but without the amplitude factors out front in equation (25), so that this becomes the
        real part of the generalised noise correlation tensor at weak laser power.
    """

    scattering_rate = np.atleast_1d(scattering_rate)[np.newaxis, np.newaxis, :]

    # Calculates the x- and y- paramagnetic current operators in the band basis.
    band_basis = hamiltonian.get_band_basis(params)
    x_current_operator = band_basis_projector.rotate_to_band_basis(band_basis, ParamagneticCurrentX.lattice_basis(params, 0))
    y_current_operator = band_basis_projector.rotate_to_band_basis(band_basis, ParamagneticCurrentY.lattice_basis(params, 0))

    # Calculates the j^{-+}j^{+-} term. Since these are just scalars they commute, so doesn't really matter
    # which is which - they are just the off-diagonal elements.
    off_diagonal_current = np.array([
        np.abs(x_current_operator[0, 1])**2,
        np.abs(y_current_operator[0, 1])**2
    ])[:, np.newaxis, np.newaxis]

    gamma_m = params.decayConstant * (np.arange(1, params.maxN + 1)**2)[np.newaxis, :, np.newaxis]
    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :, np.newaxis]
    Q_cm = omega_m / (2 * gamma_m)

    # Has shape (mu, m, M).
    real_noise_correlation_tensor = (
        (2 * scattering_rate * off_diagonal_current)
        / (scattering_rate**2 + (2 * hamiltonian.energy(params) + omega_m)**2)
    )

    # Multiplies by the front amplitude factors to get the population variance. Same shape as above.
    dc_population_variance = Q_cm * (params.matter_light_coupling / omega_m)**2 * real_noise_correlation_tensor

    return dc_population_variance.squeeze(), real_noise_correlation_tensor.squeeze()

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

def imaginary_time_avg_generalised_noise_correlation_tensor_weak_laser(
    params : ModelParameters,
    scattering_rate : float | np.ndarray[float]
) -> np.ndarray[complex]:
    """
    Calculates the inner sum term of the imaginary part of the time-averaged generalised noise correlation tensor.
    Does not actually return the full imaginary part of the tensor - this is calculated per-momentum right
    outside this function and saved into current data, and the mean over momentums is taken at the end.

    Parameters
    ----------
    params : ModeLparameters
        The parameters of the model.
    scattering_rate : float | ndarray[float]
        The scattering rate of the system. Can be given as a float or a vector
        of scattering rates. Note that this is the value of gamma_M, NOT the
        parameterised value of gamma_M / Delta.

    Returns
    -------
    ndarray[complex]
        The relevant inner sum in equation (31) with shape (2, m). If scattering rate is
        given as a vector, then has shape (2, m, scattering_rate.size).
    """

    scattering_rate = np.atleast_1d(scattering_rate)[np.newaxis, np.newaxis, :]

    # Calculates the x- and y- paramagnetic current operators in the band basis.
    band_basis = hamiltonian.get_band_basis(params)
    x_current_operator = band_basis_projector.rotate_to_band_basis(band_basis, ParamagneticCurrentX.lattice_basis(params, 0))
    y_current_operator = band_basis_projector.rotate_to_band_basis(band_basis, ParamagneticCurrentY.lattice_basis(params, 0))

    # Calculates the j^{-+}j^{+-} term. Since these are just scalars they commute, so doesn't really matter
    # which is which - they are just the off-diagonal elements.
    off_diagonal_current = np.array([
        np.abs(x_current_operator[0, 1])**2,
        np.abs(y_current_operator[0, 1])**2
    ])[:, np.newaxis, np.newaxis]

    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :, np.newaxis]

    # Should have shape (mu, m, gamma).
    return (
        (-(2 * hamiltonian.energy(params) + omega_m) * off_diagonal_current)
        / (scattering_rate**2 + (2 * hamiltonian.energy(params) + omega_m)**2)
    ).squeeze()

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
    parameterised_noise_tensor = time_averaged_generalised_noise_tensor

    return -10 * np.log10(
        1 + 4 * Q_cm * (params.matter_light_coupling / omega_m)**2
        * (parameterised_noise_tensor.real - np.abs(parameterised_noise_tensor) / np.sqrt(1 + 4 * Q_cm**2))
    )

def calculate_weak_laser_noise_tensor(
    params: ModelParameters,
    scattering_rates: np.ndarray[float],
    real_component: np.ndarray[complex],
) -> np.ndarray[complex]:

        band_basis = hamiltonian.get_band_basis(params)
        undriven_diamagnetic_current = np.array([
            band_basis_projector.rotate_to_band_basis(band_basis, DiamagneticCurrentXX.lattice_basis(params, 0))[1, 1],
            band_basis_projector.rotate_to_band_basis(band_basis, DiamagneticCurrentYY.lattice_basis(params, 0))[1, 1]
        ])[:, np.newaxis, np.newaxis]

        imaginary_component = imaginary_time_avg_generalised_noise_correlation_tensor_weak_laser(
            params,
            scattering_rates
        )

        return (
            real_component
            + 1j * (undriven_diamagnetic_current + imaginary_component)
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

    # Interpolates the diamagnetic current to find its values within one steady state (i.e. at points along the t-axis).
    # This makes sure it has the right shape to broadcast with spectral_noise_tensor.
    interpolated_diamagnetic_current = diamagnetic_current[:, 
        np.around(axes.t_axis_sec * axes.tau_axis_sec.size / np.max(axes.tau_axis_sec), 0).astype(int)
    ]

    return (Q_cm * (params.matter_light_coupling / omega_m)**2 * 0.5j * interpolated_diamagnetic_current[:, np.newaxis, :]
        + spectral_noise_tensor[idx, idx, :, :])

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

    return -10 * np.log10(
        1 + 4 * (averaged_noise_tensor.real - np.abs(averaged_noise_tensor) / np.sqrt(1 + 4 * Q_cm**2))
    )