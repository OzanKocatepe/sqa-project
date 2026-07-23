import numpy as np

from data import Fourier, ModelParameters, band_basis
from physics import ParamagneticCurrentX, ParamagneticCurrentY, DiamagneticCurrentXX, DiamagneticCurrentYY, hamiltonian

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
        band_basis.rotated_minus_coeff(basis, jpx) * sigmam
        + band_basis.rotated_plus_coeff(basis, jpx) * sigmap
        + band_basis.rotated_z_coeff(basis, jpx) * sigmaz
    )

    current[1, :] = (
        band_basis.rotated_minus_coeff(basis, jpy) * sigmam
        + band_basis.rotated_plus_coeff(basis, jpy) * sigmap
        + band_basis.rotated_z_coeff(basis, jpy) * sigmaz
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
        band_basis.rotated_minus_coeff(basis, jdxx) * sigmam
        + band_basis.rotated_plus_coeff(basis, jdxx) * sigmap
        + band_basis.rotated_z_coeff(basis, jdxx) * sigmaz,

        band_basis.rotated_minus_coeff(basis, jdyy) * sigmam
        + band_basis.rotated_plus_coeff(basis, jdyy) * sigmap
        + band_basis.rotated_z_coeff(basis, jdyy) * sigmaz
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
    basis = hamiltonian.get_band_basis(params)
    x_current_operator = band_basis.rotate_to_band_basis(basis, ParamagneticCurrentX.lattice_basis(params, 0))
    y_current_operator = band_basis.rotate_to_band_basis(basis, ParamagneticCurrentY.lattice_basis(params, 0))

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
    summand_no_factor = (
        (2 * scattering_rate * off_diagonal_current)
        / (scattering_rate**2 + (2 * hamiltonian.energy(params) + omega_m)**2)
    )

    # Multiplies by the front amplitude factors to get the population variance. Same shape as above.
    dc_population_variance = Q_cm * (params.matter_light_coupling / omega_m)**2 * summand_no_factor
    real_noise_correlation_tensor = (params.matter_light_coupling**2 / (2 * omega_m)) * summand_no_factor

    return dc_population_variance.squeeze(), real_noise_correlation_tensor.squeeze()

def calculate_double_time_current(
    params: ModelParameters,
    tAxis: np.ndarray[float],
    tauAxis: np.ndarray[float],
    singleTimeFourier: list[Fourier],
    doubleTimeCorrelators: np.ndarray[complex]
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
        An array of shape (2, 2, tAxis.size, tauAxis.size) containing the second order current correlations,
        <j(t + tau) j(t)>. The first and second axes correspond to the direction of the current operator,
        with indices 0 and 1 corresponding to the x- and y- current respectively.
        The last two axes correspond to the values of t and tau respectively.
    ndarray[complex]
        The same shape as above, but containing the <j(t + tau)> <j(t)> product term.
    ndarray[complex]
        The difference between the two terms above, <j(t + tau) j(t)> - <j (t + tau)> <j(t)>,
        which is the matter correlation tensor.
    """

    second_order_current = np.zeros((2, 2, tAxis.size, tauAxis.size), dtype=complex)
    current_product = np.zeros((2, 2, tAxis.size, tauAxis.size), dtype=complex)
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
                band_basis.rotated_minus_coeff(basis, currentOperators[leftDirection].lattice_basis(params, tAxis)),
                band_basis.rotated_plus_coeff(basis, currentOperators[leftDirection].lattice_basis(params, tAxis)),
                band_basis.rotated_z_coeff(basis, currentOperators[leftDirection].lattice_basis(params, tAxis)),
            ])

            secondOperatorCoefficients = np.array([
                band_basis.rotated_minus_coeff(basis, currentOperators[rightDirection].lattice_basis(params, tPlusTauAxis.flatten())).reshape(tPlusTauAxis.shape),
                band_basis.rotated_plus_coeff(basis, currentOperators[rightDirection].lattice_basis(params, tPlusTauAxis.flatten())).reshape(tPlusTauAxis.shape),
                band_basis.rotated_z_coeff(basis, currentOperators[rightDirection].lattice_basis(params, tPlusTauAxis.flatten())).reshape(tPlusTauAxis.shape),
            ])

            # Now we can loop through each of the possible combinations of these coefficients,
            # and multiply by the respective connected correlator.
            for firstCoeff in range(3):
                for secondCoeff in range(3):
                    # Calculates the product term in the connected correlator.
                    correlation_product = (singleTimeFourier[firstCoeff].Evaluate(tAxis)[:, np.newaxis]
                            * singleTimeFourier[secondCoeff].Evaluate(tPlusTauAxis))

                    # Multiplies the first and second coefficient, resulting in an array of shape
                    # (t.size, tau.size). Then, multiplies that by the connected correlator at each point.
                    # Adds this onto the connected current correlator for this direction, so by the end
                    # of the loop we've added on every contribution for this direction.
                    second_order_current[leftDirection, rightDirection, :, :] += (
                        firstOperatorCoefficients[firstCoeff][:, np.newaxis]
                        * secondOperatorCoefficients[secondCoeff]
                        * doubleTimeCorrelators[firstCoeff, secondCoeff, :, :]
                    )

                    current_product[leftDirection, rightDirection, :, :] += (
                        firstOperatorCoefficients[firstCoeff][:, np.newaxis]
                        * secondOperatorCoefficients[secondCoeff]
                        * correlation_product
                    )

    # Corrects the time ordering in both cases.
    second_order_current = np.swapaxes(second_order_current, 0, 1).conj()
    current_product = np.swapaxes(current_product, 0, 1)
    matter_correlation_tensor = second_order_current - current_product

    return second_order_current, current_product, matter_correlation_tensor

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
    basis = hamiltonian.get_band_basis(params)
    x_current_operator = band_basis.rotate_to_band_basis(basis, ParamagneticCurrentX.lattice_basis(params, 0))
    y_current_operator = band_basis.rotate_to_band_basis(basis, ParamagneticCurrentY.lattice_basis(params, 0))

    # Calculates the j^{-+}j^{+-} term. Since these are just scalars they commute, so doesn't really matter
    # which is which - they are just the off-diagonal elements.
    off_diagonal_current = np.array([
        np.abs(x_current_operator[0, 1])**2,
        np.abs(y_current_operator[0, 1])**2
    ])[:, np.newaxis, np.newaxis]

    omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :, np.newaxis]

    # Should have shape (mu, m, gamma).
    return (params.matter_light_coupling**2 / omega_m) * (
        (-(2 * hamiltonian.energy(params) + omega_m) * off_diagonal_current)
        / (scattering_rate**2 + (2 * hamiltonian.energy(params) + omega_m)**2)
    ).squeeze()

def calculate_weak_laser_noise_tensor(
    params: ModelParameters,
    scattering_rates: np.ndarray[float],
    real_component: np.ndarray[complex],
) -> np.ndarray[complex]:

        omega_m = params.angularFreq * np.arange(1, params.maxN + 1)[np.newaxis, :, np.newaxis]

        basis = hamiltonian.get_band_basis(params)
        undriven_diamagnetic_current = (params.matter_light_coupling**2 / omega_m) * np.array([
            band_basis.rotate_to_band_basis(basis, DiamagneticCurrentXX.lattice_basis(params, 0))[1, 1],
            band_basis.rotate_to_band_basis(basis, DiamagneticCurrentYY.lattice_basis(params, 0))[1, 1]
        ])[:, np.newaxis, np.newaxis]

        imaginary_component = imaginary_time_avg_generalised_noise_correlation_tensor_weak_laser(
            params,
            scattering_rates
        )

        return (
            real_component
            + 1j * (0.5 * undriven_diamagnetic_current + imaginary_component)
        )