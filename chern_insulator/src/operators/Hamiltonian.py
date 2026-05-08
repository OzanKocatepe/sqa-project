import numpy as np
from functools import cache, cached_property
from scipy import special

from data import ModelParameters
from operators.BandBasisProjector import BandBasisProjector
from operators.Operator import Operator

class Hamiltonian(Operator):
    """The Hamiltonian operator."""

    sigmax = np.array([[0, 1],
                       [1, 0]], dtype=complex)
    
    sigmay = np.array([[0, -1j],
                       [1j, 0]], dtype=complex)

    sigmaz = np.array([[1, 0],
                       [0, -1]], dtype=complex)

    def __init__(self, params: ModelParameters):
        """Initialises the Hamiltonian for a given model.        

        Parameters
        ----------
        params : ModelParameters
            The parameters of the model for which
            the Hamiltonian will be calculated.
        """

        self._params = params
        self._hamiltonian = self

    def Ax(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the value of the driving field in the x-direction at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | ndarray[float]:
            The value of the driving field in the x-direction at time(s) t.
            The type returned is the same as the type of t.
        """

        return self._params.drivingAmp * np.sin(self._params.angularFreq * t)

    def hx(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_x in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_x in the undriven Hamiltonian.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_x in the driven Hamiltonian at time(s) t.
            The type returned is the same as the type of t, if given.
            If no t is given, returns a float.
        """

        return np.sin(self._params.kx - self.Ax(t))
 
    @cache   
    def hy(self) -> float:
        """
        Returns the coefficient of sigma_y in the Hamiltonian.
        This result is cached, since we are only driving in the x-direction and hence
        this component has no dependence on time, and the momentum doesn't change once given.
        
        Returns
        -------
        float:
            The coefficient of sigma_y in the Hamiltonian.
        """

        return np.sin(self._params.ky)
    
    def hz(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_z in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_z in the undriven Hamiltonian.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_z in the driven Hamiltonian at time(s) t.
            The type returned is the same as the type of t, if given.
            If no t is given, returns a float.
        """

        return self._params.delta + np.cos(self._params.kx - self.Ax(t)) + np.cos(self._params.ky)
    
    @cache
    def energy(self) -> float:
        """
        Returns the unperturbed energy of the system at this momentum point.
        This result is cached, since the unperturbed energy has no dependence on time,
        and the momentum doesn't change once given.

        Returns
        -------
        float:
            The energy of the unperturbed system at this momentum.
        """

        return np.sqrt(self.hx()**2 + self.hy()**2 + self.hz()**2)

    def lattice_basis(self, t: float | np.ndarray[float]=0) -> np.ndarray[complex]:
        """
        Gets the value of the Hamiltonian in the lattice basis at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, to evaluate the Hamiltonian at.
            If zero, this is the unperturbed Hamiltonian.

        Returns
        -------
        ndarray[complex]:
            The Hamiltonian evaluated at each time. Has the shape
            (t.size, 2, 2), or (2, 2) if t is a scalar.
        """

        t = np.atleast_1d(t)

        H = np.multiply.outer(self.hx(t), self.sigmax) \
            + (self.hy() * self.sigmay)[np.newaxis, :, :] \
            + np.multiply.outer(self.hz(t), self.sigmaz)
        
        return H.squeeze()
    
    def hxn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the driven term hx(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficient(s). Of the same type as n.
        """

        # Form numpy array so that we can iterate through n.
        n = np.atleast_1d(n)
        coeffs = np.zeros_like(n, dtype=complex)

        zeroMask = n == 0
        evenMask = n % 2 == 0
        oddMask = ~evenMask

        # Calculates the relevant coefficients in a vectorised manner.
        # Not sure if scipy is actually faster vectorised, but regardless it neatens up the code.
        coeffs[oddMask] = np.sign(n[oddMask]) * 1j * special.jv(np.abs(n[oddMask]), self._params.drivingAmp) * np.cos(self._params.kx)
        coeffs[evenMask] = special.jv(np.abs(n[evenMask]), self._params.drivingAmp) * np.sin(self._params.kx)
        coeffs[zeroMask] = special.jv(0, self._params.drivingAmp) * np.sin(self._params.kx)

        # Returns the array as a float if it has size 1.
        if coeffs.size == 1:
            return coeffs[0]

        return coeffs
    
    def hyn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the term hyn.
        This is a constant term, so returning the 'Fourier coefficients' just
        returns the value itself for n = 0, and 0 for all other n.
        This function is just a utility to use for compatibility with the other Fourier
        coefficient functions.

        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficient(s). Of the same type as n.
        """

        # Form numpy array so that we can iterate through n.
        n = np.atleast_1d(n)
        coeffs = np.zeros_like(n, dtype=complex)

        # Sets n = 0 to the value of the constant,
        # otherwise leaves all other coefficients as zero.
        coeffs[n == 0] = self.hy()

        # Returns the array as a float if it has size 1.
        if coeffs.size == 1:
            return coeffs[0]

        return coeffs


    def hzn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the driven term hz(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficient(s). Of the same type as n.
        """

        # Form numpy array so that we can iterate through n.
        n = np.atleast_1d(n)
        coeffs = np.zeros_like(n, dtype=complex)

        zeroMask = n == 0
        evenMask = n % 2 == 0
        oddMask = ~evenMask

        # Calculates the relevant coefficients in a vectorised manner.
        # Not sure if scipy is actually faster vectorised, but regardless it neatens up the code.
        coeffs[evenMask] = special.jv(np.abs(n[evenMask]), self._params.drivingAmp) * np.cos(self._params.kx)
        coeffs[oddMask] = np.sign(n[oddMask]) * -1j * special.jv(np.abs(n[oddMask]), self._params.drivingAmp) * np.sin(self._params.kx)
        coeffs[zeroMask] = self._params.delta + np.cos(self._params.kx) * special.jv(0, self._params.drivingAmp) \
                    + np.cos(self._params.ky)

        # Returns the array as a float if it has size 1.
        if coeffs.size == 1:
            return coeffs[0]

        return coeffs
    
    def lattice_fourier_coefficient(self, n: int | np.ndarray[int]) -> np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for H(t), the driven lattice basis Hamiltonian.
        
        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficient(s). Of the same type as n. Comes in the shape
            (n.size, 2, 2), where the third axis disappears if n is a scalar.
        """

        n = np.atleast_1d(n)

        H = np.array([[self.hzn(n), self.hxn(n) - 1j * self.hyn(n)],
                      [self.hxn(n) + 1j * self.hyn(n), -self.hzn(n)]], dtype=complex)
        
        return np.moveaxis(H, -1, 0).squeeze()
    
    def band_fourier_coefficient(self, n: int | np.ndarray[int]) -> np.ndarray[complex]:
        """The nth Fourier coefficient for the driven Hamiltonian in the band basis.
        
        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficient(s) in the band basis. Of the same type as n. Comes in the shape
            (n.size, 2, 2), where the third axis disappears if n is a scalar.
        """

        projector = BandBasisProjector(self)
        return projector.rotate_to_band_basis(self.lattice_fourier_coefficient(n))

    def fourier_minus(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Returns the corresponding coefficient in the Fourier series for H-(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficients, corresponding to the entries of n.
            The type returned is the same as the type of n.
        """

        coeffs = self.band_fourier_coefficient(n)
        return BandBasisProjector.minus_coeff(coeffs)
    
    def fourier_plus(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Returns the corresponding coefficient in the Fourier series for H+(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficients, corresponding to the entries of n.
            The type returned is the same as the type of n.
        """

        coeffs = self.band_fourier_coefficient(n)
        return BandBasisProjector.plus_coeff(coeffs)
    
    def fourier_z(self, n : int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Returns the corresponding coefficient in the Fourier series for Hz(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficients, corresponding to the entries of n.
            The type returned is the same as the type of n.
        """

        coeffs = self.band_fourier_coefficient(n)
        return BandBasisProjector.z_coeff(coeffs)
 
    @cached_property
    def plusEigenvector(self) -> np.ndarray[complex]:
        _, U = np.linalg.eigh(self.lattice_basis())
        return U[:, 1].reshape(2, 1)
    
    @cached_property
    def minusEigenvector(self) -> np.ndarray[complex]:
        _, U = np.linalg.eigh(self.lattice_basis())
        return U[:, 0].reshape(2, 1)
    
    @cached_property
    def U(self) -> np.ndarray[complex]:
        U = np.column_stack((self.plusEigenvector, self.minusEigenvector))

        # Raise an exception if U is not unitary.
        if not np.allclose(U @ U.conj().T, np.eye(2)) or not np.allclose(U.conj().T @ U , np.eye(2)):
            raise ValueError("Matrix U is not unitary.")
        
        return U
    
    @cached_property
    def plusProjection(self) -> np.ndarray[complex]:
        return 0.5 * (np.eye(2) + self.lattice_basis() / self.energy())
    
    @cached_property
    def minusProjection(self) -> np.ndarray[complex]:
        return 0.5 * (np.eye(2) - self.lattice_basis() / self.energy())