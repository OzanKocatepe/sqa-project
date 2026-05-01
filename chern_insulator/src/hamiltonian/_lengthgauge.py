import numpy as np
from functools import cache

class LengthGaugeMixin:
    """Contains logic for calculating the system in the length gauge."""

    def Ex(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the value of the driving electric field in the x-direction at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | ndarray[float]:
            The value of the driving field in the x-direction at time t.
        """

        return -self._params.drivingAmp * self._params.angularFreq * np.cos(self._params.angularFreq * t)
    
    @cache
    def XEigenvectorPartials(self) -> np.ndarray[complex]:
        """
        Returns the partial derivative of the eigenvectors with respect to kx.
        Calculated numerically with finite difference method.

        Returns
        -------
        ndarray[complex]:
            An array of shape (2, 2). The column [:, 0] is the partial derivative of the positive eigenvector
            w.r.t. kx, and [:, 1] is the partial of the negative eigenvector.
        """

        deltaK = 1e-5

        # We already have the eigenvectors, so we calculate the eigenvectors at kx + deltaK, ky.
        hx = np.sin(self._params.kx + deltaK)
        hy = self.hy()
        hz = self._params.delta + np.cos(self._params.kx + deltaK) + np.cos(self._params.ky)

        H = hx * self.sigmax + hy * self.sigmay + hz * self.sigmaz
        _, forwardEigenvectors = np.linalg.eigh(H) # Recall that this returns eigenvectors in ascending order of eigenvalue.

        hx = np.sin(self._params.kx - deltaK)
        hy = self.hy()
        hz = self._params.delta + np.cos(self._params.kx - deltaK) + np.cos(self._params.ky)

        H = hx * self.sigmax + hy * self.sigmay + hz * self.sigmaz
        _, backwardEigenvectors = np.linalg.eigh(H) # Recall that this returns eigenvectors in ascending order of eigenvalue.

        # Swaps eigenvectors so that they are positive first, then negative.
        forwardEigenvectors[:, [0, 1]] = forwardEigenvectors[:, [1, 0]]
        backwardEigenvectors[:, [0, 1]] = backwardEigenvectors[:, [1, 0]]

        # eigh can return the eigenvectors with any arbitrary phase, and it can vary between calls,
        # so we fix the gauge relative to the original eigenvectors.
        for col in range(2):
            forward_overlap = np.vdot(self.U[:, col], forwardEigenvectors[:, col])
            forwardEigenvectors[:, col] *= np.exp(-1j * np.angle(forward_overlap))

            backward_overlap = np.vdot(self.U[:, col], backwardEigenvectors[:, col])
            backwardEigenvectors[:, col] *= np.exp(-1j * np.angle(backward_overlap))

        return (forwardEigenvectors - backwardEigenvectors) / (2 * deltaK)


    @cache
    def YEigenvectorPartials(self) -> np.ndarray[complex]:
        """
        Returns the partial derivative of the eigenvectors with respect to ky.
        Calculated numerically with finite difference method.

        Returns
        -------
        ndarray[complex]:
            An array of shape (2, 2). The column [:, 0] is the partial derivative of the positive eigenvector
            w.r.t. ky, and [:, 1] is the partial of the negative eigenvector.
        """

        deltaK = 1e-5

        # We already have the eigenvectors, so we calculate the eigenvectors at kx + deltaK, ky.
        hx = self.hx()
        hy = np.sin(self._params.ky + deltaK)
        hz = self._params.delta + np.cos(self._params.kx) + np.cos(self._params.ky + deltaK)

        H = hx * self.sigmax + hy * self.sigmay + hz * self.sigmaz
        _, forwardEigenvectors = np.linalg.eigh(H) # Recall that this returns eigenvectors in ascending order of eigenvalue.

        # We already have the eigenvectors, so we calculate the eigenvectors at kx + deltaK, ky.
        hx = self.hx()
        hy = np.sin(self._params.ky - deltaK)
        hz = self._params.delta + np.cos(self._params.kx) + np.cos(self._params.ky - deltaK)

        H = hx * self.sigmax + hy * self.sigmay + hz * self.sigmaz
        _, backwardEigenvectors = np.linalg.eigh(H) # Recall that this returns eigenvectors in ascending order of eigenvalue.

        # Swaps eigenvectors so that they are positive first, then negative.
        forwardEigenvectors[:, [0, 1]] = forwardEigenvectors[:, [1, 0]]
        backwardEigenvectors[:, [0, 1]] = backwardEigenvectors[:, [1, 0]]

        # eigh can return the eigenvectors with any arbitrary phase, and it can vary between calls,
        # so we fix the gauge relative to the original eigenvectors.
        for col in range(2):
            forward_overlap = np.vdot(self.U[:, col], forwardEigenvectors[:, col])
            forwardEigenvectors[:, col] *= np.exp(-1j * np.angle(forward_overlap))

            backward_overlap = np.vdot(self.U[:, col], backwardEigenvectors[:, col])
            backwardEigenvectors[:, col] *= np.exp(-1j * np.angle(backward_overlap))
        
        return (forwardEigenvectors - backwardEigenvectors) / (2 * deltaK)
    
    @cache
    def rx(self) -> np.ndarray[complex]:
        """
        Returns the matrix representation of the x-position operator,
        in momentum space, in the band basis.

        Returns
        -------
        ndarray[complex]:
            A 2x2 matrix representing the x-position operator in the band basis.
        """

        # Gets the partials of the eigenvectors w.r.t. kx.
        kxPartials = self.XEigenvectorPartials()

        rx = np.zeros((2, 2), dtype=complex)

        for i in range(2):
            for j in range(2):
                rx[i, j] = 1j * np.vdot(self.U[:, i], kxPartials[:, j])

        return rx

    @cache
    def ry(self) -> np.ndarray[complex]:
        """
        Returns the matrix representation of the y-position operator,
        in momentum space, in the band basis.

        Returns
        -------
        ndarray[complex]:
            A 2x2 matrix representing the y-position operator in the band basis.
        """

        # Gets the partials of the eigenvectors w.r.t. kx.
        kyPartials = self.YEigenvectorPartials()

        ry = np.zeros((2, 2), dtype=complex)

        for i in range(2):
            for j in range(2):
                ry[i, j] = 1j * np.vdot(self.U[:, i], kyPartials[:, j])

        return ry
    
    def DensityMatrixODE(self, t: float, rho: np.ndarray[complex]) -> np.ndarray[complex]:
        """
        Returns the time derivative of the density matrix at time t, given the density matrix at time t.

        Parameters
        ----------
        t : float
            The time, in seconds, at which to evaluate the derivative.
        rho : ndarray[complex]
            The density matrix at time t, represented as a 2x2 complex array.

        Returns
        -------
        ndarray[complex]:
            The time derivative of the density matrix at time t, represented as a 2x2 complex array.
        """

        # Reshapes the rho input which was previously flattened.
        rho = rho.reshape(2, 2)

        # Pulls all the parameters that we need.
        Ex = self.Ex(t)
        rx = self.rx()
        energy = self.energy()
        gamma = self._params.decayConstant
        rho_0 = np.array([[0, 0],
                          [0, 1]], dtype=complex)        

        drho_dt = 2 * energy * self.sigmay * rho \
            - gamma * (rho - rho_0) \
            - 1j * Ex * (rx @ rho - rho @ rx)
        
        # print(rho)
        # print(np.linalg.eigvals(drho_dt))
        # print(rx)
        # print(2 * energy * self.sigmay * rho)
        # print(- gamma * (rho - rho_0))
        # print(1j * Ex * (rx @ rho - rho @ rx))
        # print('\n')
         
        return drho_dt.flatten()
    
    def jxLengthGauge(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Returns the current operator in the band basis in the x-direction at time t.

        Parameters
        ----------
        t : float
            The time, in seconds, at which to evaluate the current operator.
            Can be vectorised.

        Returns
        -------
        ndarray[complex]:
            The paramagnetic current in the x-direction at time(s) t, with shape (2, 2).
            If t is vectorised, this will have shape (t.size, 2, 2).
        """

        energy = self.energy()
        rx = self.rx()

        return -2 * energy * self.sigmay * rx

    def jyLengthGauge(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Returns the current operator in the band basis in the y-direction at time t.

        Parameters
        ----------
        t : float
            The time, in seconds, at which to evaluate the current operator.
            Can be vectorised.

        Returns
        -------
        ndarray[complex]:
            The paramagnetic current in the y-direction at time(s) t, with shape (2, 2).
            If t is vectorised, this will have shape (t.size, 2, 2).
        """

        t = np.atleast_1d(t)
        Ex = self.Ex(t)
        energy = self.energy()
        rx = self.rx()
        ry = self.ry()

        return (-2 * energy * self.sigmay * ry[np.newaxis, :, :]
            + 1j * np.multiply.outer(Ex, rx @ ry - ry @ rx)).squeeze()