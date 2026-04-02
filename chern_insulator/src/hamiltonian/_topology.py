import numpy as np

class TopologyMixin:
    """
    Contains all functions related to the topology of the system.
    
    Methods
    -------
    BerryCurvature: Calculates the berry curvature at some momentum points, vectorised.
    ChernNunber: Calculates the chern number of the system given some delta.
    """

    @staticmethod
    def BerryCurvature(delta: float, kx: float | np.ndarray[float], ky: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Calculates the berry curvature at some momentum. Used to calculate the Chern number.
        
        Parameters
        ----------
        delta : float
            The delta, or mass term, in our Hamiltonian.
        kx : float | ndarray[float]
            The x-component of the momentum.
        ky : float | ndarray[float]
            The y-component of the momentum. Should be a vector
            if and only if kx is a vector of the same size.

        Returns
        -------
        float | ndarray[float]:
            The berry curvature. Return type is the same as the given type of kx and ky.
        """

        # Define our hamiltonian components. Each have shape (k.size).
        hx = np.sin(kx)
        hy = np.sin(ky)
        hz = delta + np.cos(kx) + np.cos(ky)

        # Defines the energy, also of shape (k.size).
        energy = np.sqrt(hx**2 + hy**2 + hz**2)

        # Stacks into an array of shape (3, k.size) to
        # form our h vector.
        h = np.stack([hx, hy, hz], axis=0)

        # Defines our derivatives, each of shape (3, k.size)
        partialX = np.stack([np.cos(kx), np.zeros_like(kx), -np.sin(kx)], axis = 0)
        partialY = np.stack([np.zeros_like(ky), np.cos(ky), -np.sin(ky)], axis = 0)
        
        # The cross product takes in partialX and partialY, of shape (3, k.size), and
        # returns a new array with the cross product results of shape (3, k.size).
        # Then, we take the dot product of each column of h with the corresponding column
        # of the cross product by taking the transpose of h and matrix multiplying.

        # This result is (k.size, 3) * (3, k.size) = (k.size, k.size) array.
        # Since we only care about the dot product for vectors at the same k, the values
        # we care about sit on the diagonal.
        return -0.5 * np.diag(h.T @ np.cross(partialX, partialY, axis=0)) / energy**3
    
    @classmethod
    def ChernNumber(cls, delta : float) -> float:
        """
        Integrates the Berry Curvature over the BZ in order to find the Chern number.
        
        Parameters
        ----------
        delta : float
            The delta, or mass term, in our Hamiltonian.

        Returns
        -------
        float:
            The chern number.
        """

        resolution = 100
        kxAxis = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
        kyAxis = np.linspace(-np.pi, np.pi, resolution, endpoint=False)

        kx, ky = np.meshgrid(kxAxis, kyAxis)
        
        berry = cls.BerryCurvature(delta, kx.flatten(), ky.flatten())
        return np.sum(berry) * 2 * np.pi / resolution**2