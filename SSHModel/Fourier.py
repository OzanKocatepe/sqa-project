import numpy as np
import math

class Fourier:
    """Contains all of the information for a fourier expansion."""

    def __init__(self, baseFreq: float, coeffs: np.ndarray[complex]=None, samples: np.ndarray[complex] | np.ndarray[float]=None, samplesX: np.ndarray[float]=None, numPeriods: int=None):
        r"""
        Initialises a fourier object.
        
        Parameters
        ----------
        baseFreq : float
            The base frequency with respect to which we will expand out the function. The basis
            of the function will consist of exponentials $e^{i n \omega}$ where $\omega = 2\pi f$, with $f$ being the base frequency.
        coeffs : ndarray[complex]
            The coefficients of a fourier expansion. Must be an array of the shape
            (2n + 1,) for some integer n, containing the coefficients with index -n to n.
            If given, 'samples' will be ignored.
        samples : ndarray[complex] | ndarray[float]
            The samples of some function that we want to calculate the fourier expansion of.
            Will only be used if 'coeffs' is none.
        samplesX : ndarray[float]
            The x-axis of the samples of the function given in 'samples'. Must be given if 'coeffs' is None
            and 'samples' is given, otherwise will be ignored.
        numPeriods: int
            The number of periods that 'samplesX' contains. Must be an integer number of periods, and must be
            given if 'coeffs' is None.

        Raises
        ------
        AttributeError
            If neither 'coeffs' nor 'samples' is given.
        """

        self.__baseFreq = baseFreq
        self.__coeffs = coeffs
        self.__samples = samples
        self.__samplesX = samplesX
        self.__numPeriods = numPeriods

        # Our expansion contains coefficients from -n to n.
        self.__n: int

        if self.__coeffs is None and self.__samples is None:
            raise AttributeError("Either 'coeffs' or 'samples' must be given.")
        elif self.__coeffs is None:
            # Calculates the number and value of the coefficients based on the samples.
            self.__CalculateCoefficients()
        else:
            # Calculates the number of coefficients from the coefficients array.
            self.__n = (self.__coeffs.size - 1) // 2

    def __getitem__(self, n: int):
        """
        Returns the 'n'th coefficient. A convenience function
        so that we don't need to do the index shifting if we are only interested
        in a couple of coefficients.
        
        Raises
        ------
        IndexError
            If the given index is out of bounds.
        """

        if np.abs(n) > self.__n:
            raise IndexError("Index out of bounds.")
        
        return self.__coeffs[n + self.__n]

    def __CalculateCoefficients(self) -> None:
        """
        Determines the coefficients of a fourier expansion based on the given samples.
        Saves the coefficients to the 'coeffs' and 'n' attributes on its own.
        This is used when the coefficients are not directly given.
        """

        self.__n = self.DetermineMaxN(self.__samplesX, self.__baseFreq)
        self.__coeffs = np.zeros((2 * self.__n + 1), dtype=complex)

        # Calculates the mean manually to add to the zero frequency coefficient later.
        functionMean = np.mean(self.__samples)
        meanSubtractedSamples = self.__samples - functionMean

        # Calculates each coefficient in turn.
        for k in range(-self.__n, self.__n + 1):
            self.__coeffs[k + self.__n] = self.__baseFreq / self.__numPeriods * np.trapezoid(
                y = meanSubtractedSamples * np.exp(-1j * 2 * np.pi * k * self.__baseFreq * self.__samplesX),
                x = self.__samplesX
            )

        # Manually adds the mean back to the zero frequency coefficient.
        self.__coeffs[self.__n] += functionMean

    @staticmethod
    def DetermineMaxN(xAxis: np.ndarray[float], baseFreq: float) -> None:
        """
        Determines the maximum harmonic of the base frequency allowed based on the
        Nyquist frequency of the sampling.

        Parameters
        ----------
        xAxis : ndarray[float]
            The points at which we sampled the function.
        baseFreq : float
            The frequency which we will use to form the Fourier basis.
        """

        dx = np.mean(np.diff(xAxis))
        return math.floor(1 / (4 * dx * np.pi * baseFreq))

    def Evaluate(self, tPoints: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Evaluates the fourier expansion at the given points.
        
        Parameters
        ----------
        tPoints: float | ndarray[float]
            The points that we will be evaluating the fourier expansion at.

        Returns
        -------
        float | ndarray[float]
            The value of the fourier expansion at each point in 'tPoints'.
        """

        # Makes sure we can apply vectorised operations.
        tPoints: np.ndarray[float] = np.atleast_1d(tPoints)

        # Calculates terms $-in\omega, \dots, in\omega = -2i\pi nf, \dots, 2i\pi nf$..
        expTerms: np.ndarray[complex] = 2j * np.pi * np.arange(-self.__n, self.__n + 1) * self.__baseFreq
        r"""
        Takes the outer product with $t_0, \dots, t_m$ so that we get an array of shape
        (tPoints.size, 2n + 1), where the first dimension corresponds to the time, and the second
        corresponds to the exponent.
        """
        expTerms = np.outer(tPoints, expTerms)
        # Makes the terms actually exponential.
        expTerms = np.exp(expTerms)

        # Takes the dot product of each row (each row has constant time, contains frequency exponentials)
        # with the coefficients. Results in an array of shape (tPoints.size,) with the value of the expansion at each time.
        return np.dot(expTerms, self.__coeffs)
    
    @property
    def n(self):
        return self.__n
    
    @property
    def coeffs(self):
        return self.__coeffs
    
    @property
    def baseFreq(self):
        return self.__baseFreqs