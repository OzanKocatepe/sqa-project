import numpy as np
from dataclasses import dataclass

@dataclass
class Fourier:
    r"""Stores a Fourier series $\sum_{n = -N}^N c_n e^{i n \omega t}$, where $\omega = 2\pi f$ and $f$ is the base frequency given to the system, in Hz."""

    baseFreq: float
    coeffs: np.ndarray[complex]

    angularFreq: float = None
    n : int = None

    def __post_init__(self):
        """Defines any derived terms after initialisation."""

        self.angularFreq = 2 * np.pi * self.baseFreq
        self.n = (self.coeffs.size - 1) // 2

    def __add__(self, other: Fourier) -> Fourier:
        """
        Adds two fourier series together coefficient-wise.

        Parameters
        ----------
        other : Fourier
            Another Fourier object with the same baseFreq value.

        Returns
        -------
        Fourier
            A new Fourier object.

        Raises
        ------
        ValueError
            If the base frequencies don't match.
        """

        if np.abs(self.baseFreq - other.baseFreq) > 1e-3:
            raise ValueError("Operands must have the same base frequency.")

        return Fourier(
            baseFreq = self.baseFreq,
            coeffs = self.coeffs + other.coeffs
        ) 

    def __getitem__(self, k: int) -> complex:
        """
        Returns the 'k'th coefficient. A convenience function
        so that we don't need to do the index shifting if we are only interested
        in a couple of coefficients.

        Parameters
        ----------
        k : int
            The index of the coefficient that we want.
        
        Raises
        ------
        IndexError
            If the given index is out of bounds (not within -n to n).
        """

        if np.abs(k) > self.n:
            raise IndexError("Index out of bounds.")
        
        return self.coeffs[k + self.n]

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
            The value of the fourier expansion at each given point.
        """

        # Makes sure we can apply vectorised operations.
        tPoints: np.ndarray[float] = np.atleast_1d(tPoints)

        # Calculates terms $-in\omega, \dots, in\omega = -2i\pi nf, \dots, 2i\pi nf$..
        expTerms: np.ndarray[complex] = 1j * self.angularFreq * np.arange(-self.n, self.n + 1)

        r"""
        Takes the outer product with $t_0, \dots, t_m$ so that we get an array of shape
        (tPoints.size, 2n + 1), where the first dimension corresponds to the time, and the second
        corresponds to the exponent.
        """
        expTerms = np.outer(tPoints, expTerms)
        # Makes the terms actually exponential.
        expTerms = np.exp(expTerms, dtype=complex)

        # Takes the dot product of each row (each row has constant time, contains frequency exponentials)
        # with the coefficients. Results in an array of shape (tPoints.size,) with the value of the expansion at each time.
        return np.dot(expTerms, self.coeffs)

    @staticmethod
    def DetermineMaxN(x: np.ndarray[float], baseFreq: float) -> int:
        """
        Determines the maximum integer multiple of the base frequency allowed by the
        sampling frequency.

        Parameters
        ----------
        x : ndarray[float]
            The points at which we sampled the function.
        baseFreq : float
            The frequency, in Hz, whose harmonics form the exponential basis
            of the Fourier series.

        Returns
        -------
        int
            The maximum allowed integer multiple of the base frequency.
        """

        dx = np.mean(np.diff(x))
        return np.floor(1 / (4 * dx * np.pi * baseFreq)).astype(int)

    @classmethod
    def Convolve(cls, first: Fourier, second: Fourier) -> Fourier:
        """
        Convolves two fourier series with the same base frequency and number of coefficients.
        Essentially, calculates the Fourier series formed by multiplying the two series together.

        Parameters
        ----------
        first : Fourier
            The first fourier series.
        second : Fourier
            The second fourier series.

        Returns
        -------
        Fourier
            The product of the two given fourier series.

        Raises
        ------
        ValueError
            If the base frequencies don't match.
        """

        if np.abs(first.baseFreq - second.baseFreq) > 1e-3:
            raise ValueError("Arguments must have the same base frequency.")

        r"""
        The following code is slightly convoluted, so a better explanation is given here. We have two fourier expansions.
        
        $\sum_{n = -N_1}^{N_1} a_n e^{in \omega t}, \quad \sum_{m = -N_2}^{N_2} b_m e^{im \omega t}$

        We want to multiply these together. Clearly, this is just

        $\sum_{n = -N_1}^{N_1} \sum_{m = -N_2}^{N_2} a_n b_m e^{i(n + m) \omega t}$

        However, we want this as a single sum in the basis of $e^{in\omega t}$ so that we can express the entire thing as a
        fourier expansion. To do this, we will consider that the only possible values of $n + m$ are $\in [-N_1 - N_2, N_1 + N_2] \subseteq \mathbb{Z}$.
        We also know that for any fixed $c := n + m$, the two coefficients that form a coefficient of $e^{ic\omega t}$ are $a_n b_{c - n}$
        for every possible value of $n$ that we can achieve. Hence, the final form of the system is

        $\sum_{n = -N_1 - N_2}^{N_1 + N_2} \left( \sum_{m} a_m b_{n - m} \right) e^{in \omega t}$

        where we sum over all the possible values of $m$ that allow both coefficients to exist. So, for each fixed $n$,
        we will sum over all the $m$ such that $m \in [-N_1, N_1]$ and $n - m \in [-N_2, N_2]$. Hence, in code, we can loop over all the values
        of $m \in [-N_1, N_1]$ and manually check if $n - m \in [-N_2, N_2]$ for the current values of $m$ and $n$, and only add that term to the
        current $n$th coefficient of the final fourier expansion if we return true.
        """

        n1, n2 = first.n, second.n

        coefficients = np.zeros((2 * (n1 + n2) + 1), dtype=complex)

        # Looping through every possible value of the harmonic frequency.
        for frequencySum in range(-n1 - n2, n1 + n2 + 1):
            # Loops through every possible coefficient index, from -n1 to n1, of the first fourier series.
            for firstIndex in range(-n1, n1 + 1):
                # Checks if the required second coefficient required to make the two indices sum to frequencySum exists
                # in the second fourier series.
                if np.abs(frequencySum - firstIndex) <= n2:
                    coefficients[frequencySum + (n1 + n2)] += first[firstIndex] * second[frequencySum - firstIndex]

        return cls(
            baseFreq = first.baseFreq,
            coeffs = coefficients
        )
    
    @classmethod
    def FromSamples(cls, baseFreq: float, y: np.ndarray[complex] | np.ndarray[float], x: np.ndarray[float], numPeriods: int) -> Fourier:
        r"""
        Creates a fourier series from a set of samples.

        Parameters
        ----------
        baseFreq : float
            The base frequency, in Hz, with respect to which we will expand out the function. The basis
            of the function will consist of exponentials $e^{i n \omega}$ where $\omega = 2\pi f$, with $f$ being the base frequency.
        y : ndarray[complex] | ndarray[float]
            The samples of some function that we want to calculate the fourier expansion of.
        x : ndarray[float]
            The x-axis of the samples of the function given in 'y.
        numPeriods: int
            The number of periods that 'x' contains. Only accepts an integer number of periods.

        Returns
        -------
        Fourier
            A Fourier object representing the fourier series, with the maximum number of coefficients allowed
            by the sampling rate.
        """

        n = cls.DetermineMaxN(x, baseFreq)
        coeffs = np.zeros((2 * n + 1), dtype=complex)
        angularFreq = 2 * np.pi * baseFreq

        # Calculates the mean manually to add to the zero frequency coefficient later.
        functionMean = np.mean(y)
        meanSubtractedSamples = y - functionMean

        # Calculates each coefficient in turn.
        for k in range(-n, n + 1):
            coeffs[k + n] = baseFreq / numPeriods * np.trapezoid(
                y = meanSubtractedSamples * np.exp(-1j * angularFreq * k * x),
                x = x
            )

        # Manually adds the mean back to the zero frequency coefficient.
        coeffs[n] += functionMean

        return cls(
            baseFreq = baseFreq,
            coeffs = coeffs
        )