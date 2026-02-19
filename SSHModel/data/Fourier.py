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

        self.coeffs = np.array(self.coeffs, dtype=complex)
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

    def __getitem__(self, k: int | slice) -> complex | np.ndarray[complex]:
        """
        Returns the 'k'th coefficient. A convenience function
        so that we don't need to do the index shifting if we are only interested
        in a couple of coefficients.

        Parameters
        ----------
        k : int | slice
            The index or slice. Using this, it is impossible to use negative integers
            to reference values from the end of the list, since negative integers refer
            to the coefficients with negative indices. Hence, all slices will just be interpreted as
            a:b:c -> from a (incl.), to b (excl.), in steps of c.

        Returns
        -------
        complex | np.ndarray[complex]
            The indices stored at the desired position.
        
        Raises
        ------
        IndexError
            If any of the given indices are out of bounds (not within -n to n).
        """

        # If k is an integer,
        if isinstance(k, int):
            # Check if its in bounds.
            if np.abs(k) > self.n:
                raise IndexError("Index out of bounds.")
            
            # Otherwise, return the relevant coefficient.
            return self.coeffs[k + self.n]
        
        # If k is a slice.
        elif isinstance(k, slice):
            # Replaces None values with default values.
            start = k.start if k.start is not None else -self.n
            stop = k.stop if k.stop is not None else self.n + 1
            step = k.step if k.step is not None else 1

            # If start is less than the first coefficient, just start at the beginning of the list.
            if start < -self.n:
                start = -self.n
            # If we are trying to start past the end of the list, choose the last element.           
            elif start > self.n:
                start = self.n

            # If we are trying to stop before the list starts, just stop at the first element (exclusive).
            if stop < -self.n:
                stop = -self.n
            # If we are trying to stop after we have already finished all the elements, just stop after finishing all the
            # elements.
            elif stop > self.n + 1:
                stop = self.n + 1

            # Shifts the indices for use in the coeffs array.
            start += self.n
            stop += self.n

            return self.coeffs[range(start, stop, step)]

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
        # Uses outer product to multiply every possible harmonic term by every possible tPoint.
        # However, if tPoints is more than one-dimensional, np.outer flattens it.
        expTerms = np.outer(tPoints, expTerms)
        # Makes the terms actually exponential.
        expTerms = np.exp(expTerms, dtype=complex)

        # Takes the dot product of each row (each row has constant time, contains frequency exponentials)
        # with the coefficients. Results in an array of shape (tPoints.size,) with the value of the expansion at each time.
        result = np.dot(expTerms, self.coeffs)

        # Result now has shape (tPoints.flatten().size,), so we have to reshape it back to the original shape
        # for user convenience.
        return result.reshape(tPoints.shape)

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
    
    def BuildConvolutionMatrix(self) -> np.ndarray[complex]:
        """
        Creates a convolution matrix from the coefficients of this Fourier series.
        The matrix is such that if it is left-multiplied onto a vector
        containing the coefficients of another Fourier series, the resulting vector
        will be the coefficients of the convolution of the two Fourier series,
        truncated to -N to N.

        Returns
        -------
        np.ndarray[complex]
            The (2n + 1, 2n + 1) convolution matrix.
        """

        # Creates the empty matrix.
        M = 2 * self.n + 1
        convMatrix = np.zeros((M, M), dtype=complex)

        # Loops through each element in the coefficient array.
        for index in np.arange(-self.n, self.n + 1, dtype=int):
            # Calculates the size of the diagonal. For every unit
            # that we are above or below the diagonal, the size of the diagonal
            # is reduced by 1 from its starting size of 2n + 1.
            diagonalSize = M - np.abs(index)

            # Creates a tuple of the current coefficient of the right size
            # for the diagonal.
            arr = (self[int(index)],) * diagonalSize
            
            # Adds the diagonal to the convolution matrix.
            convMatrix += np.diag(arr, k = -index)

        return convMatrix

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
            The product of the two given fourier series, truncated or padded to have the same
            number of coefficients as the first fourier series.

        Raises
        ------
        ValueError
            If the base frequencies don't match.
        """

        if np.abs(first.baseFreq - second.baseFreq) > 1e-3:
            raise ValueError("Arguments must have the same base frequency.") 

        n1, n2 = first.n, second.n

        # If the second series is too large, truncate it to have coefficients from -n1 to n1.
        if n2 > n1:
            newCoeffs = second[-n1:n1 + 1]
        # If the second series is too small, pad it with zeros to match the first.
        elif n2 < n1:
            # Determines the padding on each side of the coefficient array.
            padding = np.zeros((n1 - n2), dtype=complex)
            newCoeffs = np.concat([padding,
                                  second.coeffs,
                                  padding])
        else:
            newCoeffs = second.coeffs

        # Now that the second Fourier is the same size as the first, calculate the convolution.
        convolvedCoeffs = first.BuildConvolutionMatrix() @ newCoeffs
        
        return cls(
            baseFreq = first.baseFreq,
            coeffs = convolvedCoeffs
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