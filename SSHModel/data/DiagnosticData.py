import numpy as np
from dataclasses import dataclass

from .AxisData import AxisData
from .SSHParameters import ModelParameters
from .CorrelationData import CorrelationData

@dataclass
class DiagnosticData:
    """Contains data only used for diagnostic purposes, such as intermediary values in calculations."""
    
    # The double-time current product (second half of the connected correlator) evaluated
    # numerically using the timeDomainCurrent at multiple times, rather than using the fourier series
    # derived analytically. Used to make the product term in the connected correlator is correct.
    numericalDoubleTimeCurrentProduct: np.ndarray[complex] = None

    r"""
    The connected current correlator is of the form

    $ \frac{1}{T} \int dt\, \langle j(t) j(t + \tau) \rangle  - \langle j(t) \rangle \langle j(t + \tau) \rangle $

    We know that we can express these terms in the form

    $\langle j(t) j (t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle = \sum_{i, j} c_{ij} (t, t + \tau) (\langle \sigma_i (t) \sigma_j (t + \tau) \rangle - \langle \sigma_i(t) \rangle \sigma_j j(t + \tau) \rangle)$

    Hence, to figure out why our integrated connected correlator isn't decaying to zero, we want to store
    the terms

    $\frac{1}{T} \int dt\, c_{ij} (t, t + \tau) ( \langle \sigma_i(t) \sigma_j(t + \tau) \rangle - \langle \sigma_i (t) \rangle \langle \sigma_j(t + \tau) \rangle)$
    
    In reality, the correlator only has four unique $c_{ij}$ coefficients, so we only store four of these terms,
    where the operator terms are the sum of multiple connected correlators.
    """

    # This stores the four terms described above.
    integratedConnectedCurrentTerms: np.ndarray[complex] = None

    def __add__(self, other: DiagnosticData) -> DiagnosticData:
        """
        Adds another DiagnosticData instance to itself to create a new DiagnosticData instance.

        Parameters
        ----------
        other : DiagnosticData
            The other DiagnosticData object to add to this one.

        Returns
        -------
        DiagnosticData
            The result of the sum.
        """

        return DiagnosticData(
            # numericalDoubleTimeCurrentProduct = self.numericalDoubleTimeCurrentProduct + other.numericalDoubleTimeCurrentProduct,
            integratedConnectedCurrentTerms = self.integratedConnectedCurrentTerms + other.integratedConnectedCurrentTerms
        )
    
    def CalculateConnectedCorrelatorTerms(self, axes: AxisData, params: ModelParameters, coeffs: np.ndarray[complex], corr: CorrelationData) -> None:
        """
        Calculates each term in the integrated connected current correlator individually,
        so that we can see which terms are going non-zero.

        Parameters
        ----------
        axes : AxisData
            The AxisData, so that we can integrate over the t-axis.
        params : ModelParameters
            The parameters of the model, specifically for the driving frequency and t2.
        coeffs : ndarray[complex]
            An array of shape (4, tAxis.size, tauAxis.size), corresponding to the value of each of the
            double-time current coefficients at times t and t' = t + tau.
        corr : CorrelationData
            The CorrelationData containing the single- and double-time correlations.
        """

        # The required double-time operators.
        operatorTerms = np.array([
            corr.doubleTime[2, 2],
            corr.doubleTime[1, 1] - corr.doubleTime[1, 0] - corr.doubleTime[0, 1] + corr.doubleTime[0, 0],
            corr.doubleTime[1, 2] - corr.doubleTime[0, 2],
            corr.doubleTime[2, 1] - corr.doubleTime[2, 0]
        ])

        # Calculates the $\langle \sigma_i (t) \rangle \langle \sigma_j(t + \tau) \rangle$ values at times t and tau for all i, j.
        productTerms = np.zeros((3, 3, axes.tAxisSec.size, axes.tauAxisSec.size), dtype=complex)
        for i in range(3):
            for j in range(3):
                productTerms[i, j] = corr.singleFourierSeries[i].Evaluate(axes.tAxisSec)[:, np.newaxis] * corr.singleFourierSeries[j].Evaluate(np.add.outer(axes.tAxisSec, axes.tauAxisSec))

        # Calculates the required connected correlator terms for each i, j.
        connectedOperatorTerms = operatorTerms - np.array([
            productTerms[2, 2],
            productTerms[1, 1] - productTerms[1, 0] - productTerms[0, 1] + productTerms[0, 0],
            productTerms[1, 2] - productTerms[0, 2],
            productTerms[2, 1] - productTerms[2, 0]
        ])

        # Multiplies by the required coefficient for each term.
        connectedOperatorTerms *= params.t2**2 * coeffs

        # Finally, integrates the four terms with respect to a steady state period along
        # the t-axis.
        self.integratedConnectedCurrentTerms = params.drivingFreq * np.trapezoid(
            y = connectedOperatorTerms,
            x = axes.tAxisSec,
            axis = 1
        )