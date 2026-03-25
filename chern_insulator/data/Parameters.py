from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass(slots=True)
class EnsembleParameters:
    """
    Parameters shared by the entire ensemble of Chern insulators at
    a range of momentum points.

    Attributes
    ----------
    delta : float
        The mass term in the Chern insulator Hamiltonian.
        Controls the topological phase of the system.
    drivingAmplitude : float
        The amplitude of the driving field in the x-direction.
    drivingFrequency : float
        The frequency of the driving field in the x-direction, in Hz.
    decayConstant : float
        The decay constant for the system, in s^-1.
    maxN : int
        The maximum harmonic that we will calculate Fourier series up to. For
        maxN = n, we will calculate the coefficients c_{-n} to c_n.
    angularFreq : float
        The angular frequency of the driving field, in radians per second.
    """

    delta: float
    drivingAmp: float
    decayConstant: float
    maxN: int

    # Not set to init=False, since we want to explicitly set it
    # when we create a ModelParameters instance, rather than recalculating it
    # in __post_init__.
    drivingFreq: float = field(default=None)
    angularFreq: float = field(init = False)

    def __post_init__(self) -> None:
        """
        Calculates useful terms one time, rather than having to recalculate in
        the main code.
        """

        # Only calculates the band gap again IF this is an instance of EnsembleParameters,
        # meaning drivingFreq is none. If we are creating a ModelParameters instance, we will explicitly
        # set the drivingFreq and so we will not recalculate it.
        
        if self.drivingFreq is None:
            # Imports only in this local space to avoid circular dependency and incomplete import stuff.
            # I'd rather just important the function but I can't seem to do that.
            from Hamiltonian import Hamiltonian

            # Samples the BZ based on the resolution. This can be very detailed since
            # it should only happen a single time in the code.
            resolution = 25
            axisPoints = np.linspace(-np.pi, np.pi, resolution)
            x, y = np.meshgrid(axisPoints, axisPoints)

            energies = Hamiltonian.StaticEnergy(x.flatten(), y.flatten(), self.delta)
            bandGap = 2 * np.min(energies)

            # Finds the minimum band gap over the Brillouin zone.
            # The band gap for an ensemble is controlled only by the delta.

            self.angularFreq = bandGap * 2 / 5
            self.drivingFreq =  self.angularFreq / (2 * np.pi)

@dataclass(slots=True)
class ModelParameters(EnsembleParameters):
    """
    Parameters specific to a single Chern insulator model at a single
    momentum point.

    Parameters
    ----------
    kx : float
        The x-component of the momentum.
    ky : float
        The y-component of the momentum.
    """

    # Given default values so as to not destroy the constructor,
    # since drivingFreq is already given a default value.
    kx: float = field(default = None)
    ky: float = field(default = None)

    @staticmethod
    def FromEnsemble(kx: float, ky: float, params: EnsembleParameters) -> ModelParameters:
        """
        Creates an instance of ModelParameters from an instance of EnsembleParameters.

        Parameters
        ----------
        kx : float
            The x-component of the momentum.
        ky : float
            The y-component of the momentum.
        params : EnsembleParameters
            The object containing the remaining parameters required
            for the ModelParameters instance.

        Returns
        -------
        ModelParameters:
            An instance containing the same attributes as params, along with the
            given kx and ky.
        """
        
        modelParams = ModelParameters(
            kx = kx,
            ky = ky,
            delta = params.delta,
            drivingAmp = params.drivingAmp,
            drivingFreq = params.drivingFreq,
            decayConstant = params.decayConstant,
            maxN = params.maxN
        )

        # Must set outside constructor, since angular freq is not in the constructor.
        modelParams.angularFreq = params.angularFreq

        return modelParams