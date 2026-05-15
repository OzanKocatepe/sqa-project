from dataclasses import dataclass
import jax.numpy as jnp

from .Parameters import ModelParameters
from operators import Hamiltonian

@dataclass(slots=True, frozen=True)
class DoubleTimeODEParams:
    """Pre-computed static data for the double-time ODE.
    
    This data is used to reproduce the information in the Hamiltonian class
    -- this is less than ideal, but its necessary in order to actually use JAX to speed
    up the ODE calculation.
    """

    U: jnp.ndarray[jnp.complex64]
    kx: float
    ky: float
    delta: float
    drivingAmp: float
    angularFreq: float
    gamma: float

    def build_from_params(params: ModelParameters, hamiltonian: Hamiltonian) -> DoubleTimeODEParams:
        """Creates the double-time ODE params from existing objects.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the model.
        hamiltonian : Hamiltonian
            The hamiltonian operator for this model.
        """

        return DoubleTimeODEParams(
            U = jnp.array(hamiltonian.U),
            kx = float(params.kx),
            ky = float(params.ky),
            delta = float(params.delta),
            drivingAmp = float(params.drivingAmp),
            angularFreq = float(params.angularFreq),
            gamma = float(params.decayConstant)
        )