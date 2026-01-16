import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Callable

t1 = 1
t2 = 1
drivingAmplitude = 1
drivingFreq = 1
k = np.pi / 4

def ClassicalDrivingTerm(t: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    The classical sinusoidal driving term that we are considering. This models a laser with frequency $\Omega$ and amplitude $A_0$.

    Parameters
    ----------
        t : ArrayLike
            The time.

    Returns
    -------
        ArrayLike
            The value of the driving term at that time.
    """

    return drivingAmplitude * np.sin(drivingFreq * t)

def ClassicallyDrivenSSHEquations(t: float, c: np.ndarray[float], A: Callable[[np.typing.ArrayLike], np.typing.ArrayLike]) -> np.ndarray[float]:
    """
    The ODE (equations of motion) for the single-time expectations of $\sigma_-(t)$, $\sigma_+(t)$, and $\sigma_z(t)$. 

    Parameters
    ----------
        t : float
            The time t.
        c : ndarray[float, dtype[Any]]
            The 3-component single-time correlation array.
        A : Callable[[np.typing.ArrayLike], np.typing.ArrayLike]
            The classical driving term. Should be a vectorisable function of t.

    Returns:
    --------
        ndarraypfloat, dtype[Any]] 
            The value of dc/dt at some time t.
    """

    # Coefficient matrix at time t. Don't know if we can make a useful time-independent transformation here.
    HPlus = t1 + t2 * np.exp(-1j * (k - A(t)))
    HMinus = t1 + t2 * np.exp(1j * (k - A(t)))
    B = np.array([[0            , 0          ,  1j * HPlus    ],
                  [0            , 0          ,  -1j * HMinus  ],
                  [2j * HMinus  , -2j * HPlus,  0             ]], dtype=complex)
    
    return B @ c

# ===============================
# ==== NUMERICALLY SOLVE ODE ====
# ===============================

tDomain = (0, 30)
tAxis = np.linspace(tDomain[0], tDomain[1], 250)
initialConditions = np.array([0, 0, -1]) # We assume that the system is in its ground state at time 0.

numericalSol = integrate.solve_ivp(fun=ClassicallyDrivenSSHEquations,
                                    t_span=tDomain,
                                    y0=initialConditions,
                                    t_eval=tAxis,
                                    rtol=1e-10,
                                    atol=1e-12,
                                    args=(ClassicalDrivingTerm,))