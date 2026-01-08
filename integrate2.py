import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpmath import invertlaplace

tau = 1        # Characteristic decay scale.
omegaTilde = 10 # Atomic transition frequency - photon frequency.
rabiFreq = 1

def TimeIndependentBlochEquations(t: np.typing.ArrayLike, c: np.ndarray[float]) -> np.ndarray[float]:
    """
    Function for the optical Bloch equations with radiative damping from
    the (Kocabas et. al., 2012) paper - https://link.aps.org/doi/10.1103/PhysRevA.85.023817.
    These are the time-independent form of the equations from Appendix C.

    Parameters
    ----------
        t : ArrayLike
            The time (t') that we are evaluating the ODE at. Since the equations are time-independent,
            this should only impact the inhomogenous term.
        c : ndarray[float, dtype[Any]]
            The array of correlations.

    Returns
    -------
        ndarray[float, dtype[Any]]
            The value of dc/dt at the given time t.
    """

    # Ensures t is a numpy array.
    t = np.array(t)

    # Common term defined for convenience.
    omegaTerm = -(1 / tau + 1j * omegaTilde)

    # Coefficient matrix.
    M = np.array([[omegaTerm    ,    0                    ,    0.5j * rabiFreq ],
                  [0            ,    omegaTerm.conjugate(),    -0.5j * rabiFreq],
                  [1j * rabiFreq,    -1j * rabiFreq       ,    -2 / tau        ]])

    # Inhomogenous coefficient.
    b = np.array([0, 0, - 2 / tau])

    return M @ c + b

# Atom is in ground state.
initialConditions = np.array([-1, -1, -1], dtype=complex)

# Plotting up to this many tau time intervals.
timeLimit = 30

# The points within the range (0, timeLimit) we will evaluate the numerical solution at.
tAxis = np.linspace(0, timeLimit, 100)
# The points that we evaluate the Laplaced solutions at.
sAxis = np.linspace(0, 50, 100)

# Numerically solves the ODE for n = 1.
numericalSol = integrate.solve_ivp(fun=TimeIndependentBlochEquations, t_span=(0, timeLimit), y0=initialConditions, t_eval=tAxis)

# Defines the analytical solution for the time-independent sigma-minus expectation.
P = lambda s: (s + 2 / tau) * ( (s + 1 / tau)**2 + omegaTilde**2 ) + rabiFreq**2 * (s + 1 / tau)
analyticalFunc = lambda s: -0.5j * rabiFreq * (s + 2 / tau) * (s + 1 / tau - 1j * omegaTilde) / (s * P(s))
# Takes the inverse laplace of the solution and evaluates it at each time t that we desire.
# analyticalSol = np.array([invertlaplace(analyticalFunc, t) for t in numericalSol.t])
analyticalSol = analyticalFunc(sAxis)

# Takes the Laplace Transform of the numerical solution.
numericalLaplacedFunc = lambda s: integrate.simpson(numericalSol.y[0] * np.exp(-s * numericalSol.t), numericalSol.t)
print(numericalSol.y[0].shape, numericalSol.t.shape, sAxis.shape) # All of these should have the same shape.
numericalLaplacedSol = np.array([numericalLaplacedFunc(s) for s in sAxis])
print(numericalLaplacedSol.shape)

diff = numericalSol.y[0] - analyticalSol
# print(np.vstack((numericalSol.y[0], analyticalSol, diff)).T)

# ===========================================================
# Plotting analytical and numerical solutions for sigma-minus.
# ===========================================================

numColor = 'black'
analyticColor = 'blue'

magStyle = 'solid'
realStyle = 'dashed'
imagStyle = 'dotted'

fig, ax = plt.subplots(1, 3, figsize=(32, 6))

# Magnitude of expectation.
ax[0].plot(sAxis, np.abs(numericalLaplacedSol), color=numColor, label='Numerical Magnitude', linestyle=magStyle)
ax[0].plot(sAxis, np.abs(analyticalSol), color=analyticColor, label='Analytical Magnitude', linestyle=magStyle)

# Real part of expectation.
ax[1].plot(sAxis, numericalLaplacedSol.real, color=numColor, label='Numerical Real Part', linestyle=realStyle)
ax[1].plot(sAxis, analyticalSol.real, color=analyticColor, label='Analytical Real Part', linestyle=realStyle)

# Imaginary part of expectation.
ax[2].plot(sAxis, numericalLaplacedSol.imag, color=numColor, label='Numerical Imaginary Part', linestyle=imagStyle)
ax[2].plot(sAxis, analyticalSol.imag, color=analyticColor, label='Analytical Imaginary Part', linestyle=imagStyle)

xLabel = "$s$"
ax[0].set_xlabel(xLabel)
ax[1].set_xlabel(xLabel)
ax[2].set_xlabel(xLabel)

ax[0].set_ylabel("$\\mathcal{L}_t \\left[ e^{ikt} \\langle \\sigma_-(t) \\rangle \\right]$")

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()