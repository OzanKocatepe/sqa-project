import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpmath import invertlaplace

# All time scales are given in units of tau, and all frequencies
# are given in units of 1 / tau.
tau = 1
atomicTransitionFrequency = 1
rabiFrequency = 1
photonFrequency = 1

def RadiativeDampedBlochEquations(t: float, c: np.ndarray[float]) -> np.ndarray[float]:
    """A function representing the optical Bloch equations with radiative damping from
    the (Kocabas et. al., 2012) paper - https://link.aps.org/doi/10.1103/PhysRevA.85.023817
    
    Parameters:
        t (float):
            The second time considered by the operators.
            This is also the variable we are solving the equations with respect to.
        c (float):
            The vector of correlation functions which we are solving for.
    """

    # Defines useful terms for coefficient matrix.
    omegaTerm = 1 + 1j * atomicTransitionFrequency
    expTerm = rabiFrequency * np.exp(1j * photonFrequency * t, dtype=complex)


    # Coefficient matrix.
    B = (1 / tau) * np.array([[-omegaTerm  ,   0                        ,   0.5j * expTerm.conjugate()],
                              [0           ,   -omegaTerm.conjugate()   ,   -0.5j * expTerm           ],
                              [1j * expTerm,   -1j * expTerm.conjugate(),   -2                        ]], dtype=complex)

    # Inhomogenous term.
    b = np.array([0, 0, -2 / tau])

    # Returns the ODE.
    return B @ c + b

initialConditions = np.array([-1, -1, -1], dtype=complex) # Atom is in ground state.
timeLimit = 30 # Going up to 5 tau time intervals.
xAxis = np.linspace(0, timeLimit, 100) # Points to evaluate the solution at.

# Solves the ODE.
solution = solve_ivp(fun=RadiativeDampedBlochEquations, t_span=(0, timeLimit), y0=initialConditions, t_eval=xAxis)

def P(s: float) -> float:
    return (s + 2 / tau) * ( (s + 1 / tau)**2 + (atomicTransitionFrequency - photonFrequency)**2 ) + rabiFrequency**2 * (s + 1 / tau)

# Manually writes out the Laplace transformed ODEs.
laplaceMinusExpectation = lambda s: -0.5j * rabiFrequency * (s + 2 / tau) * (s + 1 / tau - 1j * (atomicTransitionFrequency - photonFrequency)) / (s * P(s))
analyticMinusExpectation = np.array([invertlaplace(laplaceMinusExpectation, t) for t in xAxis[1:]])

# Plots the ODE.
plt.plot(solution.t, solution.y[0], color='blue', label='$\\langle \\sigma_- (t) \\rangle$', linestyle='dashed')
plt.plot(xAxis[1:], analyticMinusExpectation, color='red', label='Analytic $\\langle \\sigma_- (t) \\rangle$', linestyle='dashed')
# plt.plot(solution.t, solution.y[1], color='red', label='$\\langle \\sigma_+ (t) \\rangle$', linestyle='dashed')
# plt.plot(solution.t, solution.y[2], color='green', label='$\\langle \\sigma_z(t) \\rangle$', linestyle='dashed')
plt.xlabel("t' / $ \\tau $ (dimensionless)")
plt.ylabel("Expectation")
plt.legend()
plt.show()