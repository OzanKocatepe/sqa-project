import numpy as np
from scipy.integrate import solve_ivp

def RadiativeDampedBlochEquations(t: float, t_prime: float, c: np.ndarray[float], n) -> np.ndarray[float]:
    """A function representing the optical Bloch equations with radiative damping from
    the (Kocabas et. al., 2012) paper - https://link.aps.org/doi/10.1103/PhysRevA.85.023817
    
    Parameters:
        t (float):
            The first time considered by the operators. Not considered in c_1.
        t_prime (float):
            The second time considered by the operators.
            This is also the variable we are solving the equations with respect to.
        c (float):
            The vector of correlation functions which we are solving for.
        n (float):
            The degree of the correlation function we are solving for.
            c_1, c_2, and c_3 respectively are defined in the above paper
            in equation (5).
    """

    # Coefficient matrix.
    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])

    # Inhomogenous term.
    b = np.array([0, 0, 0])

    # Returns the ODE.
    return B @ c + b