import numpy as np

m = np.array([[0, 0],
              [1, 0]])

p = np.array([[0, 1],
              [0, 0]])

x = np.array([[0, 1],
              [1, 0]])

y = np.array([[0, -1j],
              [1j, 0]])

z = np.array([[1, 0],
              [0, -1]])

def comm(A, B):
    return A @ B - B @ A

def anti_comm(A, B):
    return A @ B + B @ A