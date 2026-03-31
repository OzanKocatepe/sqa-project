import numpy as np

H = np.array([[1, 5 + 1j],
              [5 - 1j, 2]], dtype=complex)

eigs, U = np.linalg.eigh(H)
U_dag = U.conj().T

plus = U[:, 0].reshape(2, 1)
minus = U[:, 1].reshape(2, 1)

P_plus = plus @ plus.conj().T
P_minus = minus @ minus.conj().T

A = np.array([[2, 8 + 3j],
              [8 - 3j, 1]], dtype=complex)

# Picking r.
r = (plus + minus) / np.sqrt(2)
r = r.reshape(2, 1)
rT = r.conj().T

print("Matrix A in the eigenbasis of H:")
print(U_dag @ A @ U)

print("Value of <+|A|+>:")
print(plus.conj().T @ A @ plus)
print("Value of Tr(P_+ A):")
print(np.trace(P_plus @ A))

print("Value of <+|A|->:")
print(plus.conj().T @ A @ minus)
print("Value found using arbitrary vector r:")
print(rT @ P_plus @ A @ P_minus @ r / (np.sqrt(rT @ P_plus @ r @ rT @ P_minus @ r)))
print("Value of P_+ A P_-:")
print(np.trace(P_plus @ A @ P_minus))