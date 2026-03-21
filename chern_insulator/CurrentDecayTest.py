import numpy as np
import matplotlib.pyplot as plt

from Hamiltonian import Hamiltonian
from data import ModelParameters

resolution = 50
x, y = np.meshgrid(np.linspace(-np.pi, np.pi, resolution), np.linspace(-np.pi, np.pi, resolution))
kAxis = np.stack((x.flatten(), y.flatten()), axis = -1)

jym = np.zeros_like(kAxis[:, 0], dtype=complex)
jyp = np.zeros_like(kAxis[:, 0], dtype=complex)

for i, k in enumerate(kAxis):
    kx = k[0]
    ky = k[1]

    params = ModelParameters(
        delta = 3,
        drivingAmp = 0.2,
        decayConstant = 0.2,
        maxN = 25,
        kx = kx,
        ky = ky
    )

    h = Hamiltonian(params)
    jym[i] = h.jym()
    jyp[i] = h.jyp()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(np.mean(jym.real - jyp.real))
print(np.std(jym.real - jyp.real))
ax.plot(kAxis[:, 0], kAxis[:, 1], jym.real - jyp.real, label='Real Part')
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(np.mean(jym.imag + jyp.imag))
print(np.std(jym.imag + jyp.imag))
ax.plot(kAxis[:, 0], kAxis[:, 1], jym.imag + jyp.imag, label='Imaginary Part')
ax.legend()
plt.show()