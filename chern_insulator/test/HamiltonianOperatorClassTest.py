from data import ModelParameters
from operators.Hamiltonian import Hamiltonian

import numpy as np

params = ModelParameters(
    delta = 3,
    drivingAmp = 0.2,
    decayConstant = 0.2,
    maxN = 50,
    kx = np.pi / 4,
    ky = np.pi / 8
)

h = Hamiltonian(params)

print(h.lattice_basis(0))
print(h.lattice_basis(10))
print(h.band_basis(0))
print(h.band_basis(10))
print(h.minus(0))
print(h.plus(0))
print(h.z(0))
print(h.energy())

print(h.lattice_fourier_coefficient(1))
print(h.band_fourier_coefficient(1))
print(h.fourier_minus(1))
print(h.fourier_plus(1))
print(h.fourier_z(1))