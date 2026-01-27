import numpy as np
import matplotlib.pyplot as plt
import math

from SSH import SSH
from SSHSimulation import SSHSimulation
from SSHVisualiser import SSHVisualiser

tAxis = np.linspace(0, 30, 10000)
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

params = {
    't1' : 2,
    't2' : 1,
    'decayConstant' : 0.1,
    'drivingAmplitude' : 0.2,
    'drivingFreq' : 2 / 3.01
}

ssh = SSH(np.pi / 4, **params)
ssh.Solve(tAxis, initialConditions)
coeff = ssh.CalculateCurrentExpectationCoefficients()
n_val = (coeff.size - 1) // 2
print([np.abs(coeff[n].conjugate() - coeff[-(n + 1)]) < 1e-10 for n in range(n_val)])