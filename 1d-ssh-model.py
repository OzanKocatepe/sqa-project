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

ssh = SSH(k = np.pi / 4, **params)
ssh.Solve(tAxis, initialConditions)

# Calculates maximum allowed n based on nyquist frequency.
dx = np.mean(np.diff(tAxis / params['decayConstant']))
maxAllowedN = math.floor(1 / (4 * dx * np.pi * params['drivingFreq']))
n = maxAllowedN
print(maxAllowedN)

coefficients = ssh.CalculateFourierCoefficients(n, steadyStateCutoff=20, numPeriods=10) # (3, 2n + 1)