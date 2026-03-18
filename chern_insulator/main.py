import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from data import ModelParameters

params = ModelParameters(
    kx = np.pi / 4,
    ky = -np.pi / 8,
    delta = 1,
    drivingAmp = 0.3,
    drivingFreq = 2 / 3.01 * 1 / (2 * np.pi),
    decayConstant = 0.2,
    maxN = 50
)

m = Model(params)
f, sigma = m.Run(tauMax = 20)
time = np.linspace(0, 20, 4000)

labels = ['-', '+', 'z']
functionLabels = ['Real Part of', 'Imag Part of']
whateverman = ['Real', 'Imag']

for fi, function in enumerate([lambda x: x.real, lambda x: x.imag]):
    for index in [0, 1, 2]:
        plt.plot(time, function(sigma[index, :]), color='black')
        plt.plot(time, function(f[index].Evaluate(time / params.decayConstant)), color='blue')
        plt.title(f"{functionLabels[fi]} {labels[index]}")
        plt.savefig(f"chern_insulator/plots/{whateverman[fi]} {labels[index]}")
        plt.show()