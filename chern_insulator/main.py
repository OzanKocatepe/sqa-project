import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from data import ModelParameters

tauMax = 20
time = np.linspace(0, tauMax, 4000)
current = np.zeros((2, time.size), dtype=complex)

for kx, ky in [(np.pi / 4, -np.pi / 8), (-np.pi / 4, np.pi / 8)]:
    params = ModelParameters(
        kx = kx,
        ky = ky,
        delta = 1,
        drivingAmp = 0.3,
        drivingFreq = 2 / 3.01 * 1 / (2 * np.pi),
        decayConstant = 0.2,
        maxN = 50
    )

    m = Model(params)
    sigma, current_temp = m.Run(tauMax)
    current += current_temp

timeSec = time / params.decayConstant
labels = ['x', 'y']
functionLabels = ['Real Part of', 'Imag Part of']
whateverman = ['Real', 'Imag']

# for fi, function in enumerate([lambda x: x.real, lambda x: x.imag]):
#     for index in [0, 1]:
#         plt.plot(time, function(current[index, :]), label=f"j{labels[index]}")

#     plt.title(f"{functionLabels[fi]} Current")
#     plt.legend()
#     plt.savefig(f"chern_insulator/plots/[{whateverman[fi]}] Current")
#     plt.show()

sampleSpacing = (np.max(timeSec) - np.min(timeSec)) / timeSec.size
freqAxis = np.fft.fftshift(np.fft.fftfreq(timeSec.size, sampleSpacing)) / params.drivingFreq
plt.semilogy(freqAxis, np.abs(np.fft.fftshift(np.fft.fft(current[0, :]))), label='Jx')
plt.semilogy(freqAxis, np.abs(np.fft.fftshift(np.fft.fft(current[1, :]))), label='Jy')
plt.xlim(-12.5, 12.5)
plt.legend()
plt.savefig(f"chern_insulator/plots/Current FFT.png")
plt.show()