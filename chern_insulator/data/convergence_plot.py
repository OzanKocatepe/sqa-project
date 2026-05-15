import numpy as np
import matplotlib.pyplot as plt

from config.paths import DATA_DIR

xAmplitude = np.zeros((10,), dtype=float)
yAmplitude = np.zeros((10,), dtype=float)
kVals = range(10, 110, 10)

for i, k in enumerate(kVals):
    _, currentData = np.load(DATA_DIR / f"D=1.0, k={k}.npy", allow_pickle=True)
    xAmplitude[i] = np.max(np.abs(currentData.totalCurrent[0, -1000:]))
    yAmplitude[i] = np.max(np.abs(currentData.totalCurrent[1, -1000:]))

plt.plot(kVals, xAmplitude - np.mean(xAmplitude), color='blue', label='x', marker='x')
plt.plot(kVals, yAmplitude - np.mean(yAmplitude), color='orange', label='y', marker='x')
plt.xlabel("Sampling Density")
plt.ylabel("Current Amplitude")
plt.legend()
plt.show()