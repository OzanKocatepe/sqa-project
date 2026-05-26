import numpy as np
import matplotlib.pyplot as plt
from config.paths import DATA_DIR, PLOTTING_DIR

axes, one_current = np.load(DATA_DIR / "D=1.0, k=21.npy", allow_pickle = True)
_, three_current = np.load(DATA_DIR / "D=3.0, k=21.npy", allow_pickle = True)

plt.plot(np.arange(1, 51), one_current.second_order_correlation_function[0], marker = 'x', label=r"$\Delta = 1$, x")
plt.plot(np.arange(1, 51), one_current.second_order_correlation_function[1], marker = 'x', label=r"$\Delta = 1$, y")
plt.plot(np.arange(1, 51), three_current.second_order_correlation_function[0], marker = 'x', label=r"$\Delta = 3$, x")
plt.plot(np.arange(1, 51), three_current.second_order_correlation_function[1], marker = 'x', label=r"$\Delta = 3$, y")
plt.legend()
plt.xlim((0, 11))
plt.xlabel(r"$\omega / \Omega$")
plt.ylabel(r"$g_2(0)$")
plt.savefig(PLOTTING_DIR / "g2(0).png", dpi=300)
plt.show()