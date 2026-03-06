from DenisSSHModel import One_D_SSH_Model
import matplotlib.pyplot as plt
import numpy as np

numK, numT = 25, 10

plt.style.use('stylesheet.mplstyle')
plt.rcParams['text.usetex'] = False
plottingFunctions = [lambda x: np.abs(x), lambda x: x.real, lambda x: x.imag]
plottingPrefixes = ['Magnitude of', 'Real part of', 'Imaginary part of']

ssh = One_D_SSH_Model(t_points = numT)
final_corr = np.zeros((len(ssh.time_inf), len(ssh.time)), dtype=complex)

momentums = np.linspace(-np.pi, np.pi, numK)
for i, k in enumerate(momentums):
    print(f"Calculating correlation for k = {k} ({i + 1} / {momentums.size})...")
    final_corr += ssh.correlator(k, order='reverse')

    # nrows, ncols = 1, 2
    # fig, ax = plt.subplots(nrows, ncols)

    # for col in range(ncols):
    #     mesh = ax[col].pcolormesh(ssh.time,
    #                             ssh.time_inf,
    #                             plottingFunctions[col](corr),
    #                             cmap = 'bwr',
    #                             shading = 'nearest')

    #     ax[col].set_xlabel("tau")
    #     ax[col].set_ylabel("t")
    #     plt.colorbar(mesh, ax=ax[col])

    # plt.tight_layout()
    # # plt.show()
    # plt.close()

integratedcorr = ssh.integration_over_period(final_corr)

nrows, ncols = 3, 1
fig, ax = plt.subplots(nrows, ncols)

for row in range(nrows):
    ax[row].plot(ssh.time,
                 plottingFunctions[row](integratedcorr),
                 color = 'black')
    ax[row].set_xlabel("tau")
    ax[row].set_ylabel(f"{plottingPrefixes[row]} Connected Current Correlator")

plt.plot(ssh.time, integratedcorr.imag)
plt.tight_layout()
plt.show()
plt.close()

plt.figure()
mask = ssh.time >= 60
masked_time = ssh.time[mask]
freqAxis = np.fft.fftshift(np.fft.fftfreq(masked_time.size, d=np.mean(np.diff(masked_time)))) / ssh.omega
plt.semilogy(freqAxis, np.abs(np.fft.fftshift(np.fft.fft(integratedcorr[mask]))))

# Adds dashed lines at the harmonics.       
for n in range(-12, 13):
    plt.axvline(n, color='blue', linestyle='dashed')

plt.xlim(-12.5, 12.5)
plt.show()
plt.close()