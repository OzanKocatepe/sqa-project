from DenisSSHModel import One_D_SSH_Model
import matplotlib.pyplot as plt
import numpy as np
import gzip
import pickle

numK, numT = 25, 21


ssh = One_D_SSH_Model(t_points = numT)
final_corr = np.zeros((len(ssh.time_inf), len(ssh.time)), dtype=complex)

momentums = np.linspace(-np.pi, np.pi, numK)
for i, k in enumerate(momentums):
    print(f"Calculating correlation for k = {k} ({i + 1} / {momentums.size})...")
    final_corr += ssh.correlator(k, order='reverse')

integratedcorr = ssh.integration_over_period(final_corr)

fileName = f"simulation-instances/[Denis] numK: {numK}, numT: {numT}.pkl.gz"
with gzip.open(fileName, 'wb') as file:
    pickle.dump(integratedcorr, file)
print(f"Saved integrated correlator to {fileName}.")