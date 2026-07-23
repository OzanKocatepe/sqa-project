import numpy as np
from config.paths import DATA_DIR

axes, one_model_data, one_ensemble_data = np.load(DATA_DIR / "A=0.1, D=1.0, k=21, t=6.npy", allow_pickle=True)
_, three_model_data, three_ensemble_data = np.load(DATA_DIR / "A=0.1, D=3.0, k=21, t=6.npy", allow_pickle=True)

one_second_order_current = one_model_data.second_order_current
three_second_order_current = three_model_data.second_order_current

directions = ['x', 'y']
deltas = [1.0, 3.0]
currents = [one_second_order_current, three_second_order_current]

for delta_index in range(2):
    for left_operator in range(2):
        for right_operator in range(2):
            np.savetxt(DATA_DIR / f"<j{directions[left_operator]}(t + tau) j{directions[right_operator]}(t)>, delta={deltas[delta_index]}.txt",
                       currents[delta_index][left_operator, right_operator, :, :])