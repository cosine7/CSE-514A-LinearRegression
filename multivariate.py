import pandas as pd
import math
import numpy as np

n = 900
data = pd.read_excel("data/Concrete_Data.xls", "Sheet1").to_numpy()[:n, :]

step_size = 0.0001
y = data[:, -1].reshape(-1, 1)
x = np.concatenate([data[:, :-1], np.ones(data.shape[0]).reshape(-1, 1)], axis=1)
mb = np.zeros(9).reshape(-1, 1)
norm_mb = math.inf
last_loss = math.inf
# WTXTX−yTX
while norm_mb > 0.01:
    loss = ((y - x @ mb) ** 2).sum() / n
    partial_mb = (2 * x.T @ x @ mb - 2 * x.T @ y) / n

    mb -= step_size * partial_mb
    if last_loss >= loss:
        step_size *= 1.01
    else:
        step_size *= 0.5
    last_loss = loss
    norm_mb = np.linalg.norm(partial_mb)
    print(f"loss:{loss}, partial_m:{partial_mb}")
