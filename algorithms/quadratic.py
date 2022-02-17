import numpy as np
import math
from algorithms import variance_explained, utility


def run(data):
    row_count, column_count = data.shape
    step_size = 0.00001
    y = data[:, -1].reshape(-1, 1)
    mb = np.zeros(45).reshape(-1, 1)
    x = utility.quad_matrix(data)
    norm_mb = math.inf
    last_loss = math.inf
    count = 0
    while norm_mb > 0.01 and count < 100000:
        loss = ((y - x @ mb) ** 2).sum() / row_count
        partial_mb = (2 * x.T @ x @ mb - 2 * x.T @ y) / row_count

        mb -= step_size * partial_mb
        if last_loss >= loss:
            step_size *= 1.01
        else:
            step_size *= 0.5
        last_loss = loss
        norm_mb = np.linalg.norm(partial_mb)
        count += 1
    return [mb, variance_explained.calc(x @ mb, y)]
