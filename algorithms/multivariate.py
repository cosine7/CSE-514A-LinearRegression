import numpy as np
from algorithms import utility


def run(data):
    row_count, column_count = data.shape
    y = data[:, -1].reshape(-1, 1)
    x = np.concatenate([data[:, :-1], np.ones(row_count).reshape(-1, 1)], axis=1)
    mb = np.zeros(column_count).reshape(-1, 1)
    return utility.regression_use_matrix(x, y, mb)
