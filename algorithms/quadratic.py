import numpy as np
from algorithms import utility


def run(data):
    y = data[:, -1].reshape(-1, 1)
    mb = np.zeros(45).reshape(-1, 1)
    x = utility.quad_matrix(data)
    return utility.regression_use_matrix(x, y, mb)
