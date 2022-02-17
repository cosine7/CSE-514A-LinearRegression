import numpy as np
import math
from algorithms import variance_explained


def regression_use_matrix(x, y, mb):
    row_count, column_count = x.shape
    norm_mb = math.inf
    last_loss = math.inf
    step_size = 0.00001
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
        # print(f"loss:{loss}, partial_mb:{partial_mb}")
    return [mb, variance_explained.calc(x @ mb, y)]


def mean_normalization(data):
    row_count, column_count = data.shape
    result = np.array(data)
    for column in range(column_count - 1):
        x = data[:, column]
        mean = x.mean()
        max_ = x.max()
        min_ = x.min()
        denominator = max_ - min_

        def normalize(val):
            return (val - mean) / denominator

        result[:, column] = np.vectorize(normalize)(x)
    return result


def quad_matrix(data):
    row_count, column_count = data.shape
    x = np.ones(45 * row_count).reshape(row_count, 45)
    column = column_count - 1
    for i in range(column_count - 1):
        x[:, i] = data[:, i]
        for j in range(i, column_count - 1):
            x[:, column] = data[:, i] * data[:, j]
            column += 1
    return x
