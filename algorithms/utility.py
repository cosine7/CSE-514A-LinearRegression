import numpy as np


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
