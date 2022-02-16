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
        if denominator == 0:
            result[:, column] = 0
            continue

        def normalize(val):
            return (val - mean) / denominator

        result[:, column] = np.vectorize(normalize)(x)
    return result
