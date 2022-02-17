import numpy as np


def calc(predicted, response):
    variance = np.var(response)
    mse = np.square(np.subtract(response, predicted)).mean()
    return 1 - mse / variance


def univariate(mb, data, response):
    result = []
    for column in range(data.shape[1] - 1):
        m = mb[column][0]
        b = mb[column][1]

        def predicted(n):
            return m * n + b
        result.append(calc(np.vectorize(predicted)(data[:, column]), response))
    return result


def use_matrix(mb, data, response):
    row_count, column_count = data.shape
    y = response.reshape(-1, 1)
    x = np.concatenate([data[:, :-1], np.ones(row_count).reshape(-1, 1)], axis=1)
    return calc(x @ mb, y)
