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

