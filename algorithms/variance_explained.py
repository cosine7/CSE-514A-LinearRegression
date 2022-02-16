import numpy as np


def calc(predicted, response):
    variance = np.var(response)
    mse = np.square(np.subtract(response, predicted)).mean()
    return 1 - mse / variance
