import math
import matplotlib.pyplot as plt
import numpy as np
from algorithms import variance_explained


def run(data):
    y = data[:, -1]
    row_count, column_count = data.shape
    result = []
    for column in range(column_count - 1):
        m = 10
        b = 10
        x = data[:, column]
        mean = x.mean()
        max_ = x.max()
        min_ = x.min()

        def mean_normalization(val):
            return (val - mean) / (max_ - min_)
        normalized_x = np.vectorize(mean_normalization)(x)

        partial_m = 10
        partial_b = 10
        last_loss = math.inf
        count = 0
        step_size = 0.00001
        while abs(partial_m) + abs(partial_b) > 0.01 and count < 100000:
            loss = ((y - normalized_x * m - b) ** 2).sum() / row_count
            partial_m = (-2 * normalized_x * (y - m * normalized_x - b)).sum() / row_count
            partial_b = (-2 * (y - m * normalized_x - b)).sum() / row_count

            m -= step_size * partial_m
            b -= step_size * partial_b
            if last_loss >= loss:
                step_size *= 1.01
            else:
                step_size *= 0.5
            last_loss = loss
            # print(f"loss:{loss}, partial_m:{partial_m}, partial_b:{partial_b}")
            count += 1

        # print(last_loss)

        def predicted(n):
            return m * n + b

        predicted = np.vectorize(predicted)(normalized_x)
        plt.scatter(x, y)
        plt.plot(x, predicted, color="yellow")
        plt.savefig(f"results/univariate/{column}")
        plt.clf()
        result.append([m, b])
        print(variance_explained.calc(predicted, y))
    return result
