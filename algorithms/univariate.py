import math
import numpy as np
from algorithms import variance_explained
import matplotlib.pyplot as plt


def run(data, data_type, raw):
    y = data[:, -1]
    row_count, column_count = data.shape
    result = [[], []]
    for column in range(column_count - 1):
        m = 10
        b = 10
        x = data[:, column]

        partial_m = 10
        partial_b = 10
        last_loss = math.inf
        count = 0
        step_size = 0.00001
        while abs(partial_m) + abs(partial_b) > 0.01 and count < 100000:
            loss = ((y - x * m - b) ** 2).sum() / row_count
            partial_m = (-2 * x * (y - m * x - b)).sum() / row_count
            partial_b = (-2 * (y - m * x - b)).sum() / row_count

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

        predicted = np.vectorize(predicted)(x)
        plt.scatter(raw[:, column], y)
        plt.xlabel("predictor variable")
        plt.ylabel("response variable")
        plt.plot(raw[:, column], predicted, color="yellow")
        plt.savefig(f"results/univariate_{data_type}/{column}")
        plt.clf()
        result[0].append([m, b])
        result[1].append(variance_explained.calc(predicted, y))
    return result
