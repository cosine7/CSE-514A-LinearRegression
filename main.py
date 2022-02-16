import numpy as np
import pandas as pd
from algorithms import multivariate, univariate, variance_explained


def double(n):
    return n * n


if __name__ == '__main__':
    data = pd.read_excel("data/Concrete_Data.xls", "Sheet1").to_numpy()
    training_data = data[:900, :]
    testing_data = data[900:, :]

    # mb_training = univariate.run(training_data)
    # mb_testing = univariate.run(testing_data)
    result_multi = multivariate.run(training_data)

    # y_training = training_data[:, -1]
    # y_testing = testing_data[:, -1]
    # print(variance_explained.univariate(mb_training, training_data, y_training))
    # print(variance_explained.univariate(mb_testing, testing_data, y_testing))


