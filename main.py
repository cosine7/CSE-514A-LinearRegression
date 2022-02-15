import pandas as pd
from algorithms import multivariate, univariate

if __name__ == '__main__':
    data = pd.read_excel("data/Concrete_Data.xls", "Sheet1").to_numpy()
    training_data = data[:900, :]
    testing_data = data[900:, :]

    univariate.run(training_data)
    multivariate.run(training_data)


