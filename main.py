import pandas as pd
from algorithms import multivariate, univariate, utility, variance_explained

if __name__ == '__main__':
    data = pd.read_excel("data/Concrete_Data.xls", "Sheet1").to_numpy()
    training_data = data[:900, :]
    training_data_normalized = utility.mean_normalization(training_data)

    testing_data = data[900:, :]
    testing_data_normalized = utility.mean_normalization(testing_data)
    testing_response = testing_data[:, -1]

    result_uni = univariate.run(training_data, "raw", training_data)
    print(result_uni[1])
    print(variance_explained.univariate(result_uni[0], testing_data, testing_response))

    result_uni_normalized = univariate.run(training_data_normalized, "normalized", training_data)
    print(result_uni_normalized[1])
    print(variance_explained.univariate(result_uni_normalized[0], testing_data_normalized, testing_response))

    result_multi = multivariate.run(training_data)
    print(result_multi[1])
    print(variance_explained.multivariate(result_multi[0], testing_data, testing_response))

    result_multi_normalized = multivariate.run(training_data_normalized)
    print(result_multi_normalized[1])
    print(variance_explained.multivariate(result_multi_normalized[0], testing_data_normalized, testing_response))
