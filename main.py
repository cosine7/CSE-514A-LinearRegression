import numpy as np
import pandas as pd
from algorithms import multivariate, univariate, utility, variance_explained, quadratic

if __name__ == '__main__':
    data = pd.read_excel("data/Concrete_Data.xls", "Sheet1").to_numpy()
    data_normalized = utility.mean_normalization(data)
    training_data = data[:900, :]
    training_data_normalized = data_normalized[:900, :]

    testing_data = data[900:, :]
    testing_data_normalized = data_normalized[900:, :]
    testing_response = testing_data[:, -1]

    result_uni = univariate.run(training_data, "raw", training_data)
    print("uni - m")
    print(np.array(result_uni[0])[:, 0])
    print("uni - variance explained - training")
    print(result_uni[1])
    print("uni - variance explained - testing")
    print(variance_explained.univariate(result_uni[0], testing_data, testing_response))
    print("-----------------------------------------------")

    result_uni_normalized = univariate.run(training_data_normalized, "normalized", training_data)
    print("uni-normalized - m")
    print(np.array(result_uni_normalized[0])[:, 0])
    print("uni-normalized - variance explained - training")
    print(result_uni_normalized[1])
    print("uni-normalized - variance explained - testing")
    print(variance_explained.univariate(result_uni_normalized[0], testing_data_normalized, testing_response))
    print("-----------------------------------------------")

    result_multi = multivariate.run(training_data)
    print("multi - mb")
    print(result_multi[0])
    print("multi - variance explained - training")
    print(result_multi[1])
    print("multi - variance explained - testing")
    print(variance_explained.multivariate(result_multi[0], testing_data, testing_response))
    print("-----------------------------------------------")

    result_multi_normalized = multivariate.run(training_data_normalized)
    print("multi-normalized - mb")
    print(result_multi_normalized[0])
    print("multi-normalized - variance explained - training")
    print(result_multi_normalized[1])
    print("multi-normalized - variance explained - testing")
    print(variance_explained.multivariate(result_multi_normalized[0], testing_data_normalized, testing_response))
    print("-----------------------------------------------")

    result_quad = quadratic.run(training_data)
    print("quad - mb")
    print(result_quad[0])
    print("quad - variance explained - training")
    print(result_quad[1])
    print("quad - variance explained - testing")
    print(variance_explained.multivariate(result_quad[0], utility.quad_matrix(testing_data), testing_response))
    print("-----------------------------------------------")

    result_quad_normalized = quadratic.run(training_data_normalized)
    print("quad-normalized - mb")
    print(result_quad_normalized[0])
    print("quad-normalized - variance explained - training")
    print(result_quad_normalized[1])
    print("quad-normalized - variance explained - testing")
    print(variance_explained.multivariate(
        result_quad_normalized[0], utility.quad_matrix(testing_data_normalized), testing_response))
    print("-----------------------------------------------")
    