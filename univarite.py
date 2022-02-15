import pandas as pd
import math
data = pd.read_excel("data/Concrete_Data.xls", "Sheet1")
size = len(data.columns)

step_size = 0.00001
y = data[data.columns[size - 1]]
y = y.to_numpy()
n = 900

for column in range(size-1):
    m = 10
    b = 10
    x = data[data.columns[column]]
    x = x.to_numpy()
    partial_m = 10
    partial_b = 10
    last_loss = math.inf
    while abs(partial_m) + abs(partial_b) > 0.01:
        sum_m = 0
        sum_b = 0
        loss = 0
        loss = ((y-x*m-b)**2).sum()/n
        partial_m = (-2*x*(y-m*x-b)).sum()/n
        partial_b =(-2*(y-m*x-b)).sum()/n
        # for i in range(n):
        #     sum_m += -2 * x[i] * (y[i] - m * x[i] - b)
        #     sum_b += -2 * (y[i] - m * x[i] - b)
        #     loss+=(y[i] - m * x[i] - b)**2
            # print(x[i])
        # partial_m = sum_m / n
        # partial_b = sum_b / n
        m -= step_size * partial_m
        b -= step_size * partial_b
        if last_loss>=loss:
            step_size*=1.5
        else:
            step_size*=0.5
        last_loss=loss
        # print(partial_m)
        # print(partial_b)
        # print(loss)
        print(f"loss:{loss}, partial_m:{partial_m}, partial_b:{partial_b}")
