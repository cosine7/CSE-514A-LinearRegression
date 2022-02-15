import pandas as pd
import math

data = pd.read_excel("data/Concrete_Data.xls", "Sheet1")
size = len(data.columns)

y = data[data.columns[size - 1]].to_numpy()
n = 900

for column in range(size - 1):
    m = 10
    b = 10
    x = data[data.columns[column]].to_numpy()
    partial_m = 10
    partial_b = 10
    last_loss = math.inf
    count = 0
    step_size = 0.00001
    while abs(partial_m) + abs(partial_b) > 0.01 and count < 10000:
        loss = ((y - x * m - b) ** 2).sum() / n
        partial_m = (-2 * x * (y - m * x - b)).sum() / n
        partial_b = (-2 * (y - m * x - b)).sum() / n

        m -= step_size * partial_m
        b -= step_size * partial_b
        if last_loss >= loss:
            step_size *= 1.01
        else:
            step_size *= 0.5
        last_loss = loss
        # print(f"loss:{loss}, partial_m:{partial_m}, partial_b:{partial_b}")
        count += 1
    print(last_loss)
