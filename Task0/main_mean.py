import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

train_dataset = pd.read_csv('train.csv', header=0)
x_train = train_dataset.iloc[:, 2:]
y_train = train_dataset.iloc[:, 1]

y_train_pred = x_train.mean(axis=1)
RMSE = mean_squared_error(y_train, y_train_pred)**0.5
print("RMSE={}".format(RMSE))

test_dataset = pd.read_csv('test.csv', header=0)
x_test = test_dataset.iloc[:, 1:]
y_test_pred = x_test.mean(axis=1)
test_dataset['y'] = y_test_pred

header = ["Id", "y"]
test_dataset.to_csv('solution.csv', columns=header, index=False)
