import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

train_dataset = pd.read_csv('train.csv', header=0)
x_train = train_dataset.iloc[:, 2:]
y_train = train_dataset.iloc[:, 1]
svc = svm.SVR(kernel='linear')

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
rmse_scorer = make_scorer(rmse, greater_is_better=False)
model = GridSearchCV(estimator=svc, param_grid=param_grid, scoring=rmse_scorer, cv=3)

model.fit(x_train, y_train)

print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(-model.best_score_)

y_train_pred = model.predict(x_train)

RMSE = mean_squared_error(y_train, y_train_pred)**0.5
print("RMSE={}".format(RMSE))

test_dataset = pd.read_csv('test.csv', header=0)
x_test = test_dataset.iloc[:, 1:]
y_test_pred = model.predict(x_test)
test_dataset['y'] = y_test_pred

header = ["Id", "y"]
test_dataset.to_csv('solution.csv', columns=header, index=False)


print("Done")
