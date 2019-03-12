from datetime import datetime
import numpy as np
import pandas as pd
import warnings
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import random_forest

class PipelineRFE(Pipeline):
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self


# Balanced Multi Class Accuracy
def bmac_score(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


bmac_scorer = make_scorer(bmac_score, greater_is_better=True)

print('{}: Reading train dataset...'.format(datetime.now().strftime("%H:%M:%S")))
# read train data
x_train_dataset = pd.read_csv('X_train.csv', header=0)
y_train_dataset = pd.read_csv('y_train.csv', header=0)
x_train_data = x_train_dataset.iloc[:, 1:]
y_train_data = y_train_dataset.iloc[:, 1].values

# split train/validation data
X_train, X_val, y_train, y_val = train_test_split(x_train_data, y_train_data, test_size=0.33, random_state=123)


class_count = np.unique(y_train_data, return_counts=True)
print(class_count)

# scaler
min_max_scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
scaler = min_max_scaler


# model parameters
kernel = 'linear'
Cs = [12]
gammas = [0.015]
param_grid = {'estimator__svc__C': Cs, 'estimator__svc__gamma': gammas}
cv=3

# Model
svc = svm.SVC(kernel=kernel)

clf = PipelineRFE([
    ('scale_features', scaler),
    ('svc', svc)
])


# Feature selector
rfe = RFE(clf)

# Model selection
gscv = GridSearchCV(estimator=rfe, param_grid=param_grid, scoring=bmac_scorer, cv=cv)
model = gscv
print("{}: Model selection with kernel '{}'".format(datetime.now().strftime("%H:%M:%S"), kernel))
model.fit(X_train, y_train)
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)

print("Num Features: %d") % model.n_features_
print("Selected Features: %s") % model.support_
print("Feature Ranking: %s") % model.ranking_


best_C = 12  # model.best_params_['C']
best_gamma = 0.015
# Selected Model train
print('{}: Selected model: {}, {}, cv={}'.format(datetime.now().strftime("%H:%M:%S"), kernel, "C: " + str(best_C) + ", gamma: " + str(best_gamma), cv))
model = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)
model.fit(X_train, y_train)

print('{}: Validating on validation set...'.format(datetime.now().strftime("%H:%M:%S")))
y_val_pred = model.predict(X_val)
bmac_score = bmac_score(y_val, y_val_pred)
print("BMAC score={}".format(bmac_score))

#plot_decision_regions(X_train, y_train, classifier=model)

print('{}: Predicting on test set...'.format(datetime.now().strftime("%H:%M:%S")))
test_dataset = pd.read_csv('X_test.csv', header=0)
x_test = test_dataset.iloc[:, 1:]

x_test = scaler.transform(x_test)  # use same scaler and parameters which was fitted on training data
y_test_pred = model.predict(x_test)

test_dataset['y'] = y_test_pred
header = ["id", "y"]
test_dataset.to_csv('solution.csv', columns=header, index=False)

print('Done')
