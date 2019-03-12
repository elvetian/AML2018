import pandas as pd
from sklearn import svm, linear_model
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def rmse(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

def r2(y, y_pred):
    return r2_score(y, y_pred)


x_train_dataset = pd.read_csv('X_train.csv', header=0)
y_train_dataset = pd.read_csv('y_train.csv', header=0)
x_train_data = x_train_dataset.iloc[:, 1:]
y_train_data = y_train_dataset.iloc[:, 1].values

#class_count = y_train_data.value_counts()

# plt.plot(x_train['x4'], y_train,'.')
# plt.show()

imp = SimpleImputer(strategy="mean")
x_train_imp = imp.fit_transform(x_train_data)

scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
x_train_imp = scaler.fit_transform(x_train_imp)
X_train, X_test, y_train, y_test = train_test_split(x_train_imp, y_train_data, test_size=0.2, random_state=123)

#cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# parameters
kernel = 'rbf'
Cs = [2e1, 2e3, 2e5]
gammas = [2e-8, 2e-7, 2e-6]
param_grid = {'C': Cs, 'gamma' : gammas}
r2_scorer = make_scorer(r2, greater_is_better=True)

# Models
svr = svm.SVR(kernel=kernel)
#regr = linear_model.LinearRegression()
ridge = linear_model.Ridge (alpha = .5)
#svc = svm.SVC(gamma='scale')
svc = svm.SVC(kernel=kernel)

gscv = GridSearchCV(estimator=svr, param_grid=param_grid, scoring=r2_scorer, cv=3)
model = gscv
print("Fitting model with kernel '{}'".format(kernel))
model.fit(X_train, y_train)
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)

svr = svm.SVR(kernel='rbf', C=model.best_params_['C'], gamma=model.best_params_['gamma'])
svr.fit(X_train, y_train)

print('Predicting...')
y_test_pred = svr.predict(X_test)
r2_score = r2_score(y_test, y_test_pred)
print("R2 score={}".format(r2_score))

RMSE = mean_squared_error(y_test, y_test_pred)**0.5
print("RMSE={}".format(RMSE))



#
test_dataset = pd.read_csv('X_test.csv', header=0)

x_test = test_dataset.iloc[:, 1:]

x_test_imp = imp.fit_transform(x_test)
x_test_imp = scaler.fit_transform(x_test_imp)

y_test_pred = svr.predict(x_test_imp)


test_dataset['y'] = y_test_pred

header = ["id", "y"]
test_dataset.to_csv('solution.csv', columns=header, index=False)


print("Done")
