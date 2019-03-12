
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random_forest


# Balanced Multi Class Accuracy
def bmac_score(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

bmac_scorer = make_scorer(bmac_score, greater_is_better=True)

print('{}: Reading train dataset...'.format(datetime.now().strftime("%H:%M:%S")))
# read train data
x_train_dataset = pd.read_csv('X_train.csv', header=0)
y_train_dataset = pd.read_csv('y_train.csv', header=0)

test_dataset = pd.read_csv('X_test.csv', header=0)
x_test = test_dataset.iloc[:, 1:]
x_train_data = x_train_dataset.iloc[:, 1:]
y_train_data = y_train_dataset.iloc[:, 1].values

# split train/validation data
X_train, X_val, y_train, y_val = train_test_split(x_train_data, y_train_data, test_size=0.33, random_state=123)

# scaler
min_max_scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
scaler = min_max_scaler
x_train_data = scaler.fit_transform(x_train_data)


y_test_pred, y_valid_pred = random_forest.predict(X_train, y_train, X_val, y_val, x_test, bmac_scorer, bmac_score)

print('{}: * k-nearest neighbour classifier ...'.format(datetime.now().strftime("%H:%M:%S")))
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto',n_neighbors=5,weights='uniform') # try different n_neighbors
print('{}: Correct predications: {}/{}'.format(datetime.now().strftime("%H:%M:%S"), np.sum(knn.predict(X_val) == y_val), len(y_val)))
