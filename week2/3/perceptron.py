import numpy as np
import pandas as pn

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train = pn.read_csv('perceptron-train.csv', header=None)
test = pn.read_csv('perceptron-test.csv', header=None)

X_train = train[[1, 2]]
X_test = test[[1, 2]]

y_train = train[0]
y_test = test[0]

clf = Perceptron()
clf.fit(X_train, y_train)
accurancy = accuracy_score(y_test, clf.predict(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron()
clf.fit(X_train_scaled, y_train)
accurancy_scaled = accuracy_score(y_test, clf.predict(X_test_scaled))

print accurancy, accurancy_scaled, accurancy_scaled - accurancy
