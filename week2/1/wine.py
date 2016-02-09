import pandas as pn
import numpy as np

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import scale

data = pn.read_csv('wine.data')
fields = [str(i) for i in xrange(2, 15)]

X = data[fields]
Y = data['1']

kf = KFold(len(data), n_folds=5, shuffle=True, random_state=42)

results = []
for k in xrange(1, 50):
  results.append((k, np.mean(cross_val_score(estimator=knn(n_neighbors=k), cv=kf, X=X, y=Y)))) 

print max(results, key=lambda x: x[1])

X = scale(X)
results = []
for k in xrange(1, 50):
  results.append((k, np.mean(cross_val_score(estimator=knn(n_neighbors=k), cv=kf, X=X, y=Y)))) 

print max(results, key=lambda x: x[1])