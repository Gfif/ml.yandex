import pandas as pn
import numpy as np

import sklearn.datasets
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor as knr
from sklearn.preprocessing import scale

data = sklearn.datasets.load_boston()

X = scale(data.data)
Y = data.target

kf = KFold(len(X), n_folds=5, shuffle=True, random_state=42)

results = []
for p in np.linspace(1, 10, 200):
	results.append((p, np.mean( \
		cross_val_score(estimator=knr(n_neighbors=5, weights='distance', metric='minkowski', p=p), cv=kf, X=X, y=Y) \
	)))

print max(results, key=lambda x: x[1])