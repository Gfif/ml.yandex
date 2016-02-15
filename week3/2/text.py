import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
            )

vect = TfidfVectorizer()
X = vect.fit_transform(newsgroups.data)
y = newsgroups.target

clf = SVC(kernel='linear', random_state=241)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

best_C = max(gs.grid_scores_, key=lambda x: x.mean_validation_score).parameters['C']

clf =  SVC(kernel='linear', random_state=241, C=best_C)

clf = clf.fit(X, y)

indeces = pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index

res = []
for i in indeces:
	res.append(vect.get_feature_names()[i])

print " ".join(sorted(res))